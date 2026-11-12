#!/usr/bin/env python
"""KQ-LMPC closed-loop tracking on the Skydio X2 (MuJoCo), lemniscate, 1.4 s
horizon: the condensed QP of every control step solved on the GPU by FAME-QP
or CuPIQP, and on the CPU by DAQP (KQ-LMPC's own solver, the reference).

Self-contained: imports only fame_qp, cupiqp, daqp, mujoco and the KQ-LMPC package
kept in external/kq-lmpc-quadrotor-updated -- nothing from bench/ or other
scripts.

Problem (santoshrajkumar/kq-lmpc-quadrotor-updated, unchanged except the QP call)
    Analytical Koopman lift of the quadrotor on SE(3); for a frozen seed
    trajectory (the previous predicted trajectory, shifted -- "PPOT") the
    lifted dynamics are linear, so each control step is a convex QP.  The
    demo's reference generator, constraints, LQR fallback and MuJoCo plant
    with the X2's own rotor mixer are used as is; KqLmpcOcpSolver.solve is
    reproduced with only the daqp.solve call replaced (for the DAQP method
    the replacement calls daqp.solve exactly as KQ-LMPC does).
    Horizon HORIZON = 1.4 s split into N = HORIZON / ctrl_dt = 70 stages, so
    one stage equals one 20 ms control period (needed for the shifted warm
    start).  Condensed QP: n = 4 N = 280 inputs, rows [I; C] -- the input
    bounds (diagonal) and (N - 1) * 15 = 1035 lifted-state rows: FAME-QP's
    ``split`` structure.  The seed changes every step, so the Hessian and C
    change every step, not only q and the bounds.
    Flight: lemniscate, TSIM = 20 s, 20 ms control period.  The X2 starts ON
    the reference, at its start point with the reference velocity (the demo's
    --approach 0 --start on: no minimum-snap approach leg from the ground).

Solvers (float64, tolerance 1e-4, iteration cap 2e4)
    fame-qp   eps_abs = eps_rel = 1e-4, convergence check every 10 iterations,
              automatic mode.  Every step: Solver.update_matrices (the new H
              and C rescaled with the setup-time scaling; lazy: only the
              active rho-bank entry re-inverted, the others marked stale and
              inverted if a rho switch selects them), then update(q, l, h).
    cupiqp    CuPIQP dense, eps_abs = eps_rel = 1e-4, max 1000 iterations.
              Input bounds as variable bounds, the lifted-state rows as G.
              Every step: update() with the new P, c, G and bounds, then the
              captured solve graph.
    daqp      KQ-LMPC's own call (CPU, active set): primal_tol = dual_tol =
              1e-7, iter_limit 2000, the same all-zero ``sense`` every step
              (DAQP does not write its active set back, so every solve starts
              from an empty working set -- a cold start, as KQ-LMPC runs it).

Warm start (as random_lmpc_gpu.py and g1_balance_gpu.py)
    Before the timed loop, on the QP of step 0 and outside the timing, each
    solver is set up (FAME-QP: bank, setup test, penalty probe; CuPIQP:
    compile and graph capture) and solves that QP once from a cold start.
    Step 0 (timed) then starts from that solution unshifted.  From step 1 on,
    FAME-QP starts from the previous primal and dual solution shifted one
    stage along the horizon, last stage held (Solver.warm_start_shifted).
    CuPIQP, an interior-point method, has no starting-point input and
    computes its own initial point at every solve.  DAQP is run as KQ-LMPC
    runs it (cold; one untimed solve of the first QP too, for symmetry).

Timing
    QP time per control step = wall clock of the solver call: matrix update,
    data update, warm-start shift, solve and copying the solution to the host
    (KQ-LMPC's numpy condensing before it is not included, the same for both).
    Every step counts (step 0 included: it is a warm solve).  Deadline = the
    20 ms control period.  Each method runs in its own spawned process.

Output (``--out``, default data/kqlmpc_gpu/; <tag> = H<horizon>, e.g. H1.4)
    raw/kqlmpc_<tag>.json            settings, machine, per method the summary,
                                     every step (time, iterations, status,
                                     position error) and the flown and
                                     reference positions
    raw/run_<tag>.log                console log
    processed/summary_<tag>.csv      one row per method
    processed/steps_<tag>.csv        one row per control step
    tables/tab_kqlmpc_<tag>.tex      QP time mean +- std and tracking (booktabs)
    tables/tab_kqlmpc_<tag>.csv      the numbers in the table
    figures/fig_kqlmpc_<tag>.{pdf,png}
                                     MuJoCo frame at the end of the FAME-QP
                                     flight: the whole reference path (red,
                                     solid) and the flown path (yellow, dashed),
                                     the X2 at its final pose, with a legend

    python experiments/quadrotor/kqlmpc_gpu.py
    python experiments/quadrotor/kqlmpc_gpu.py --methods fame-qp --redo
    python experiments/quadrotor/kqlmpc_gpu.py --report-only     # csv, table, figure from saved data
"""

from __future__ import annotations

import os

# One BLAS thread for the CPU side of the loop (KQ-LMPC's numpy condensing):
# OpenBLAS's default one-thread-per-core spinning starves the GPU solvers'
# kernel launches.  Set before numpy is imported; spawned children inherit it.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import csv  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # experiments/
ROOT = os.path.dirname(EXP)                                      # repository root
sys.path.insert(0, ROOT)                       # fame_qp package
KQ = os.path.join(EXP, "external", "kq-lmpc-quadrotor-updated")
sys.path.insert(0, KQ)                         # kq_lmpc_quadrotor_2 package

HORIZON = 1.4            # s
TSIM = 20.0              # s of lemniscate tracking (the demo's default)
APPROACH = 0.0           # s of approach leg; 0: start on the reference
EPS = 1e-4               # both solvers
CAP_ITERS = 20000        # FAME-QP safety cap
CUPIQP_MAX_ITER = 1000
CHECK = {"fame-qp": 10}  # iterations between convergence checks
DAQP_SETTINGS = {"primal_tol": 1e-7, "dual_tol": 1e-7, "iter_limit": 2000}   # KQ-LMPC's
METHODS = ["fame-qp", "cupiqp", "daqp"]
NAMES = {"fame-qp": r"\textbf{FAME-QP}", "cupiqp": r"CuPIQP$^\ast$",
         "daqp": r"DAQP$^\S$ (CPU)"}
# statuses that mean "solved to the solver's own tolerance"
SOLVED = {"solved", "CUPIQP_SOLVED", "optimal", "soft_optimal"}
FIG_METHOD = "fame-qp"                   # flight shown in the figure
FIG_W, FIG_H = 960, 640                  # rendered pixels
REF_RGBA = (0.85, 0.10, 0.10, 1.0)       # reference: red, solid
TRJ_RGBA = (1.00, 0.80, 0.00, 1.0)       # flown path: yellow, dashed
DASH = 5                                 # control steps per dash (and per gap)
LOG = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOG:
        with open(LOG, "a") as f:
            f.write(s + "\n")


def load_demo():
    spec = importlib.util.spec_from_file_location(
        "skydio_x2_demo", os.path.join(KQ, "example_scripts", "skydio_x2_demo.py"))
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    return demo


# ---------------------------------------------------------------------------
# QP backends:  min 0.5 U'HU + f'U  s.t.  bl <= [I; Aqp] U <= bu  (DAQP's form)
#   __call__ returns U (None on failure) and records the timed part only
# ---------------------------------------------------------------------------
def segments(nv, n_gen, nu=4):
    """Stage layout of KQ-LMPC's QP: U = col(u_0..u_{N-1}); rows = [I; Aqp],
    the input rows by stage, then the nbx lifted-state rows of stages
    1..N-1, by stage."""
    N = nv // nu
    nbx = n_gen // (N - 1) if N > 1 else 0
    z_seg = [(0, nu, N)]
    y_seg = [(0, nu, N)] + ([(nv, nbx, N - 1)] if nbx else [])
    return z_seg, y_seg


class FameBackend:
    name = "fame-qp"

    def __init__(self):
        from fame_qp import QP, Solver
        self.QP, self.Solver = QP, Solver
        self.s = None
        self.times, self.iters, self.status, self.fails = [], [], [], 0
        self.info = {}

    def _solve(self, H, f, G, bl, bu, shift):
        s = self.s
        s.update_matrices(P=H, G=G)
        s.update(q=f, l=bl, h=bu)
        if shift:
            s.warm_start_shifted(*segments(H.shape[0], G.shape[0] - H.shape[0]))
        r = s.solve()
        x = r.z.cpu().numpy().astype(float)
        return x, r

    def __call__(self, H, f, Aqp, bu, bl):
        import torch
        G = np.vstack([np.eye(H.shape[0]), Aqp])
        first = self.s is None
        if first:                                    # untimed: setup + cold solve
            t0 = time.perf_counter()
            qp = self.QP.create(H, f, G, bl, bu, device="cuda")
            self.s = self.Solver(qp, precision="float64", eps_abs=EPS, eps_rel=EPS,
                                 max_iter=CAP_ITERS, check_interval=CHECK["fame-qp"],
                                 lazy_update=True)
            self.s.reset()
            r0 = self.s.solve()
            torch.cuda.synchronize()
            self.info = dict(mode=str(self.s.mode), structure=str(self.s.structure),
                             setup_s=float(self.s.setup_time),
                             init_s=time.perf_counter() - t0, init_iters=int(r0.iters),
                             probed_rho=float(self.s.eng.rho_value))
        t0 = time.perf_counter()
        x, r = self._solve(H, f, G, bl, bu, shift=not first)   # step 0: unshifted
        self.times.append(time.perf_counter() - t0)
        self.iters.append(int(r.iters))
        self.status.append(str(r.status))
        self.info["lazy_inversions"] = int(self.s.eng.n_lazy)
        return x if np.all(np.isfinite(x)) else None


class CupiqpBackend:
    name = "cupiqp"

    def __init__(self):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":0:0")
        import warp as wp
        from cupiqp import DenseSolver, Status
        if hasattr(wp.config, "log_level"):
            wp.config.log_level = 30
        wp.config.quiet = True
        self.wp, self.DenseSolver, self.Status = wp, DenseSolver, Status
        self.s = None
        self.times, self.iters, self.status, self.fails = [], [], [], 0
        self.info = {}

    def _arr(self, v):
        return self.wp.array(np.ascontiguousarray(np.asarray(v, float)),
                             dtype=self.wp.float64, device="cuda")

    def _data(self, H, f, Aqp, bu, bl):
        a, nv = self._arr, H.shape[0]
        return dict(P=a(0.5 * (H + H.T)), c=a(f), G=a(Aqp), h_l=a(bl[nv:]), h_u=a(bu[nv:]),
                    x_l=a(bl[:nv]), x_u=a(bu[:nv]))

    def __call__(self, H, f, Aqp, bu, bl):
        if self.s is None:                           # untimed: setup, capture, solve
            t0 = time.perf_counter()
            s = self.DenseSolver(dtype=self.wp.float64)
            s.settings.verbose = False
            s.settings.eps_abs = EPS
            s.settings.eps_rel = EPS
            s.settings.max_iter = CUPIQP_MAX_ITER
            s.setup(**self._data(H, f, Aqp, bu, bl))
            s.solve()
            s.result.x.numpy()
            self.s = s
            self.info = dict(init_s=time.perf_counter() - t0)
        t0 = time.perf_counter()
        self.s.update(**self._data(H, f, Aqp, bu, bl))
        self.s.solve()
        x = np.asarray(self.s.result.x.numpy(), float).reshape(-1)
        self.times.append(time.perf_counter() - t0)
        st = int(self.s.result.info.status.numpy()[0])
        self.iters.append(int(self.s.result.info.iter.numpy()[0]))
        self.status.append(self.Status(st).name)
        return x if np.all(np.isfinite(x)) else None


class DaqpBackend:
    """DAQP on the CPU, called exactly as KqLmpcOcpSolver.solve calls it."""
    name = "daqp"

    def __init__(self):
        import daqp
        self.daqp = daqp
        self.sense = None
        self.started = False
        self.times, self.iters, self.status, self.fails = [], [], [], 0
        self.info = dict(settings=DAQP_SETTINGS)

    def _solve(self, H, f, Aqp, bu, bl):
        U, _, flag, info = self.daqp.solve(H, f, Aqp, bu, bl, self.sense, **DAQP_SETTINGS)
        return U, int(flag), info

    def __call__(self, H, f, Aqp, bu, bl):
        if self.sense is None:
            self.sense = np.zeros(H.shape[0] + Aqp.shape[0], dtype=np.int32)
        if not self.started:                         # untimed solve of the first QP
            self._solve(H, f, Aqp, bu, bl)
            self.started = True
        t0 = time.perf_counter()
        U, flag, info = self._solve(H, f, Aqp, bu, bl)
        U = np.asarray(U, float).flatten()
        self.times.append(time.perf_counter() - t0)
        self.iters.append(int(info.get("iterations", 0)))
        self.status.append({1: "optimal", 2: "soft_optimal"}.get(flag, f"flag{flag}"))
        return U if flag > 0 and np.all(np.isfinite(U)) else None


BACKENDS = {"fame-qp": FameBackend, "cupiqp": CupiqpBackend, "daqp": DaqpBackend}


def make_solver_class(base_cls, backend):
    """KqLmpcOcpSolver with only the daqp.solve call replaced (the rest of
    KqLmpcOcpSolver.solve reproduced as is)."""

    class GpuKq(base_cls):
        def solve(self):
            t_start = time.perf_counter()
            nx, nu, N = self.nx, self.nu, self.N
            Xr = self._lift_many(self.p[:, 0:18].T)
            Ur = self.p[0:N, 18:22].reshape(-1)
            v, S = self._condense()
            t_lin = time.perf_counter()
            Sk = S[1:]
            WS = self.Q @ Sk
            WS[-1] = self.Qe @ Sk[-1]
            Ss = Sk.reshape(N * nx, self.nv)
            Ws = WS.reshape(N * nx, self.nv)
            H = 2.0 * (Ss.T @ Ws + self.blkR)
            f = 2.0 * (Ws.T @ (v[1:] - Xr[1:]).reshape(-1) - self.blkR @ Ur)
            H = 0.5 * (H + H.T)
            bupper, blower, Aqp = self._bu, self._bl, self._A
            bupper[:self.nv] = self.ubu.reshape(-1)
            blower[:self.nv] = self.lbu.reshape(-1)
            Aqp[:] = S[1:N][:, self.idxbx, :].reshape(self.n_gen, self.nv)
            offset = v[1:N][:, self.idxbx]
            bupper[self.nv:] = (self.ubx[1:N] - offset).reshape(-1)
            blower[self.nv:] = (self.lbx[1:N] - offset).reshape(-1)
            t_qp0 = time.perf_counter()
            U = backend(H, f, Aqp.copy(), bupper.copy(), blower.copy())
            t_qp1 = time.perf_counter()
            status = 0 if U is not None else -1
            if status == 0:
                self.u_traj = U.reshape(N, nu)
                self.x_traj[:] = v + S @ U
            else:
                backend.fails += 1
            self._stats = {"time_tot": time.perf_counter() - t_start,
                           "time_lin": t_lin - t_start,
                           "time_qp": t_qp1 - t_qp0, "iterations": 0, "status": status}
            return status
    return GpuKq


# ---------------------------------------------------------------------------
# one flight
# ---------------------------------------------------------------------------
def fly(method, horizon, tsim):
    demo = load_demo()
    from kq_lmpc_quadrotor_2.mpc import kq_lmpc as kq
    base = demo.build_parser().parse_args([])
    div = int(round(horizon / base.ctrl_dt))
    if abs(horizon / div - base.ctrl_dt) > 1e-9:
        raise ValueError(f"horizon {horizon} s is not a whole number of {base.ctrl_dt} s "
                         "control periods (the shifted warm start needs one stage = one period)")
    args = demo.build_parser().parse_args(
        ["--traj", "lemniscate", "--tsim", str(tsim), "--div", str(div),
         "--horizon", str(horizon), "--approach", str(APPROACH), "--start", "on",
         "--no-view", "--no-plot", "--no-gif"])
    args.gif = None
    args.center = list(demo.CENTER_IN_FLIGHT)
    be = BACKENDS[method]()
    holder = {}

    class RecordedX2(demo.SkydioX2):                 # keep the plant for its final pose
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            holder["plant"] = self
    demo.SkydioX2 = RecordedX2
    cls = make_solver_class(kq.KqLmpcOcpSolver, be)
    demo.setup_ocp_kqlmpc = lambda d, hor, koop, **kw: cls(d, hor, koop, **kw)
    with contextlib.redirect_stdout(io.StringIO()):  # the demo's own report
        X, Xd, _, _, _ = demo.run(args)
    err = np.linalg.norm(X[0:3, :] - Xd[0:3, :], axis=0)       # m, per plant sample
    k_app = min(int(round(args.approach / args.ctrl_dt)), err.size - 1)
    qt = np.asarray(be.times) * 1e3
    dl = args.ctrl_dt * 1e3
    steps = [dict(k=k, t=float((k + 1) * args.ctrl_dt), ms=float(qt[k]), iters=be.iters[k],
                  status=be.status[k], pos_err_m=float(err[k + 1]),
                  phase="approach" if k + 1 < k_app else "track")
             for k in range(len(qt))]
    nv = 4 * div
    return dict(method=method, horizon_s=horizon, N=div, n=nv,
                m=nv + (div - 1) * 15, ctrl_dt_ms=dl, steps_n=len(qt),
                approach_s=float(args.approach), tsim_s=tsim,
                lap_s=float(2 * np.pi / args.omega),   # lemniscate period
                qp_mean_ms=float(qt.mean()), qp_std_ms=float(qt.std(ddof=1)),
                qp_median_ms=float(np.median(qt)), qp_p99_ms=float(np.percentile(qt, 99)),
                qp_max_ms=float(qt.max()), deadline_miss=float(np.mean(qt > dl)),
                iters_mean=float(np.mean(be.iters)), iters_p99=float(np.percentile(be.iters, 99)),
                solved_to_tol=int(sum(s in SOLVED for s in be.status)),
                inaccurate=int(sum(s == "solved_inaccurate" for s in be.status)),
                not_solved=int(sum(s not in SOLVED and s != "solved_inaccurate"
                                   for s in be.status)),
                fallbacks=int(be.fails),
                rmse_m=float(np.sqrt(np.mean(err ** 2))),
                rmse_track_m=float(np.sqrt(np.mean(err[k_app:] ** 2))),
                max_err_track_m=float(err[k_app:].max()),
                info=be.info, step_log=steps,
                traj=dict(X=X[0:3].tolist(), Xd=Xd[0:3].tolist()),
                qpos_end=holder["plant"].data.qpos.copy().tolist())


def _child(conn, method, horizon, tsim):
    try:
        conn.send(fly(method, horizon, tsim))
    except Exception as exc:  # noqa: BLE001
        conn.send(dict(error=f"{type(exc).__name__}: {exc}"[:300],
                       trace=traceback.format_exc()[-1500:]))


def run_one(method, horizon, tsim, cap):
    ctx = mp.get_context("spawn")
    a, b = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(b, method, horizon, tsim))
    t0 = time.perf_counter()
    p.start()
    out = a.recv() if a.poll(cap) else dict(over_cap_s=cap)
    p.join(10)
    if p.is_alive():
        p.kill()
        p.join()
    out["wall_s"] = time.perf_counter() - t0
    return out


# ---------------------------------------------------------------------------
# bookkeeping and report
# ---------------------------------------------------------------------------
def machine_info():
    info = dict(platform=platform.platform(), python=sys.version.split()[0],
                numpy=np.__version__, cpu_threads=os.cpu_count())
    try:
        info["cpu_model"] = subprocess.check_output(
            "lscpu | grep 'Model name' | head -1", shell=True, text=True).split(":", 1)[1].strip()
        info["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], text=True).strip()
        info["on_ac_power"] = open(next(
            os.path.join("/sys/class/power_supply", d, "online")
            for d in os.listdir("/sys/class/power_supply") if d.startswith("A"))).read().strip() == "1"
    except Exception:
        pass
    vers = {}
    for m in ("torch", "mujoco", "warp-lang"):
        try:
            from importlib.metadata import version
            vers[m] = version(m)
        except Exception:
            pass
    info["versions"] = vers
    return info


def save(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1, default=float)
    os.replace(tmp, path)


SUMMARY_KEYS = ["horizon_s", "N", "n", "m", "steps_n", "qp_mean_ms", "qp_std_ms", "qp_median_ms",
                "qp_p99_ms", "qp_max_ms", "deadline_miss", "iters_mean", "iters_p99",
                "solved_to_tol", "inaccurate", "not_solved", "fallbacks", "rmse_m", "rmse_track_m", "max_err_track_m", "wall_s"]


def write_csvs(payload, proc, tag):
    rows, steps = [], []
    for m in METHODS:
        r = payload["results"].get(m)
        if r is None:
            continue
        rows.append(dict(method=m, status="ok" if "qp_mean_ms" in r else
                         ("over_cap" if "over_cap_s" in r else "error"),
                         **{k: r.get(k) for k in SUMMARY_KEYS}))
        for s in r.get("step_log", []):
            steps.append(dict(method=m, **s))
    for fname, data in ((f"summary_{tag}.csv", rows), (f"steps_{tag}.csv", steps)):
        if data:
            keys = list(dict.fromkeys(k for r in data for k in r))
            with open(os.path.join(proc, fname), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(data)


def _fmt(mean, std):
    """3 significant digits on the mean; std to the same decimal place."""
    dec = 0 if mean >= 100 else 1 if mean >= 10 else 2 if mean >= 1 else 3
    return f"{mean:.{dec}f}", f"{std:.{dec}f}"


def write_table(payload, tabs, tag):
    res = payload["results"]
    ok = {m: res[m] for m in METHODS if m in res and "qp_mean_ms" in res[m]}
    best = min(ok, key=lambda m: ok[m]["qp_mean_ms"]) if ok else None
    any_r = next(iter(ok.values()), {})
    N, n, mrows = any_r.get("N"), any_r.get("n"), any_r.get("m")
    L = [r"% generated by experiments/quadrotor/kqlmpc_gpu.py -- do not edit by hand",
         r"\begin{table}[t]",
         r"\centering",
         rf"\caption{{KQ-LMPC tracking a lemniscate with the Skydio X2 in MuJoCo, "
         rf"{payload['horizon_s']:g}\,s horizon ($N={N}$ stages of 20\,ms; $n={n}$, "
         rf"$m={mrows}$; Hessian and constraint matrix change every step), "
         rf"{payload['tsim_s']:g}\,s starting on the reference"
         + (rf" after a {any_r['approach_s']:g}\,s approach" if any_r.get("approach_s") else "")
         + r". QP time per control "
         r"step [ms] (matrix and data update, warm-start shift, solve, copy to host), mean "
         r"$\pm$ standard deviation over all steps; share of steps over the 20\,ms control "
         r"period; share of steps solved to the solver's tolerance (FAME-QP stops the "
         r"others when its residual stalls within $100\times$ of it); position RMSE and "
         r"largest error in the tracking phase [mm]. GPU solvers in \texttt{float64} at "
         rf"tolerance $10^{{{int(np.log10(EPS))}}}$; every solver is set up and solves the "
         r"first QP once before the flight (untimed); FAME-QP then starts each step from "
         r"the previous solution shifted one stage. $^\ast$Interior point, no warm start "
         r"(computes its own starting point every solve). $^\S$KQ-LMPC's own solver, "
         r"active set on the CPU, tolerance $10^{-7}$, cold start, as in KQ-LMPC.}",
         rf"\label{{tab:kqlmpc_{tag.replace('.', '_')}}}",
         r"\scriptsize",
         r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lrrrrrrr}",
         r"\toprule",
         r" & \multicolumn{4}{c}{QP time per step [ms]} & & \multicolumn{2}{c}{tracking [mm]} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){7-8}",
         r"Method & mean $\pm$ std & p99 & max & $>20$\,ms & to tol. & RMSE & max \\",
         r"\midrule"]
    for m in METHODS:
        r = res.get(m)
        if r is None:
            continue
        if "qp_mean_ms" not in r:
            st = "over cap" if "over_cap_s" in r else "error"
            L.append(f"{NAMES[m]} & \\multicolumn{{7}}{{c}}{{{st}}} \\\\")
            continue
        mu, sd = _fmt(r["qp_mean_ms"], r["qp_std_ms"])
        cell = rf"{{\boldmath${mu} \pm {sd}$}}" if m == best else rf"${mu} \pm {sd}$"
        sts = [x["status"] for x in r.get("step_log", [])]
        tol = 100.0 * sum(x in SOLVED for x in sts) / max(len(sts), 1)
        L.append(f"{NAMES[m]} & {cell} & {r['qp_p99_ms']:.1f} & {r['qp_max_ms']:.1f} & "
                 f"{100 * r['deadline_miss']:.0f}\\% & {tol:.0f}\\% & "
                 f"{1e3 * r['rmse_track_m']:.1f} & "
                 f"{1e3 * r['max_err_track_m']:.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    os.makedirs(tabs, exist_ok=True)
    path = os.path.join(tabs, f"tab_kqlmpc_{tag}.tex")
    with open(path, "w") as f:
        f.write("\n".join(L))
    rows = [dict(method=m, **{k: res[m].get(k) for k in SUMMARY_KEYS}) for m in ok]
    if rows:
        with open(os.path.join(tabs, f"tab_kqlmpc_{tag}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    log("wrote", path)


def _segments(scene, pts, rgba, width, on=None):
    """Capsules between consecutive points; ``on[i]`` False skips segment i
    (a dashed line)."""
    import mujoco
    n0 = scene.ngeom
    for i in range(len(pts) - 1):
        if on is not None and not on[i]:
            continue
        if scene.ngeom >= scene.maxgeom:
            break
        a, b = np.asarray(pts[i], float), np.asarray(pts[i + 1], float)
        if np.linalg.norm(b - a) < 1e-9:
            continue
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.eye(3).flatten(), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
        scene.ngeom += 1
    return scene.ngeom - n0


def write_figure(payload, figs, tag, method=FIG_METHOD):
    """MuJoCo frame at the end of the flight: whole reference (red, solid),
    flown path (yellow, dashed), the X2 at its final pose; legend added."""
    r = payload["results"].get(method, {})
    if "qpos_end" not in r:
        log(f"figure skipped: no saved {method} flight with its final pose")
        return
    os.environ.setdefault("MUJOCO_GL", "egl")            # offscreen, no display needed
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import mujoco
    with contextlib.redirect_stdout(io.StringIO()):
        demo = load_demo()
        plant = demo.SkydioX2()
    model, data = plant.model, plant.data
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, FIG_W)
    model.vis.global_.offheight = max(model.vis.global_.offheight, FIG_H)
    data.qpos[:] = r["qpos_end"]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    X = np.asarray(r["traj"]["X"]).T                     # flown positions, (steps+1, 3)
    Xd = np.asarray(r["traj"]["Xd"]).T                   # reference positions
    pts = np.vstack([X, Xd])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    aspect = FIG_W / FIG_H
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    # top view, orthographic (drone and path at the same scale): looking
    # straight down, +x to the right and +y up in the image; fovy is then the
    # visible height in metres
    half = 1.12 * max(0.5 * (hi[1] - lo[1]), 0.5 * (hi[0] - lo[0]) / aspect)
    model.vis.global_.orthographic = 1
    model.vis.global_.fovy = 2.0 * half
    cam.lookat[:] = 0.5 * (lo + hi)
    cam.distance = 4.0
    cam.azimuth, cam.elevation = 90.0, -89.99
    # dashes by time within the lap, so every lap of the figure eight puts its
    # dashes in the same places (an approach leg, if any, by time since launch)
    dt = r.get("ctrl_dt_ms", 20.0) / 1e3
    t = np.arange(len(X)) * dt
    t_app, lap = r.get("approach_s", 5.0), r.get("lap_s", 2 * np.pi / 0.6)
    tau = np.where(t >= t_app, (t - t_app) % lap, t)
    on = (np.floor(tau / (DASH * dt)) % 2) == 0
    rend = mujoco.Renderer(model, height=FIG_H, width=FIG_W, max_geom=20000)
    rend.update_scene(data, cam)
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    _segments(rend.scene, Xd, REF_RGBA, 0.007)                 # thin, solid
    _segments(rend.scene, X, TRJ_RGBA, 0.013, on=on)         # wider dashes on top
    img = rend.render().copy()
    rend.close()

    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.5, 3.5 * FIG_H / FIG_W))
    ax.imshow(img)
    ax.set_axis_off()
    name = method.replace("fame-qp", "FAME-QP").replace("cupiqp", "CuPIQP")
    handles = [Line2D([], [], color=REF_RGBA[:3], lw=1.6, ls="-", label="reference"),
               Line2D([], [], color=TRJ_RGBA[:3], lw=1.6, ls=(0, (3, 2)),
                      label=f"{name} trajectory")]
    leg = ax.legend(handles=handles, loc="upper right", fontsize=7, frameon=True,
                    framealpha=0.85, edgecolor="none", handlelength=2.4, borderpad=0.4)
    leg.get_frame().set_facecolor("white")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    os.makedirs(figs, exist_ok=True)
    out = os.path.join(figs, f"fig_kqlmpc_{tag}")
    fig.savefig(out + ".pdf", dpi=400)
    fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    log("wrote", out + ".pdf")


def report(payload, out, tag):
    proc, tabs, figs = (os.path.join(out, s) for s in ("processed", "tables", "figures"))
    for d in (proc, tabs, figs):
        os.makedirs(d, exist_ok=True)
    for name, fn in (("csv", lambda: write_csvs(payload, proc, tag)),
                     ("table", lambda: write_table(payload, tabs, tag)),
                     ("figure", lambda: write_figure(payload, figs, tag))):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001  -- the data are already saved
            log(f"{name} FAILED: {type(exc).__name__}: {exc}")


def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--horizon", type=float, default=HORIZON, help="MPC horizon [s]")
    ap.add_argument("--tsim", type=float, default=TSIM, help="tracking time [s]")
    ap.add_argument("--cap", type=float, default=3600.0, help="seconds per method")
    ap.add_argument("--out", default=os.path.join(EXP, "data", "kqlmpc_gpu"))
    ap.add_argument("--redo", action="store_true", help="rerun methods already saved")
    ap.add_argument("--report-only", action="store_true",
                    help="only write the csv files and the table from the saved data")
    args = ap.parse_args()

    raw, proc, tabs = (os.path.join(args.out, s) for s in ("raw", "processed", "tables"))
    for d in (raw, proc, tabs):
        os.makedirs(d, exist_ok=True)
    tag = f"H{args.horizon:g}"
    LOG = os.path.join(raw, f"run_{tag}.log")
    path = os.path.join(raw, f"kqlmpc_{tag}.json")
    payload = json.load(open(path)) if os.path.exists(path) else dict(results={})
    if args.report_only:
        report(payload, args.out, tag)
        return
    payload.update(kind="kqlmpc_gpu", trajectory="lemniscate", horizon_s=args.horizon,
                   tsim_s=args.tsim, eps=EPS, cap_iters=CAP_ITERS,
                   cupiqp_max_iter=CUPIQP_MAX_ITER, check_interval=CHECK,
                   warm_start="untimed setup + cold solve at step 0; step 0 unshifted; "
                              "shifted one stage from step 1 (FAME-QP); CuPIQP cold",
                   machine=machine_info())
    if not payload["machine"].get("on_ac_power", True):
        log("WARNING: on battery -- GPU and CPU are throttled, timings will be too high")
    log(f"\n##### {time.strftime('%Y-%m-%d %H:%M:%S')}  methods {args.methods}  "
        f"horizon {args.horizon:g} s  tsim {args.tsim:g} s")
    for m in args.methods:
        if m in payload["results"] and not args.redo:
            r = payload["results"][m]
            log(f"  {m:8s} (saved) " + (f"{r['qp_mean_ms']:.3f} ms mean"
                                        if "qp_mean_ms" in r else "no result"))
            continue
        log(f"  {m:8s} running ...")
        r = run_one(m, args.horizon, args.tsim, args.cap)
        r["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if "error" in r:
            log(f"  {m:8s} ERROR {r['error']}\n{r.get('trace', '')}")
        elif "over_cap_s" in r:
            log(f"  {m:8s} over cap {args.cap:.0f} s")
        else:
            log(f"  {m:8s} N {r['N']} n {r['n']} m {r['m']}  QP {r['qp_mean_ms']:.3f} "
                f"+- {r['qp_std_ms']:.3f} ms (median {r['qp_median_ms']:.3f}, p99 "
                f"{r['qp_p99_ms']:.3f}, max {r['qp_max_ms']:.3f})  >20 ms "
                f"{r['deadline_miss']:.0%}  iters {r['iters_mean']:.1f}  to tol "
                f"{r['solved_to_tol']}/{r['steps_n']}  inaccurate {r['inaccurate']}  "
                f"not solved {r['not_solved']}  fallbacks {r['fallbacks']}  RMSE track "
                f"{1e3 * r['rmse_track_m']:.1f} mm (max {1e3 * r['max_err_track_m']:.1f})  "
                f"[{r['wall_s']:.0f} s]")
        payload["results"][m] = r
        payload["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save(path, payload)
        write_csvs(payload, proc, tag)
    report(payload, args.out, tag)
    log(f"\ndone {time.strftime('%Y-%m-%d %H:%M:%S')}; data in {args.out}")


if __name__ == "__main__":
    main()
