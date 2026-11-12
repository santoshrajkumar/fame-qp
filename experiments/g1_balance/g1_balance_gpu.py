#!/usr/bin/env python
"""Constrained balance MPC on the Unitree G1 humanoid (MuJoCo): FAME-QP and
ReLU-QP on the same LQR-pre-stabilized condensed QP.

Self-contained: imports only fame_qp, reluqp, mujoco and daqp -- nothing from
bench/ or other scripts.

Model (g1_lqr_matrices.npz, made by g1_lqr_balance.py)
    A (70x70), B (70x29): discrete linearisation of the full MuJoCo step
    (dt = 2 ms, contacts and the actuators' joint PD inside) about a bent-knee
    stance; Q (CoM over the support centre + joint costs), R = I; P, K the
    infinite-horizon LQR cost-to-go and gain.  x = [dq; v] (deviation from
    the stance), u = joint-target offset from ctrl0 [rad], |u_i| <= u_max,i =
    tau_max,i / kp_i (0.01 - 0.28 rad).

OCP (horizon N, one stage = one 2 ms control period)
    min sum_{k<N} x_k'Q x_k + u_k'R u_k + x_N'P x_N,
    x_{k+1} = A x_k + B u_k, |u_k| <= u_max.

LQR pre-stabilization (both methods)
    u_k = -K x_k + v_k, decision v.  With (K, P) the LQR pair the cost equals
    x_0'P x_0 + sum_k v_k'(R + B'PB) v_k exactly, so the problem is well
    conditioned (Hessian close to the identity).
      fame-qp, reluqp  ReLU-QP's LQR-preconditioned condensed QP (port of its
                       gen_condensed_mpc_qp): n = m = 29 N, input-bound rows
                       block lower-triangular (FAME-QP's ``general``)

Plant and protocol (as bench/run_g1_balance.py)
    Nonlinear MuJoCo G1 (external/unitree_g1/scene.xml), 2 ms steps, MPC every
    step, a PUSH_N push on the torso toward +x at 0.2 s for 0.1 s, T_END (5 s) in all.
    The plant receives ctrl0 + clip(u, +-u_max).  Recovered: never fell, both
    feet flat at the end (8 contacts) and |x| < 0.05.
    Setup (untimed) for every method at the same state with inputs on their
    bounds -- 50 ms into the push under clipped LQR (at x = 0 the solution is
    v = 0 for any penalty, so a penalty chosen there says nothing).
    Accuracy gate: the solve at the setup state must give u_0 within ACC_GATE
    of exact MPC, else the method is "not converged" and the loop is skipped.
    Exact MPC = DAQP (CPU, active set, 1e-9) on the condensed QP, also at
    every ACC_EVERY-th visited state after the loop (untimed).

Solvers (tolerance-matched; each to its own stopping test, cap 1e5 iterations)
      fame-qp   float64, eps_abs = eps_rel = 1e-4, convergence check every 10
      reluqp    official package, float64, eps_abs = 1e-4, check every 25
                (its default)
    The gate solve re-solves the setup state from the setup solution as is
    (no shift).  Before the loop, untimed, every method
    is reset and solves the problem at the first loop state (standing, x = 0)
    from a cold start; step 0 starts from that solution unshifted, so no
    method pays a cold solve, or a start from the pushed setup state, inside
    the timed loop.
    Warm start (--warm shifted, default): from step 1 on, the previous step's
    solution shifted one stage, last stage repeated (FAME-QP
    warm_start_shifted, ReLU-QP's iterate shifted); the shift is timed.
    --warm last: the previous solution unshifted (bench/run_g1_balance.py).

Reported per method and N
    Timing: QP time per control step (update + solve + u to the host; mean,
    std, median, p99, max; the first loop step is reported separately, as in
    bench/run_g1_balance.py), deadline misses (> 2 ms), iterations.
    Stabilization: recovered, closed-loop cost J, fraction of steps with an
    input on its bound, horizontal CoM deviation from the standing CoM (peak,
    RMS, at the end), state deviation |x| (peak, at the end), settling time
    (last time |x| >= SETTLE_X).  Accuracy: the largest |u_0 - u_0*| against
    exact MPC.

Output (``--out``, default data/g1_balance_gpu/; <tag> = the --warm value)
    raw/g1_balance_<tag>.json       settings, machine, per (N, method) the
                                    summary and every step (t, ms, iterations,
                                    saturated, pelvis height, CoM deviation, |x|)
    raw/traj_<tag>_<method>_N<N>.npz  every step: t, qpos, qvel, u, push force,
                                    CoM (x, y), CoM deviation, |x|
    raw/run_<tag>.log               console log
    processed/summary_<tag>.csv     one row per (method, N)
    processed/steps_<tag>.csv       one row per control step
    tables/tab_g1_balance_<tag>.tex timing mean +- std and deviations (booktabs)
    tables/tab_g1_balance_<tag>.csv the numbers in the table
    figures/fig_g1_balance_<tag>.{pdf,png}  4 MuJoCo frames of FAME-QP at
                                    N = FIG_N (push, largest lean, returning,
                                    end of the run)

    python experiments/g1_balance/g1_balance_gpu.py                 # complete run + table + figure
    python experiments/g1_balance/g1_balance_gpu.py --Ns 20 --methods fame-qp
    python experiments/g1_balance/g1_balance_gpu.py --report-only   # table + figure from saved data

A saved (N, method) result is reused unless --redo is given; results saved
before trajectories were logged are run again.
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # experiments/
ROOT = os.path.dirname(EXP)                                      # repository root
EXTERNAL = os.path.join(EXP, "external")
sys.path.insert(0, ROOT)                       # fame_qp package

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "g1_lqr_matrices.npz")
SCENE = os.path.join(EXTERNAL, "unitree_g1", "scene.xml")
NS = [10, 15, 20]
METHODS = ["fame-qp", "reluqp"]
EPS = 1e-4                       # FAME-QP, ReLU-QP
CAP_ITERS = 100000
CHECK = {"fame-qp": 10, "reluqp": 25}
T_END = 5.0                      # s
PUSH_N = 80.0                    # N, 0.1 s
PUSHES = [(0.2, (1.0, 0.0, 0.0))]
T_SETUP = 0.25                   # s: setup state 50 ms into the push
PUSH_BODY = "torso_link"
FALL_Z = 0.5                     # pelvis height below which the robot has fallen
ACC_EVERY = 20                   # exact-MPC accuracy check every k-th step
ACC_GATE = 1e-3                  # |u_0 - u_0*| required at the setup state
SETTLE_X = 0.05                  # settled: |x| below this for the rest of the run
FIG_N = 10                       # horizon of the FAME-QP run shown in the figure
FIG_METHOD = "fame-qp"
LOG = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOG:
        with open(LOG, "a") as f:
            f.write(s + "\n")


# ---------------------------------------------------------------------------
# problem
# ---------------------------------------------------------------------------
def condensed_lqr(A, B, Q, R, Qf, N, K, umin, umax):
    """Port of ReLU-QP gen_condensed_mpc_qp (LQR-preconditioned condensing).
    z = [u0; x1; ...; u_{N-1}; x_N] = F zb + G x0 with zb = [v_0..v_{N-1}],
    u_k = -K x_k + v_k.  QP at state x: H, q = g_x0 x, A, l + lu_x0 x,
    u + lu_x0 x; first input u_0 = Fu0 zb + Gu0 x."""
    nx, nu = B.shape
    blk = nu + nx
    Hs = np.zeros((N * blk, N * blk))
    for k in range(N):
        Hs[k * blk:k * blk + nu, k * blk:k * blk + nu] = R
        Hs[k * blk + nu:(k + 1) * blk, k * blk + nu:(k + 1) * blk] = Qf if k == N - 1 else Q
    Acl = A - B @ K
    F = np.zeros((N * blk, N * nu))
    for k in range(N):
        F[k * blk:k * blk + nu, k * nu:(k + 1) * nu] = np.eye(nu)
        F[k * blk + nu:(k + 1) * blk, k * nu:(k + 1) * nu] = B
    Apow = [np.eye(nx)]
    for _ in range(N):
        Apow.append(Acl @ Apow[-1])
    for j in range(1, N):
        blk_u = -K @ Apow[j - 1] @ B
        blk_x = Apow[j] @ B
        for k in range(N - j):
            F[(k + j) * blk:(k + j) * blk + nu, k * nu:(k + 1) * nu] = blk_u
            F[(k + j) * blk + nu:(k + j + 1) * blk, k * nu:(k + 1) * nu] = blk_x
    G = np.vstack([np.vstack([-K @ Apow[k], Apow[k + 1]]) for k in range(N)])
    H = F.T @ Hs @ F
    H = 0.5 * (H + H.T)
    sel = np.zeros((N * nu, N * blk))
    for k in range(N):
        sel[k * nu:(k + 1) * nu, k * blk:k * blk + nu] = np.eye(nu)
    return dict(H=H, g_x0=F.T @ Hs @ G, A=sel @ F, l=np.tile(umin, N), u=np.tile(umax, N),
                lu_x0=-sel @ G, Fu0=F[:nu], Gu0=G[:nu])


def load_problem(N):
    import mujoco
    z = np.load(DATA)
    model = mujoco.MjModel.from_xml_path(SCENE)
    kp = model.actuator_gainprm[:, 0]
    tau = model.jnt_actfrcrange[model.actuator_trnid[:, 0], 1]
    umax = tau / kp
    P = dict(A=z["A"], B=z["B"], Q=z["Q"], R=z["R"], QN=z["P"], K=z["K"], N=N,
             umin=-umax, umax=umax, nx=z["A"].shape[0], nu=z["B"].shape[1])
    c = condensed_lqr(P["A"], P["B"], P["Q"], P["R"], P["QN"], N, P["K"], -umax, umax)
    return model, z, P, c


def daqp_exact(c, x):
    """Exact MPC: DAQP (active set, 1e-9) on the condensed QP; first input."""
    import daqp
    sh = c["lu_x0"] @ x
    m = c["A"].shape[0]
    dz, _, flag, _ = daqp.solve(c["H"], c["g_x0"] @ x, c["A"], c["u"] + sh, c["l"] + sh,
                                np.zeros(m, dtype=np.int32), primal_tol=1e-9, dual_tol=1e-9,
                                iter_limit=100000)
    if flag <= 0:
        raise RuntimeError(f"DAQP flag {flag}")
    return c["Fu0"] @ dz + c["Gu0"] @ x


def shift_hold(v, size):
    """Shift a stage-major vector one stage forward, repeating the last stage."""
    out = v.clone() if hasattr(v, "clone") else np.array(v, copy=True)
    if len(v) > size:
        out[:-size] = v[size:]
    return out


# ---------------------------------------------------------------------------
# methods: make(P, c, x_setup, opts) -> step(x, k) -> u (unclipped)
#   setup (untimed) at x_setup; step.iters: iterations per call
# ---------------------------------------------------------------------------
def m_fame(P, c, x0, opts):
    from fame_qp import QP, Solver
    sh = c["lu_x0"] @ x0
    qp = QP.create(c["H"], c["g_x0"] @ x0, c["A"], c["l"] + sh, c["u"] + sh, device="cuda")
    s = Solver(qp, precision="float64", eps_abs=EPS, eps_rel=EPS, max_iter=CAP_ITERS,
               check_interval=CHECK["fame-qp"])
    s.solve()
    seg = [(0, P["nu"], P["N"])]
    shifted = opts["warm"] == "shifted"
    iters = []

    def step(x, k):
        sh = c["lu_x0"] @ x
        s.update(q=c["g_x0"] @ x, l=c["l"] + sh, h=c["u"] + sh)
        if shifted and k > 0:                        # gate (k < 0), step 0: no shift
            s.warm_start_shifted(z_segments=seg, lam_segments=seg, tail="hold")
        r = s.solve()
        iters.append(int(r.iters))
        dz = r.z.cpu().numpy().astype(float)
        return c["Fu0"] @ dz + c["Gu0"] @ x

    def init(x):                                     # untimed cold solve at x
        s.reset()
        sh = c["lu_x0"] @ x
        s.update(q=c["g_x0"] @ x, l=c["l"] + sh, h=c["u"] + sh)
        s.solve()
    step.iters, step.init = iters, init
    step.info = dict(mode=str(s.mode), structure=str(s.structure),
                     setup_s=float(s.setup_time), probed_rho=float(s.eng.rho_value),
                     precision="float64", formulation="lqr-condensed")
    return step


def m_reluqp(P, c, x0, opts):
    """ReLU-QP iterates on self.output = [x; z; lam] (its warm_start() sets
    x, z, lam but not output), so the shifted start is written to output."""
    import torch
    from reluqp.reluqpth import ReLU_QP
    sh = c["lu_x0"] @ x0
    t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float64)  # noqa: E731
    m = ReLU_QP()
    t0 = time.perf_counter()
    m.setup(t(c["H"]), t(c["g_x0"] @ x0), t(c["A"]), t(c["l"] + sh), t(c["u"] + sh),
            eps_abs=EPS, max_iter=CAP_ITERS, check_interval=CHECK["reluqp"],
            device=torch.device("cuda"))
    m.solve()
    setup = time.perf_counter() - t0
    rho0 = m.rho_ind
    n, nc, nu = c["H"].shape[0], c["A"].shape[0], P["nu"]
    shifted = opts["warm"] == "shifted"
    iters = []

    def step(x, k):
        sh = c["lu_x0"] @ x
        m.update(g=c["g_x0"] @ x, l=c["l"] + sh, u=c["u"] + sh)
        if shifted and k > 0:                        # gate (k < 0), step 0: no shift
            o = m.output
            m.output = torch.cat([shift_hold(o[:n], nu), shift_hold(o[n:n + nc], nu),
                                  shift_hold(o[n + nc:], nu)]).contiguous()
        r = m.solve()
        iters.append(int(r.info.iter))
        dz = r.x.detach().cpu().numpy().astype(float)
        return c["Fu0"] @ dz + c["Gu0"] @ x

    def init(x):                                     # untimed cold solve at x
        m.output = torch.zeros_like(m.output)
        m.x = torch.zeros_like(m.x)
        m.z = torch.zeros_like(m.z)
        m.lam = torch.zeros_like(m.lam)
        m.rho_ind = rho0
        sh = c["lu_x0"] @ x
        m.update(g=c["g_x0"] @ x, l=c["l"] + sh, u=c["u"] + sh)
        m.solve()
    step.iters, step.init = iters, init
    step.info = dict(setup_s=setup, precision="float64", formulation="lqr-condensed")
    return step


MAKERS = {"fame-qp": m_fame, "reluqp": m_reluqp}


# ---------------------------------------------------------------------------
# closed loop
# ---------------------------------------------------------------------------
def push_force(t):
    for t0, d in PUSHES:
        if t0 <= t < t0 + 0.1:
            return PUSH_N * np.asarray(d)
    return np.zeros(3)


def setup_state(model, z, K, umax):
    """State T_SETUP into the pushed run under clipped LQR."""
    import mujoco
    data = mujoco.MjData(model)
    data.qpos[:] = z["qpos0"]
    data.ctrl[:] = z["ctrl0"]
    mujoco.mj_forward(model, data)
    body = model.body(PUSH_BODY).id
    dq = np.zeros(model.nv)
    while True:
        mujoco.mj_differentiatePos(model, dq, 1, z["qpos0"], data.qpos)
        x = np.hstack([dq, data.qvel])
        if data.time >= T_SETUP - 1e-9:
            return x
        data.xfrc_applied[body, :3] = push_force(data.time)
        data.ctrl[:] = z["ctrl0"] + np.clip(-K @ x, -umax, umax)
        mujoco.mj_step(model, data)


def closed_loop(method, N, opts):
    import mujoco
    model, z, P, c = load_problem(N)
    umax, Q, R, K = P["umax"], P["Q"], P["R"], P["K"]
    qpos0, ctrl0 = z["qpos0"], z["ctrl0"]
    x_setup = setup_state(model, z, K, umax)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    data.ctrl[:] = ctrl0
    mujoco.mj_forward(model, data)
    body = model.body(PUSH_BODY).id
    dt = model.opt.timestep
    step = MAKERS[method](P, c, x_setup, opts)
    base = dict(method=method, N=N, n=N * P["nu"], m=c["A"].shape[0],
                **getattr(step, "info", {}))

    # accuracy gate at the setup state (bounds active)
    t0 = time.perf_counter()
    u_chk = step(x_setup, -1)
    t_chk = time.perf_counter() - t0
    err_chk = float(np.max(np.abs(np.clip(u_chk, -umax, umax)
                                  - np.clip(daqp_exact(c, x_setup), -umax, umax))))
    gate_iters = step.iters[-1] if step.iters else None
    if not err_chk <= ACC_GATE:
        return dict(base, not_converged=True, gate_err=err_chk, gate_ms=t_chk * 1e3,
                    gate_iters=gate_iters)

    dq = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
    step.init(np.hstack([dq, data.qvel]))            # untimed: solve at the first state
    rec, xs, us = [], [], []
    tr = dict(t=[0.0], qpos=[data.qpos.copy()], qvel=[data.qvel.copy()],
              u=[np.zeros(P["nu"])], force=[np.zeros(3)])
    J, viol, sat, zmin, fell_at = 0.0, 0.0, 0, np.inf, None
    nsteps = int(round(T_END / dt))
    for k in range(nsteps):
        data.xfrc_applied[body, :3] = push_force(data.time)
        mujoco.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
        x = np.hstack([dq, data.qvel])
        t0 = time.perf_counter()
        try:
            u = step(x, k)
        except Exception as exc:  # noqa: BLE001
            return dict(base, failed=True, failed_at=float(data.time), failed_step=k,
                        error=f"{type(exc).__name__}: {exc}"[:200], steps_ok=len(rec))
        dt_k = time.perf_counter() - t0
        if not np.all(np.isfinite(u)):
            return dict(base, failed=True, failed_at=float(data.time), failed_step=k,
                        error="non-finite input", steps_ok=len(rec))
        viol = max(viol, float(np.max(np.maximum(np.abs(u) - umax, 0))))
        u = np.clip(u, -umax, umax)
        s_k = bool(np.any(np.abs(u) >= umax * (1 - 1e-3)))
        sat += int(s_k)
        J += float(x @ Q @ x + u @ R @ u)
        if k % ACC_EVERY == 0:
            xs.append(x.copy())
            us.append(u.copy())
        data.ctrl[:] = ctrl0 + u
        f_k = data.xfrc_applied[body, :3].copy()
        mujoco.mj_step(model, data)
        zmin = min(zmin, float(data.qpos[2]))
        rec.append(dict(k=k, t=float(data.time), ms=dt_k * 1e3, iters=step.iters[-1],
                        saturated=s_k, pelvis_z=float(data.qpos[2])))
        tr["t"].append(float(data.time))
        tr["qpos"].append(data.qpos.copy())
        tr["qvel"].append(data.qvel.copy())
        tr["u"].append(u.copy())
        tr["force"].append(f_k)
        if fell_at is None and data.qpos[2] < FALL_Z:
            fell_at = float(data.time)
            break

    mujoco.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
    x_end = float(np.linalg.norm(np.hstack([dq, data.qvel])))
    recovered = fell_at is None and data.ncon == 8 and x_end < 0.05
    err = 0.0
    for xa, ua in zip(xs, us):                       # untimed, after the loop
        ue = np.clip(daqp_exact(c, xa), -umax, umax)
        err = max(err, float(np.max(np.abs(ua - ue))))
    # stabilization (untimed), from every recorded state (sample i at t[i]):
    # whole-robot CoM and the state deviation |x| from the stance
    traj = {k: np.asarray(v) for k, v in tr.items()}
    kin = mujoco.MjData(model)
    com = np.empty((len(traj["t"]), 2))
    xn = np.empty(len(traj["t"]))
    for i, (q, v) in enumerate(zip(traj["qpos"], traj["qvel"])):
        kin.qpos[:] = q
        mujoco.mj_kinematics(model, kin)
        mujoco.mj_comPos(model, kin)
        com[i] = kin.subtree_com[1, :2]
        mujoco.mj_differentiatePos(model, dq, 1, qpos0, q)
        xn[i] = np.linalg.norm(np.hstack([dq, v]))
    com_dev = 1e3 * np.linalg.norm(com - com[0], axis=1)        # mm, vs standing
    traj.update(com=com, com_dev_mm=com_dev, x_norm=xn)
    for r, cd, xi in zip(rec, com_dev[1:], xn[1:]):             # rec[k] is at t[k+1]
        r["com_dev_mm"], r["x_norm"] = float(cd), float(xi)
    # settling time: first sample after the last one with |x| >= SETTLE_X
    # (0 if never above, None if still above at the end)
    above = np.flatnonzero(traj["x_norm"] >= SETTLE_X)
    if not len(above):
        settle = 0.0
    elif above[-1] + 1 < len(traj["t"]):
        settle = float(traj["t"][above[-1] + 1])
    else:
        settle = None
    qt = np.array([r["ms"] for r in rec[1:]])        # first loop step reported apart
    it = np.array([r["iters"] for r in rec[1:]])
    return dict(base, steps=len(rec), fell_at=fell_at, survived=fell_at is None,
                recovered=bool(recovered), x_end=x_end, ncon_end=int(data.ncon),
                pelvis_zmin=zmin, J=J, sat_fraction=sat / len(rec), viol=viol,
                u_err_max=err, gate_err=err_chk, gate_ms=t_chk * 1e3, gate_iters=gate_iters,
                qp_median_ms=float(np.median(qt)), qp_mean_ms=float(qt.mean()),
                qp_std_ms=float(qt.std(ddof=1)), qp_p99_ms=float(np.percentile(qt, 99)),
                qp_max_ms=float(qt.max()), first_ms=rec[0]["ms"],
                deadline_miss=float(np.mean(qt > dt * 1e3)), ctrl_dt_ms=dt * 1e3,
                iters_mean=float(it.mean()), iters_p99=float(np.percentile(it, 99)),
                com_dev_max_mm=float(com_dev.max()),
                com_dev_rms_mm=float(np.sqrt(np.mean(com_dev ** 2))),
                com_dev_end_mm=float(com_dev[-1]),
                x_norm_max=float(traj["x_norm"].max()), settle_s=settle,
                step_log=rec, traj=traj)


def _child(conn, method, N, opts):
    try:
        conn.send(closed_loop(method, N, opts))
    except Exception as exc:  # noqa: BLE001
        import traceback
        conn.send(dict(error=f"{type(exc).__name__}: {exc}"[:300],
                       trace=traceback.format_exc()[-1500:]))


def run_one(method, N, opts, cap):
    ctx = mp.get_context("spawn")
    a, b = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(b, method, N, opts))
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
# bookkeeping
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
    except Exception:
        pass
    vers = {}
    for m in ("torch", "mujoco", "daqp"):
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


SUMMARY_KEYS = ["recovered", "survived", "fell_at", "qp_median_ms", "qp_mean_ms", "qp_std_ms",
                "qp_p99_ms", "qp_max_ms", "first_ms", "deadline_miss", "J", "sat_fraction",
                "viol", "u_err_max", "iters_mean", "iters_p99", "gate_err", "gate_ms",
                "gate_iters", "com_dev_max_mm", "com_dev_rms_mm", "com_dev_end_mm",
                "x_norm_max", "x_end", "settle_s", "pelvis_zmin", "setup_s", "mode",
                "formulation"]


def write_csvs(payload, proc, tag):
    rows, steps = [], []
    for N, rec in sorted(payload["results"].items(), key=lambda kv: int(kv[0])):
        for meth, r in rec.items():
            if meth not in METHODS:                  # e.g. piMPC from older runs
                continue
            status = ("error" if "error" in r and not r.get("failed") else
                      "over_cap" if "over_cap_s" in r else
                      "not_converged" if r.get("not_converged") else
                      "failed" if r.get("failed") else "ok")
            rows.append(dict(method=meth, N=int(N), n=r.get("n"), status=status,
                             **{k: r.get(k) for k in SUMMARY_KEYS}))
            for s in r.get("step_log", []):
                steps.append(dict(method=meth, N=int(N), **s))
    for fname, data in ((f"summary_{tag}.csv", rows), (f"steps_{tag}.csv", steps)):
        if data:
            keys = list(dict.fromkeys(k for r in data for k in r))
            with open(os.path.join(proc, fname), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(data)

# ---------------------------------------------------------------------------
# report: table and figure
# ---------------------------------------------------------------------------
NAMES = {"fame-qp": r"\textbf{FAME-QP}", "reluqp": "ReLU-QP"}
FRAME_W, FRAME_H = 330, 560                   # pixels per frame (portrait crop)
CAM = dict(azimuth=135.0, elevation=-8.0, distance=2.1, lookat=(0.05, 0.0, 0.62))


def traj_path(raw, tag, method, N):
    return os.path.join(raw, f"traj_{tag}_{method}_N{N}.npz")


def _fmt(mean, std):
    """3 significant digits on the mean; std to the same decimal place."""
    dec = 0 if mean >= 100 else 1 if mean >= 10 else 2 if mean >= 1 else 3
    return f"{mean:.{dec}f}", f"{std:.{dec}f}"


def _status(r):
    if r.get("not_converged"):
        return "not conv."
    if r.get("failed"):
        return "failed"
    if "over_cap_s" in r:
        return "over cap"
    if "error" in r:
        return "error"
    return None


def write_table(payload, tabs, tag):
    """Per N and method: QP time per step mean +- std (step 0 excluded), max,
    deadline misses; CoM deviation (peak, RMS), settling time; |u0 - u0*|."""
    res = payload["results"]
    Ns = sorted(int(n) for n in res)
    rows, L = [], [
        r"% generated by experiments/g1_balance/g1_balance_gpu.py -- do not edit by hand",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Unitree G1 push recovery in MuJoCo ({PUSH_N:.0f}\,N push for 0.1\,s, "
        rf"{T_END:.0f}\,s, MPC every 2\,ms). QP time per control step [ms], mean $\pm$ "
        r"standard deviation over the steps after the first; max; share of steps over the "
        r"2\,ms deadline. CoM dev.: horizontal deviation of the whole-robot centre of mass "
        r"from standing, peak and RMS [mm]; settle: time after which $\|x\|<"
        rf"{SETTLE_X:g}$ [s]; $|u_0-u_0^\star|$: largest deviation of the first input from "
        rf"exact MPC (DAQP). Both solvers at tolerance $10^{{{int(np.log10(EPS))}}}$, "
        rf"convergence checked every {CHECK['fame-qp']} (FAME-QP) and {CHECK['reluqp']} "
        r"(ReLU-QP) iterations, float64.}",
        rf"\label{{tab:g1_balance_{tag}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{QP time per step [ms]} & \multicolumn{2}{c}{CoM dev. [mm]}"
        r" & & \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
        r"$N$ ($n$) & Method & mean $\pm$ std & max & $>2$\,ms & peak & RMS & settle [s]"
        r" & $|u_0-u_0^\star|$ \\",
        r"\midrule"]
    for i, N in enumerate(Ns):
        rec = res[str(N)]
        meths = [m for m in METHODS if m in rec]
        ok = {m: rec[m] for m in meths if _status(rec[m]) is None and "qp_mean_ms" in rec[m]}
        best = min(ok, key=lambda m: ok[m]["qp_mean_ms"]) if ok else None
        for j, m in enumerate(meths):
            r = rec[m]
            lead = f"{N} ({N * 29})" if j == 0 else ""
            st = _status(r)
            if st or "qp_mean_ms" not in r:
                L.append(f"{lead} & {NAMES[m]} & \\multicolumn{{7}}{{c}}{{{st or '--'}}} \\\\")
                rows.append(dict(N=N, method=m, status=st or "missing"))
                continue
            mu, sd = _fmt(r["qp_mean_ms"], r["qp_std_ms"])
            cell = rf"{{\boldmath${mu} \pm {sd}$}}" if m == best else rf"${mu} \pm {sd}$"
            cd = r.get("com_dev_max_mm")
            settle = r.get("settle_s")
            fell = "" if r.get("recovered") else r"$^\times$"
            L.append(
                f"{lead} & {NAMES[m]}{fell} & {cell} & {r['qp_max_ms']:.2f} & "
                f"{100 * r['deadline_miss']:.0f}\\% & "
                + (f"{cd:.1f} & {r['com_dev_rms_mm']:.1f}" if cd is not None else "-- & --")
                + " & " + ("--" if settle is None else f"{settle:.2f}")
                + f" & {r['u_err_max']:.1e} \\\\")
            rows.append(dict(N=N, n=N * 29, method=m, status="ok", recovered=r.get("recovered"),
                             **{k: r.get(k) for k in (
                                 "qp_mean_ms", "qp_std_ms", "qp_median_ms", "qp_p99_ms",
                                 "qp_max_ms", "deadline_miss", "iters_mean",
                                 "com_dev_max_mm", "com_dev_rms_mm", "com_dev_end_mm",
                                 "x_norm_max", "x_end", "settle_s", "u_err_max")}))
        if i < len(Ns) - 1:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}}",
          r"\par\smallskip{\scriptsize $^\times$did not recover (fell, or not settled at "
          rf"$t={T_END:.0f}$\,s).}}",
          r"\end{table}", ""]
    os.makedirs(tabs, exist_ok=True)
    tex = os.path.join(tabs, f"tab_g1_balance_{tag}.tex")
    with open(tex, "w") as f:
        f.write("\n".join(L))
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(tabs, f"tab_g1_balance_{tag}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    log("wrote", tex)


def pick_frames(t, com_dev):
    """Push, largest lean, returning (CoM back to half its peak), end."""
    t_push = PUSHES[0][0]
    after = np.flatnonzero(t > t_push)
    k_peak = int(after[np.argmax(com_dev[after])])
    back = np.flatnonzero((np.arange(len(t)) > k_peak) & (com_dev <= 0.5 * com_dev[k_peak]))
    k_back = int(back[0]) if len(back) else int(np.argmin(np.abs(t - (t[k_peak] + 0.5))))
    k_push = int(np.argmin(np.abs(t - (t_push + 0.05))))
    return [k_push, k_peak, k_back, len(t) - 1], ["push", "largest lean", "returning", "end"]


def render_frames(z, idx):
    os.environ.setdefault("MUJOCO_GL", "egl")           # offscreen, no display needed
    import mujoco
    model = mujoco.MjModel.from_xml_path(SCENE)
    model.vis.global_.offheight = max(model.vis.global_.offheight, FRAME_H)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, FRAME_W)
    data = mujoco.MjData(model)
    rend = mujoco.Renderer(model, FRAME_H, FRAME_W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = CAM["azimuth"], CAM["elevation"], CAM["distance"]
    cam.lookat[:] = CAM["lookat"]
    body = model.body(PUSH_BODY).id
    imgs = []
    for k in idx:
        data.qpos[:] = z["qpos"][k]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        rend.update_scene(data, cam)
        scn = rend.scene
        com = data.subtree_com[1]                        # CoM ground projection (green)
        gm = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(gm, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.022, 0, 0]),
                            np.array([com[0], com[1], 0.01]), np.eye(3).flatten(),
                            np.array([0.1, 0.9, 0.2, 1.0]))
        scn.ngeom += 1
        f = z["force"][min(k + 1, len(z["force"]) - 1)]  # force applied from t[k] on
        if np.linalg.norm(f) > 0:                        # push arrow onto the torso
            p = data.xpos[body] + np.array([0.0, 0.0, 0.05])
            a = p - 0.45 * f / np.linalg.norm(f)
            gm = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(gm, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                                np.eye(3).flatten(), np.array([0.85, 0.1, 0.1, 1.0]))
            mujoco.mjv_connector(gm, mujoco.mjtGeom.mjGEOM_ARROW, 0.025, a, p)
            scn.ngeom += 1
        imgs.append(rend.render().copy())
    rend.close()
    return imgs


def write_figure(raw, figs, tag, N=FIG_N, method=FIG_METHOD):
    """1 x 4 tile of MuJoCo frames of one recorded run."""
    path = traj_path(raw, tag, method, N)
    if not os.path.exists(path):
        log(f"figure skipped: no trajectory {os.path.relpath(path, ROOT)}")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(path)
    t, cd = z["t"], z["com_dev_mm"]
    idx, labels = pick_frames(t, cd)
    imgs = render_frames(z, idx)
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(1, 4, figsize=(3.5, FRAME_H / FRAME_W * 3.5 / 4 + 0.26))
    for a, im, lab, k in zip(ax, imgs, labels, idx):
        a.imshow(im)
        a.set_axis_off()
        a.set_title(f"{lab}\n$t={t[k]:.2f}$ s", fontsize=6, pad=2,
                    color="#2b2b29", linespacing=1.15)
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.85, wspace=0.03)
    os.makedirs(figs, exist_ok=True)
    out = os.path.join(figs, f"fig_g1_balance_{tag}")
    fig.savefig(out + ".pdf", dpi=400)
    fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    log("wrote", out + ".pdf", f"({method}, N = {N}, frames at t =",
        ", ".join(f"{t[k]:.2f}" for k in idx), "s)")


def report(payload, out, tag, fig_n):
    raw, tabs, figs = (os.path.join(out, s) for s in ("raw", "tables", "figures"))
    for name, fn in (("table", lambda: write_table(payload, tabs, tag)),
                     ("figure", lambda: write_figure(raw, figs, tag, fig_n))):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001  -- the data are already saved
            log(f"{name} FAILED: {type(exc).__name__}: {exc}")


def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--Ns", type=int, nargs="+", default=NS)
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--warm", choices=["shifted", "last"], default="shifted",
                    help="shifted: previous solution shifted one stage (default); "
                         "last: previous solution as is (bench/run_g1_balance.py)")
    ap.add_argument("--cap", type=float, default=3600.0, help="seconds per method per N")
    ap.add_argument("--out", default=os.path.join(EXP, "data", "g1_balance_gpu"))
    ap.add_argument("--redo", action="store_true", help="rerun pairs already saved")
    ap.add_argument("--report-only", action="store_true",
                    help="only write the table and the figure from the saved data")
    ap.add_argument("--fig-N", type=int, default=FIG_N,
                    help=f"horizon of the FAME-QP run in the figure (default {FIG_N})")
    args = ap.parse_args()

    raw, proc = (os.path.join(args.out, s) for s in ("raw", "processed"))
    for d in (raw, proc):
        os.makedirs(d, exist_ok=True)
    tag = args.warm
    LOG = os.path.join(raw, f"run_{tag}.log")
    path = os.path.join(raw, f"g1_balance_{tag}.json")
    opts = dict(warm=args.warm)

    payload = json.load(open(path)) if os.path.exists(path) else dict(results={})
    if args.report_only:
        report(payload, args.out, tag, args.fig_N)
        return
    payload.pop("pimpc_tol", None)                   # from runs that included piMPC
    payload.update(
        kind="g1_balance_gpu", formulation="LQR pre-stabilized condensed QP",
        eps=EPS, cap_iters=CAP_ITERS, check_interval=CHECK,
        warm=args.warm, shift_tail="hold", t_end=T_END, push_N=PUSH_N, pushes=PUSHES,
        t_setup=T_SETUP, acc_every=ACC_EVERY, acc_gate=ACC_GATE,
        data=os.path.relpath(DATA, ROOT), scene=os.path.relpath(SCENE, ROOT),
        machine=machine_info())
    log(f"\n##### {time.strftime('%Y-%m-%d %H:%M:%S')}  Ns {args.Ns}  methods "
        f"{args.methods}  warm {args.warm}  (LQR pre-stabilized condensed QP)")

    for N in args.Ns:
        rec = payload["results"].setdefault(str(N), {})
        log(f"\n=== N={N}  n={29 * N}")
        for meth in args.methods:
            stale = (meth in rec and _status(rec[meth]) is None
                     and not os.path.exists(traj_path(raw, tag, meth, N)))
            if stale:
                log(f"  {meth:8s} saved result has no trajectory; running again")
            if meth in rec and not args.redo and not stale:
                r = rec[meth]
                log(f"  {meth:8s} (saved) " + (f"{r['qp_median_ms']:.3f} ms median"
                                               if "qp_median_ms" in r else "no result"))
                continue
            log(f"  {meth:8s} running ...")
            r = run_one(meth, N, opts, args.cap)
            r["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            traj = r.pop("traj", None)
            if traj is not None:
                np.savez_compressed(traj_path(raw, tag, meth, N), **traj)
            if r.get("failed"):
                log(f"  {meth:8s} FAILED at t = {r['failed_at']:.3f} s (step "
                    f"{r['failed_step']}): {r['error']}")
            elif "error" in r:
                log(f"  {meth:8s} ERROR {r['error']}\n{r.get('trace', '')}")
            elif "over_cap_s" in r:
                log(f"  {meth:8s} over cap {args.cap:.0f} s")
            elif r.get("not_converged"):
                log(f"  {meth:8s} NOT CONVERGED at the setup state: |u-u*| "
                    f"{r['gate_err']:.1e} after {r['gate_iters']} iterations, "
                    f"{r['gate_ms']:.0f} ms")
            else:
                log(f"  {meth:8s} {'ok  ' if r['recovered'] else 'FAIL'} "
                    f"QP median {r['qp_median_ms']:7.3f} ms  p99 {r['qp_p99_ms']:7.3f}  "
                    f"max {r['qp_max_ms']:7.3f}  miss {r['deadline_miss']:.0%}  "
                    f"J {r['J']:.4f}  sat {r['sat_fraction']:.0%}  "
                    f"|u-u*| {r['u_err_max']:.1e}  iters {r['iters_mean']:.1f}  "
                    f"CoM peak {r['com_dev_max_mm']:.1f} mm  settle "
                    + ("never" if r["settle_s"] is None else f"{r['settle_s']:.2f} s")
                    + f"  {r.get('mode', '')}  [{r['wall_s']:.0f} s]")
            rec[meth] = r
            payload["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save(path, payload)
            write_csvs(payload, proc, tag)
    report(payload, args.out, tag, args.fig_N)
    log(f"\ndone {time.strftime('%Y-%m-%d %H:%M:%S')}; data in {args.out}")


if __name__ == "__main__":
    main()
