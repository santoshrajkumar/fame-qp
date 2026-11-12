#!/usr/bin/env python
"""Table I data: FAME-QP against CPU and GPU QP solvers on random dense QPs.

Self-contained: imports only fame_qp and the solver packages (nothing from
bench/ or other scripts), and sets the CUDA library path that cuOSQP and qpax
need itself (no ``source bench/env.sh``).

Problems
    ReLU-QP's random dense QPs (``reluqp.utils.rand_qp``, same seeding):
    H = 2(R'R + I), n/4 dense equality rows, n/4 dense inequality rows
    (~30 % active).  Five timed instances per size (seeds 0-4).

Protocol (as Table I)
    Each solver sweeps its tolerance eps over 1e-3 ... 1e-8, loose to tight,
    and is reported at the first eps whose solution is within ``--target``
    (infinity norm, default 1e-3) of a Clarabel 1e-10 reference on all five
    instances.  Per instance: one untimed solve (accuracy check), warm-up
    solves, then repeated cold solves (1-30, depending on solve time), each
    timed by wall clock from Python with the solution copied to the host.
    Setup is excluded for every solver.  FAME-QP (float64) is set up -- bank
    inversion, setup test, penalty probe -- on a held-out instance of the same
    size (seed 5), and each timed instance is loaded through
    update_matrices/update, so the probe never sees a timed instance.  Each
    solver runs in its own spawned process with a wall-clock cap.

Solvers
    GPU: fame-qp (float64), reluqp (float64), cuosqp (official build, float32),
         qpax (float64), cupiqp (float64)
    CPU: osqp, piqp, proxqp, clarabel, daqp

Output (``--out``, default data/random_qp_cpu_gpu/)
    raw/random_qp_cpu_gpu.json   everything: settings, machine, and per
                                 (n, solver) the full tolerance sweep -- for
                                 each eps tried, each instance's accuracy
                                 check, and for the reported eps every timed
                                 solve (time, iterations, status, ...)
    raw/run.log                  console log
    refs/ref_n{n}.npz            Clarabel reference solutions (reused)
    processed/summary.csv        one row per (solver, n): reported eps, mean /
                                 median / min / max over instances, the five
                                 per-instance medians, iterations, setup
    processed/solves.csv         one row per timed solve

The JSON is rewritten after every (n, solver), and a rerun skips pairs that
are already done (``--redo`` to repeat them), so an interrupted run resumes.

    python experiments/random_qp/random_qp_cpu_gpu.py
    python experiments/random_qp/random_qp_cpu_gpu.py --sizes 20 50 --solvers fame-qp osqp
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time

import numpy as np

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # experiments/
ROOT = os.path.dirname(EXP)                                      # repository root
sys.path.insert(0, ROOT)                       # fame_qp package

SIZES = [20, 50, 100, 150, 200, 220, 250, 300, 350, 400, 600, 800, 1000,
         1100, 1200, 1400]
EPSS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
CHECK = 10                # FAME-QP and ReLU-QP: iterations between convergence checks
N_SEEDS = 5               # timed instances: seeds 0..4
HELD_OUT = N_SEEDS        # FAME-QP setup / penalty probe instance: seed 5
GPU = ["fame-qp", "reluqp", "cuosqp", "qpax", "cupiqp"]
CPU = ["osqp", "piqp", "proxqp", "clarabel", "daqp"]
SOLVERS = GPU + CPU
DEVICE = {**{k: "gpu" for k in GPU}, **{k: "cpu" for k in CPU}}
PRECISION = dict(cuosqp="float32")             # every other solver: float64
LOG = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOG:
        with open(LOG, "a") as f:
            f.write(s + "\n")


# ---------------------------------------------------------------------------
# environment: CUDA libraries for cuOSQP (CUDA 11.8) and JAX (pip nvidia-*)
# ---------------------------------------------------------------------------
def cuda_library_path():
    """The loader path bench/env.sh builds.  Set in this (parent) process
    before any child is spawned; spawned children start a fresh interpreter
    and inherit it, so the loader sees it there."""
    out = []
    for p in sys.path:
        nv = os.path.join(p, "nvidia")
        if os.path.isdir(nv):
            for sub in sorted(os.listdir(nv)):
                d = os.path.join(nv, sub, "lib")
                if os.path.isdir(d):
                    out.append(d)
    try:
        import importlib.util
        spec = importlib.util.find_spec("torch")
        if spec and spec.origin:
            d = os.path.join(os.path.dirname(spec.origin), "lib")
            if os.path.isdir(d):
                out.append(d)
    except Exception:
        pass
    old = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    c11 = os.path.join(EXP, "external", "cuda11", "toolkit", "lib")
    tail = [c11] if os.path.isdir(c11) else []
    local = []
    for c in (os.environ.get("CUDA_HOME", ""), os.path.join(os.path.dirname(ROOT), ".cuda")):
        if c and os.path.isdir(os.path.join(c, "lib64")):
            local.append(os.path.join(c, "lib64"))
    return ":".join(dict.fromkeys(local + out + old + tail))


def set_environment():
    os.environ["LD_LIBRARY_PATH"] = cuda_library_path()
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   # JAX shares the GPU with torch


# ---------------------------------------------------------------------------
# problems and reference
# ---------------------------------------------------------------------------
def rand_qp(n, seed):
    """reluqp.utils.rand_qp(nx=n, n_eq=n//4, n_ineq=n//4, seed), in the form
    l <= A x <= u with the n_eq equality rows first."""
    n_eq = n_ineq = int(n / 4)
    np.random.seed(seed)
    randn = np.random.randn
    H = randn(n, n)
    H = H.T @ H + np.eye(n)
    H = H + H.T
    Aeq = randn(n_eq, n)
    C = randn(n_ineq, n)
    active = randn(n_ineq) > 0.5
    mu = randn(n_eq)
    lamb = randn(n_ineq) * active
    x = randn(n)
    b = Aeq @ x
    d = C @ x - randn(n_ineq) * (~active)
    g = -H @ x - Aeq.T @ mu - C.T @ lamb
    return dict(P=H, q=g, A=np.vstack((Aeq, C)), l=np.concatenate((b, d)),
                u=np.concatenate((b, np.full(n_ineq, np.inf))), n=n, n_eq=n_eq)


def reference_solution(d, eps=1e-10):
    """Clarabel interior point at 1e-10: l <= Ax <= u as [A; -A] x <= [u; -l]
    (infinite rows dropped)."""
    import clarabel
    import scipy.sparse as sp
    A = sp.csc_matrix(d["A"])
    ru = np.isfinite(d["u"]) & (np.abs(d["u"]) < 1e20)
    rl = np.isfinite(d["l"]) & (np.abs(d["l"]) < 1e20)
    G = sp.vstack([A[ru], -A[rl]], format="csc")
    h = np.hstack([d["u"][ru], -d["l"][rl]])
    st = clarabel.DefaultSettings()
    st.verbose = False
    st.tol_gap_abs = st.tol_gap_rel = st.tol_feas = eps
    sol = clarabel.DefaultSolver(sp.csc_matrix(d["P"]), d["q"], G, h,
                                 [clarabel.NonnegativeConeT(G.shape[0])], st).solve()
    return np.array(sol.x), str(sol.status)


def references(n, refdir):
    path = os.path.join(refdir, f"ref_n{n}.npz")
    if os.path.exists(path):
        z = np.load(path)
        if int(z["n_seeds"]) == N_SEEDS:
            return [z[f"x{s}"] for s in range(N_SEEDS)], 0.0, True
    t0 = time.perf_counter()
    xs, sts = zip(*(reference_solution(rand_qp(n, s)) for s in range(N_SEEDS)))
    np.savez(path, n_seeds=N_SEEDS, statuses=np.array(sts),
             **{f"x{s}": x for s, x in enumerate(xs)})
    return list(xs), time.perf_counter() - t0, False


def split(d):
    ne = d["n_eq"]
    return ne, d["P"], d["q"], d["A"][:ne], d["l"][:ne], d["A"][ne:], d["l"][ne:], d["u"][ne:]


# ---------------------------------------------------------------------------
# solvers: make_<name>() -> (setup(d, eps), solve(state) -> (x, status, extra),
#                            cold(state), free(state) or None)
# setup is never timed; cold() restores a cold start outside the timed region.
# ---------------------------------------------------------------------------
def make_fame():
    """Setup (bank, setup test, penalty probe) on the held-out instance (seed
    5), once per (n, eps); timed instances loaded via update_matrices/update,
    keeping the probed rho, the scaling and the captured graph."""
    import torch
    from fame_qp import QP, Solver
    cache = {}

    def setup(d, eps):
        n = d["n"]
        if cache.get("key") != (n, eps):
            cache.clear()
            torch.cuda.empty_cache()
            h = rand_qp(n, HELD_OUT)
            qp = QP.create(h["P"], h["q"], h["A"], h["l"], h["u"], device="cuda")
            s = Solver(qp, precision="float64", eps_abs=eps, eps_rel=eps, max_iter=20000,
                       check_interval=CHECK, probe_rho=True)
            cache.update(key=(n, eps), s=s, setup=dict(
                setup_s=float(s.setup_time), setup_seed=HELD_OUT,
                probed_rho=float(s.eng.rho_value), gamma=float(s.gamma),
                mode=str(s.mode), structure=str(s.structure)))
        s = cache["s"]
        s.update_matrices(P=d["P"], G=d["A"])
        s.update(q=d["q"], l=d["l"], h=d["u"])
        return s

    def solve(s):
        r = s.solve()
        i = r.info
        return (r.z.cpu().numpy(), r.status,
                dict(iters=int(r.iters), rho_end=float(i["rho"]), pri_res=float(i["pri_res"]),
                     dua_res=float(i["dua_res"]), mode=str(i["mode"])))

    def cold(s):
        s.reset()                                   # zero iterate, probed rho

    def info():
        return cache.get("setup", {})
    return setup, solve, cold, None, info


def make_osqp():
    """OSQP (CPU): polishing off, warm starting on, max_iter 20000.  It keeps
    its iterate and adapted rho, so each cold solve gets a fresh setup."""
    import osqp
    import scipy.sparse as sp

    def new(st):
        p = osqp.OSQP()
        p.setup(st["P"], st["q"], st["A"], st["l"], st["u"], eps_abs=st["eps"],
                eps_rel=st["eps"], verbose=False, polishing=False, warm_starting=True,
                max_iter=20000)
        return p

    def setup(d, eps):
        st = dict(P=sp.csc_matrix(d["P"]), q=d["q"], A=sp.csc_matrix(d["A"]),
                  l=d["l"], u=d["u"], eps=eps)
        st["p"] = new(st)
        return st

    def solve(st):
        r = st["p"].solve()
        return (np.asarray(r.x, float), str(r.info.status),
                dict(iters=int(r.info.iter), rho_end=float(r.info.rho_estimate)))

    def cold(st):
        st["p"] = new(st)
    return setup, solve, cold, None, None


def make_piqp():
    """PIQP dense backend, eps_abs = eps_rel = eps; every solve() restarts."""
    import piqp

    def setup(d, eps):
        ne, P, q, A, b, G, l, u = split(d)
        s = piqp.DenseSolver()
        s.settings.verbose = False
        s.settings.eps_abs = eps
        s.settings.eps_rel = eps
        s.setup(np.asfortranarray(P), q, np.asfortranarray(A), b, np.asfortranarray(G), l, u)
        return s

    def solve(s):
        st = s.solve()
        return (np.array(s.result.x, float),
                "solved" if st == piqp.PIQP_SOLVED else str(st),
                dict(iters=int(s.result.info.iter)))
    return setup, solve, (lambda s: None), None, None


def make_proxqp():
    """ProxQP dense, eps_abs = eps (eps_rel = 0); every solve() restarts."""
    import proxsuite

    def setup(d, eps):
        ne, P, q, A, b, G, l, u = split(d)
        qp = proxsuite.proxqp.dense.QP(P.shape[0], ne, G.shape[0])
        qp.settings.eps_abs = eps
        qp.settings.verbose = False
        qp.init(P, q, A, b, G, l, u)
        return qp

    def solve(qp):
        qp.solve()
        ok = qp.results.info.status == proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED
        return (np.array(qp.results.x, float),
                "solved" if ok else str(qp.results.info.status),
                dict(iters=int(qp.results.info.iter)))
    return setup, solve, (lambda s: None), None, None


def make_clarabel():
    """Clarabel (sparse), tol_gap_abs = tol_gap_rel = tol_feas = eps; every
    solve() restarts."""
    import clarabel
    import scipy.sparse as sp

    def setup(d, eps):
        ne, P, q, A, b, G, l, u = split(d)
        Acl = sp.vstack([sp.csc_matrix(A), -sp.csc_matrix(G)], format="csc")
        st = clarabel.DefaultSettings()
        st.verbose = False
        st.tol_gap_abs = st.tol_gap_rel = st.tol_feas = eps
        return clarabel.DefaultSolver(sp.triu(sp.csc_matrix(P)).tocsc(), q, Acl,
                                      np.hstack([b, -l]),
                                      [clarabel.ZeroConeT(ne),
                                       clarabel.NonnegativeConeT(G.shape[0])], st)

    def solve(s):
        r = s.solve()
        return (np.array(r.x, float),
                "solved" if r.status == clarabel.SolverStatus.Solved else str(r.status),
                dict(iters=int(r.iterations)))
    return setup, solve, (lambda s: None), None, None


def make_daqp():
    """DAQP (dense active set), primal_tol = dual_tol = eps.  A repeated
    solve() restarts from the previous working set, so each cold solve gets a
    fresh setup (LDP transform), outside the timed region."""
    import daqp

    def new(st):
        m = daqp.Model()
        m.settings["primal_tol"] = st["eps"]
        m.settings["dual_tol"] = st["eps"]
        m.setup(*st["args"])
        return m

    def setup(d, eps):
        ne, P, q, A, b, G, l, u = split(d)
        Atot = np.vstack([G, A])
        bupper = np.hstack([np.where(np.isfinite(u), u, 1e30), b])
        blower = np.hstack([np.where(np.isfinite(l), l, -1e30), b])
        sense = np.zeros(Atot.shape[0], dtype=np.int32)
        sense[G.shape[0]:] = 5                       # equality rows
        st = dict(eps=eps, args=(P, q, Atot, bupper, blower, sense))
        st["m"] = new(st)
        return st

    def solve(st):
        x, _, flag, info = st["m"].solve()
        return (np.array(x, float), "solved" if flag > 0 else f"exitflag {flag}",
                dict(iters=int(info.get("iterations", -1))))

    def cold(st):
        st["m"] = new(st)
    return setup, solve, cold, None, None


def make_reluqp():
    """ReLU-QP (float64) with FAME-QP's check interval; cold start restores
    the zero iterate and the initial rho index."""
    import torch
    from reluqp.reluqpth import ReLU_QP

    def setup(d, eps):
        m = ReLU_QP()
        t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float64)  # noqa: E731
        m.setup(t(d["P"]), t(d["q"]), t(d["A"]), t(d["l"]), t(d["u"]), eps_abs=eps,
                check_interval=CHECK, precision=torch.float64, device=torch.device("cuda"))
        m._rho_ind0 = m.rho_ind
        return m

    def solve(m):
        r = m.solve()
        ex = dict(iters=int(r.info.iter)) if r.info.iter is not None else {}
        return r.x.detach().cpu().numpy(), str(r.info.status), ex

    def cold(m):
        m.output = torch.zeros_like(m.output)
        m.x = torch.zeros_like(m.x)
        m.z = torch.zeros_like(m.z)
        m.lam = torch.zeros_like(m.lam)
        m.rho_ind = m._rho_ind0
    return setup, solve, cold, None, None


def make_cuosqp():
    """cuOSQP, official build (float32).  Keeps iterate and rho, and one
    global CUDA handle: each cold solve destroys the old instance, then sets
    up a new one."""
    import cuosqp
    import scipy.sparse as sp

    def new(st):
        p = cuosqp.OSQP()
        p.setup(st["P"], st["q"], st["A"], st["l"], st["u"], eps_abs=st["eps"],
                eps_rel=st["eps"], verbose=False)
        return p

    def setup(d, eps):
        st = dict(P=sp.csc_matrix(d["P"]), q=np.asarray(d["q"], float),
                  A=sp.csc_matrix(d["A"]), l=np.asarray(d["l"], float),
                  u=np.asarray(d["u"], float), eps=eps)
        st["p"] = new(st)
        return st

    def solve(st):
        r = st["p"].solve()
        return np.asarray(r.x, float), str(r.info.status), dict(iters=int(r.info.iter))

    def cold(st):
        st["p"] = None
        gc.collect()
        st["p"] = new(st)

    def free(st):
        st["p"] = None
        gc.collect()
    return setup, solve, cold, free, None


def make_qpax():
    """qpax (JAX interior point, float64); jit compile on first use, not timed."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import qpax
    jitted = {}

    def setup(d, eps):
        ne = d["n_eq"]
        args = tuple(jnp.asarray(v, dtype=jnp.float64) for v in
                     (d["P"], d["q"], d["A"][:ne], d["l"][:ne], -d["A"][ne:], -d["l"][ne:]))
        key = (d["n"], ne, eps)
        if key not in jitted:
            jitted[key] = jax.jit(lambda *a: qpax.solve_qp(*a, solver_tol=eps))
        f = jitted[key]
        jax.block_until_ready(f(*args))
        return dict(f=f, args=args)

    def solve(st):
        x, _, _, _, conv, it = st["f"](*st["args"])
        return (np.asarray(x, float), "solved" if bool(conv) else "not_converged",
                dict(iters=int(it)))
    return setup, solve, (lambda st: None), None, None


def make_cupiqp():
    """CuPIQP (PIQP on the GPU, dense, float64, whole solve as one CUDA graph).
    Setup, compilation and graph capture are not timed.  CUBLAS workspace off:
    with the default workspace cuBLAS allocates inside the captured loop,
    which a conditional graph node rejects at some sizes."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":0:0")
    import warp as wp
    from cupiqp import DenseSolver, Status
    if hasattr(wp.config, "log_level"):                 # warp >= 1.10
        wp.config.log_level = 30                 # logging.WARNING
    else:
        wp.config.quiet = True
    wp.init()

    def arr(v):
        return wp.array(np.ascontiguousarray(np.asarray(v, float)), dtype=wp.float64,
                        device="cuda")

    def setup(d, eps):
        ne = d["n_eq"]
        s = DenseSolver(dtype=wp.float64)
        s.settings.verbose = False
        s.settings.eps_abs = eps
        s.settings.eps_rel = eps
        s.settings.max_iter = 1000
        s.setup(P=arr(d["P"]), c=arr(d["q"]), A=arr(d["A"][:ne]), b=arr(d["l"][:ne]),
                G=arr(d["A"][ne:]), h_l=arr(d["l"][ne:]), h_u=arr(d["u"][ne:]))
        s.solve()
        s.result.x.numpy()
        return s

    def solve(s):
        s.solve()
        x = np.asarray(s.result.x.numpy(), float).reshape(-1)
        st = Status(int(s.result.info.status.numpy()[0])).name
        return x, st, dict(iters=int(s.result.info.iter.numpy()[0]))
    return setup, solve, (lambda s: None), None, None


FACTORY = {"fame-qp": make_fame, "osqp": make_osqp, "piqp": make_piqp,
           "proxqp": make_proxqp, "clarabel": make_clarabel, "daqp": make_daqp,
           "reluqp": make_reluqp, "cuosqp": make_cuosqp, "qpax": make_qpax,
           "cupiqp": make_cupiqp}


# ---------------------------------------------------------------------------
# one solver, one size, in its own process
# ---------------------------------------------------------------------------
def n_reps(t):
    """(timed repeats, warm-up solves) from the first solve time [s]."""
    if t > 3:
        return 1, 0
    if t > 0.3:
        return 3, 1
    if t > 0.03:
        return 10, 2
    return 30, 5


def _child(conn, name, n, refs, target):
    try:
        setup, solve, cold, free, info = FACTORY[name]()
        data = [rand_qp(n, s) for s in range(N_SEEDS)]
        sweep = []
        for eps in EPSS:
            step = dict(eps=eps, passed=True, instances=[])
            for seed, (d, xr) in enumerate(zip(data, refs)):
                st = setup(d, eps)
                cold(st)
                t0 = time.perf_counter()
                x, status, extra = solve(st)
                t1 = time.perf_counter() - t0
                err = float(np.max(np.abs(x - xr))) if np.all(np.isfinite(x)) else float("inf")
                inst = dict(seed=seed, check_ms=t1 * 1e3, err=err, status=status, **extra)
                step["instances"].append(inst)
                if err > target:
                    step["passed"] = False
                    if free:
                        free(st)
                    break
                reps, warm = n_reps(t1)
                solves = []
                for k in range(warm + reps):
                    cold(st)
                    a = time.perf_counter()
                    _, s_k, ex_k = solve(st)
                    dt = time.perf_counter() - a
                    if k >= warm:
                        solves.append(dict(ms=dt * 1e3, status=s_k, **ex_k))
                inst.update(warmup=warm, reps=reps, solves=solves,
                            median_ms=float(np.median([s["ms"] for s in solves])))
                if free:
                    free(st)
            if info:
                step["setup"] = info()
            sweep.append(step)
            if step["passed"]:
                conn.send(dict(reached=True, eps=eps, sweep=sweep))
                return
        conn.send(dict(reached=False, sweep=sweep))
    except Exception as exc:  # noqa: BLE001
        import traceback
        conn.send(dict(reached=False, error=f"{type(exc).__name__}: {exc}"[:300],
                       trace=traceback.format_exc()[-1500:]))


def run_one(name, n, refs, target, cap):
    ctx = mp.get_context("spawn")
    a, b = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(b, name, n, refs, target))
    t0 = time.perf_counter()
    p.start()
    out = a.recv() if a.poll(cap) else dict(reached=False, over_cap_s=cap)
    p.join(10)
    if p.is_alive():
        p.kill()
        p.join()
    out["wall_s"] = time.perf_counter() - t0
    return out


def summarise(res):
    """Reported-eps statistics from a run_one result (None if not reached)."""
    if not res.get("reached"):
        return None
    step = res["sweep"][-1]
    med = np.array([i["median_ms"] for i in step["instances"]])
    s = dict(eps=step["eps"], mean_ms=float(med.mean()), median_ms=float(np.median(med)),
             min_ms=float(med.min()), max_ms=float(med.max()), inst_ms=med.tolist(),
             max_err=max(i["err"] for i in step["instances"]),
             statuses=sorted({i["status"] for i in step["instances"]}))
    it = [i.get("iters") for i in step["instances"]]
    if all(v is not None for v in it):
        s["iters"] = it
    if "setup" in step:
        s["setup"] = step["setup"]
    return s


# ---------------------------------------------------------------------------
# bookkeeping
# ---------------------------------------------------------------------------
def gpu_temp():
    try:
        return int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            text=True).split()[0])
    except Exception:
        return None


def machine_info():
    info = dict(platform=platform.platform(), python=sys.version.split()[0],
                numpy=np.__version__, cpu_threads=os.cpu_count())
    try:
        info["cpu_model"] = subprocess.check_output(
            "lscpu | grep 'Model name' | head -1", shell=True, text=True).split(":", 1)[1].strip()
    except Exception:
        pass
    try:
        q = "name,driver_version,memory.total"
        info["gpu"] = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], text=True).strip()
    except Exception:
        pass
    vers = {}
    for m in ["torch", "osqp", "piqp", "proxsuite", "clarabel", "daqp", "jax", "qpax",
              "warp", "scipy"]:
        try:
            from importlib.metadata import version
            vers[m] = version(m if m != "warp" else "warp-lang")
        except Exception:
            pass
    info["versions"] = vers
    return info


def save(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1, default=float)
    os.replace(tmp, path)


def write_csvs(payload, proc):
    rows, solves = [], []
    for rec in payload["results"]:
        n = rec["n"]
        for name, res in rec["solvers"].items():
            s = res.get("summary")
            base = dict(solver=name, device=DEVICE.get(name, ""),
                        precision=PRECISION.get(name, "float64"), n=n, m=rec["m"])
            if s is None:
                why = ("error" if "error" in res else
                       "over_cap" if "over_cap_s" in res else "target_not_reached")
                rows.append(dict(base, status=why))
                continue
            row = dict(base, status="ok", eps=s["eps"], mean_ms=s["mean_ms"],
                       median_ms=s["median_ms"], min_ms=s["min_ms"], max_ms=s["max_ms"],
                       max_err=s["max_err"])
            for k, v in enumerate(s["inst_ms"]):
                row[f"inst{k}_ms"] = v
            if "iters" in s:
                row["iters_mean"] = float(np.mean(s["iters"]))
            if "setup" in s:
                row["setup_s"] = s["setup"].get("setup_s")
                row["probed_rho"] = s["setup"].get("probed_rho")
                row["mode"] = s["setup"].get("mode")
            rows.append(row)
            for inst in res["sweep"][-1]["instances"]:
                for r, sv in enumerate(inst["solves"]):
                    solves.append(dict(base, eps=s["eps"], seed=inst["seed"], rep=r,
                                       ms=sv["ms"], iters=sv.get("iters", ""),
                                       status=sv["status"], err=inst["err"],
                                       rho_end=sv.get("rho_end", "")))
    for fname, data in (("summary.csv", rows), ("solves.csv", solves)):
        if not data:
            continue
        keys = list(dict.fromkeys(k for r in data for k in r))
        with open(os.path.join(proc, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(data)


def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--solvers", nargs="+", default=SOLVERS, choices=SOLVERS)
    ap.add_argument("--target", type=float, default=1e-3,
                    help="accuracy target, inf-norm distance to the reference")
    ap.add_argument("--cap", type=float, default=900.0,
                    help="seconds per solver per size (whole tolerance sweep)")
    ap.add_argument("--out", default=os.path.join(EXP, "data", "random_qp_cpu_gpu"))
    ap.add_argument("--redo", action="store_true", help="rerun (n, solver) pairs already saved")
    args = ap.parse_args()

    raw, proc, refdir = (os.path.join(args.out, s) for s in ("raw", "processed", "refs"))
    for d in (raw, proc, refdir):
        os.makedirs(d, exist_ok=True)
    LOG = os.path.join(raw, "run.log")
    path = os.path.join(raw, "random_qp_cpu_gpu.json")
    set_environment()

    payload = json.load(open(path)) if os.path.exists(path) else dict(results=[])
    payload.update(
        kind="random_qp_cpu_gpu", target=args.target, epss=EPSS, cap_s=args.cap,
        check_interval=CHECK, timed_seeds=list(range(N_SEEDS)), fame_setup_seed=HELD_OUT,
        problem="reluqp.utils.rand_qp(nx=n, n_eq=n//4, n_ineq=n//4, seed)",
        reference="Clarabel, tol 1e-10", device=DEVICE, precision_overrides=PRECISION,
        machine=machine_info())
    log(f"\n##### {time.strftime('%Y-%m-%d %H:%M:%S')}  sizes {args.sizes}  "
        f"solvers {args.solvers}  target {args.target:g}")

    for n in args.sizes:
        xs, t_ref, cached = references(n, refdir)
        rec = next((r for r in payload["results"] if r["n"] == n), None)
        if rec is None:
            rec = dict(n=n, m=2 * int(n / 4), n_eq=int(n / 4), n_ineq=int(n / 4), solvers={})
            payload["results"].append(rec)
        log(f"\n=== n={n}  m={rec['m']}  reference "
            f"{'cached' if cached else f'{t_ref:.1f} s'}")
        for name in args.solvers:
            if name in rec["solvers"] and not args.redo:
                s = rec["solvers"][name].get("summary")
                log(f"  {name:9s} (saved) " + (f"{s['mean_ms']:.3f} ms eps={s['eps']:g}"
                                                if s else "no result"))
                continue
            t0 = gpu_temp()
            res = run_one(name, n, xs, args.target, args.cap)
            res.update(gpu_temp_start_C=t0, gpu_temp_end_C=gpu_temp(),
                       timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
            res["summary"] = summarise(res)
            rec["solvers"][name] = res
            s = res["summary"]
            if s:
                ex = ""
                if "iters" in s:
                    ex += f"  it {np.mean(s['iters']):.0f}"
                if "setup" in s:
                    ex += (f"  setup {s['setup']['setup_s']:.2f} s  rho {s['setup']['probed_rho']:.0e}"
                           f"  {s['setup']['mode']}")
                log(f"  {name:9s} {s['mean_ms']:10.3f} ms (median {s['median_ms']:.3f}, "
                    f"{s['min_ms']:.3f}-{s['max_ms']:.3f})  eps={s['eps']:g}  "
                    f"err {s['max_err']:.1e}{ex}  [{res['wall_s']:.0f} s]")
            else:
                why = res.get("error") or (f"over cap {args.cap:.0f} s" if "over_cap_s" in res
                                           else "target not reached at any eps")
                log(f"  {name:9s} -- {why}")
            payload["results"].sort(key=lambda r: r["n"])
            payload["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save(path, payload)
            write_csvs(payload, proc)
    log(f"\ndone {time.strftime('%Y-%m-%d %H:%M:%S')}; data in {args.out}")


if __name__ == "__main__":
    main()
