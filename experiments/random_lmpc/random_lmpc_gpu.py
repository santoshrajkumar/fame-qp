#!/usr/bin/env python
"""Closed-loop MPC on random linear systems (paper Table II): FAME-QP against
GPU QP solvers and GPU MPC methods.

Self-contained: imports only fame_qp, the solver packages, and the two MPC
solvers kept in external/ (CUDA MPC, piMPC's PyTorch port) -- nothing from
bench/ or other scripts.  Sets the CUDA library path that cuOSQP needs itself.

Problems (ReLU-QP, Bishop et al., ICRA 2024, Sec. V-B / Fig. 1)
    Random linear system with n_u in {10, 20, 30, 40, 50}, n_x = 3 n_u,
    A random with spectral radius 1.05 (seed n_u), horizon N = 40, Q = I,
    R = 0.1 I, |u| <= 1, terminal cost = infinite-horizon LQR cost-to-go.
    Initial state x0 = s * d0, d0 a random direction (seed n_u + 1), with
    the fixed scale s = 16, 8, 16, 8, 8 for n_u = 10..50: the smallest power
    of two for which exact MPC (Clarabel, 1e-9, on the banded QP) keeps an
    input bound active on >= 30 % of the closed-loop steps, found once by
    bench/run_random_mpc.py for these seeded problems -- ReLU-QP's "initial
    states large enough to activate the constraints for a significant
    portion of the trajectory".  No CPU reference is computed here; the
    fraction of steps with a bound active is reported from each timed run.

    QP solvers get ReLU-QP's LQR-preconditioned condensed QP (u_k = -K x_k +
    du_k, decision du, n = N n_u; ReLU-QP's gen_condensed_mpc_qp, ported):
    m = N n_u input-bound rows, block lower-triangular -- FAME-QP's ``general``
    structure.  MPC methods get the OCP itself.

Protocol (tolerance-matched, as ReLU-QP compares solvers: the same
convergence tolerance for every solver, no reference solution)
    50 closed-loop MPC steps, each solved to the method's own stopping test
    (safety cap 1e5 iterations).  Step 0 (--step0 init, default): before the
    loop, outside the timing, every method is reset and solves the problem at
    x0 once from a cold start; step 0 (at x0) then starts from that solution
    UNSHIFTED, so no method pays a cold solve inside the timed loop (piMPC
    shifts its stored solution inside solve(), so it is stored shifted back
    first -- pimpc_unshift; CUDA MPC uses warm = 1; qpax and CuPIQP have no
    warm start and stay cold).  --step0 cold: step 0 from a cold start (the
    earlier protocol).  From step 1 on (--start warm, default) every method
    starts from its previous solution SHIFTED one stage, the last stage
    repeated -- the receding-horizon warm start CUDA MPC (warm = 2) and piMPC
    do internally; for the condensed QP a stage is n_u entries of du and n_u
    bound rows (primal and dual).  The shift is part of the timed step for
    every method.  --start cold: every step from a cold start (paper Table II).
      fame-qp      float64, eps_abs = eps_rel = 1e-3, check interval 10,
                   automatic mode; Solver.warm_start_shifted.  Setup --
                   bank, setup test, penalty probe -- on a HELD-OUT random
                   system of the same size (seed 1000 + n_u, initial state of
                   the same scale), then the timed system is loaded through
                   update_matrices (--fame-setup heldout, default); or on the
                   timed system at x0 (--fame-setup x0, as Table II)
      reluqp       official package, float64, eps_abs = 1e-3, check interval
                   25 (its default); shifted
                   [x; z; lam] written to its iterate (its warm_start() does
                   not reach the iterate it runs on)
      cuosqp       (not run by default; --methods cuosqp) official build
                   (single precision), eps 1e-3, convergence check every 5
                   (its CUDA default); shifted primal / dual via warm_start
      qpax         (not run by default; --methods qpax) float64 interior
                   point, solver_tol 1e-3 -- no warm start
      cupiqp       CuPIQP dense, float64, eps 1e-3 -- no warm start
                   (interior point, no starting-point input)
      cudampc      CUDA MPC (baselines/cudampc), float64, 500 iterations per
                   step (no stopping test), no preconditioning; warm = 2
      pimpc        piMPC PyTorch port, float64, tol 1e-6 on its own residual,
                   its native stopping test (every iteration; it computes the
                   residual every iteration anyway, for its Nesterov restart),
                   acceleration on, no preconditioning; its own shifted warm
                   start
    Time per step = wall clock of the solve from Python, input copied to the
    host.  The plant gets u_0 clipped to the bounds.  Reported besides the
    times: closed-loop cost J = sum_t x_{t+1}' Q x_{t+1} + u_t' R u_t
    (compare methods with each other), the largest input-bound violation of
    the returned u_0 (how loose each method's solution is at the common
    tolerance), and the fraction of steps with an input bound active.
    Each (method, n_u) runs in its own spawned process with a wall-clock cap.

Output (``--out``, default data/random_lmpc_gpu/; <tag> = warm_init (default:
--start warm --step0 init), warm (--step0 cold) or cold, plus _fame-x0 with
--fame-setup x0)
    raw/random_lmpc_gpu_<tag>.json
                                 settings, machine, and per n_u the problem
                                 (x0 scale) and per
                                 method a summary plus every step: time,
                                 iterations, status, u_0, bound violation,
                                 stage cost, ||x||
    raw/run_<tag>.log            console log
    problems/problem_nu{n_u}.npz the problem: A, B, LQR gain K, Q_N, x0
    processed/summary_<tag>.csv  one row per (method, n_u)
    processed/steps_<tag>.csv    one row per MPC step

The JSON is rewritten after every (n_u, method); a rerun skips pairs already
done (``--redo`` to repeat them).

    python experiments/random_lmpc/random_lmpc_gpu.py
    python experiments/random_lmpc/random_lmpc_gpu.py --nus 10 20 --methods fame-qp reluqp
    python experiments/random_lmpc/random_lmpc_gpu.py --start cold --fame-setup x0   # Table II
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
EXTERNAL = os.path.join(EXP, "external")
BASELINES = os.path.join(EXP, "baselines")
sys.path.insert(0, ROOT)                       # fame_qp package

NUS = [10, 20, 30, 40, 50]
HORIZON = 40
STEPS = 50
CONV_TOL = 1e-3          # FAME-QP, ReLU-QP, cuOSQP, qpax, CuPIQP
PIMPC_TOL = 1e-6         # piMPC's own residual (its AFTI-16 example)
CUDAMPC_ITERS = 500      # CUDA MPC: fixed iterations per step (no stopping test)
CAP_ITERS = 100000
CHECK = {"fame-qp": 10,  # iterations between convergence checks
         "reluqp": 25,   # its default
         "pimpc": 1}     # native: its code tests every iteration
HELDOUT_SEED = 1000      # FAME-QP setup system: seed 1000 + n_u
# x0 = X0_SCALE[n_u] * d0: smallest power of two with an input bound active on
# >= 30 % of the exact-MPC closed-loop steps (bench/run_random_mpc.py's search,
# Clarabel 1e-9 on the banded QP, for these seeded problems)
X0_SCALE = {10: 16, 20: 8, 30: 16, 40: 8, 50: 8}
METHODS = ["fame-qp", "reluqp", "cupiqp", "cudampc", "pimpc"]        # default
ALL_METHODS = METHODS + ["cuosqp", "qpax"]                             # --methods
INFO = {"fame-qp": ("qp", "float64"), "reluqp": ("qp", "float64"),
        "cuosqp": ("qp", "float32"), "qpax": ("qp", "float64"),
        "cupiqp": ("qp", "float64"), "cudampc": ("mpc", "float64"),
        "pimpc": ("mpc", "float64")}
LOG = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOG:
        with open(LOG, "a") as f:
            f.write(s + "\n")


# ---------------------------------------------------------------------------
# environment: CUDA libraries for cuOSQP (CUDA 11.8) and the pip nvidia-* libs
# ---------------------------------------------------------------------------
def set_environment():
    """The loader path bench/env.sh builds, set before any child is spawned
    (a spawned child is a fresh interpreter and inherits it)."""
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
        if spec and spec.origin and os.path.isdir(os.path.join(os.path.dirname(spec.origin), "lib")):
            out.append(os.path.join(os.path.dirname(spec.origin), "lib"))
    except Exception:
        pass
    old = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    c11 = os.path.join(EXTERNAL, "cuda11", "toolkit", "lib")
    local = [os.path.join(c, "lib64") for c in
             (os.environ.get("CUDA_HOME", ""), os.path.join(os.path.dirname(ROOT), ".cuda"))
             if c and os.path.isdir(os.path.join(c, "lib64"))]
    os.environ["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(
        local + out + old + ([c11] if os.path.isdir(c11) else [])))
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


# ---------------------------------------------------------------------------
# problem
# ---------------------------------------------------------------------------
class OCP:
    """min sum x'Qx + u'Ru + x_N'Q_N x_N, x_{k+1} = A x_k + B u_k, umin <= u <= umax
    (the fields CUDA MPC's from_ocp reads)."""

    def __init__(self, A, B, Q, R, N, QN, umin, umax):
        self.A, self.B, self.Q, self.R, self.N = A, B, Q, R, N
        self.nx, self.nu = B.shape
        self.QN, self.umin, self.umax = QN, umin, umax
        self.xmin = self.xmax = None
        self.xref = np.zeros(self.nx)


def ihlqr(A, B, Q, R, Qf, iters=5000, tol=1e-10):
    """ReLU-QP examples/mpc_utils.jl ihlqr: LQR gain and cost-to-go."""
    P = Qf.copy()
    K = np.zeros((B.shape[1], A.shape[0]))
    for _ in range(iters):
        Pp = P
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P = Q + A.T @ P @ (A - B @ K)
        if np.linalg.norm(P - Pp, 2) < tol:
            break
    return K, 0.5 * (P + P.T)


def build(nu, N=HORIZON, seed=None):
    """ReLU-QP Fig. 1 system (seed nu unless given), terminal cost = LQR
    cost-to-go.  Returns (ocp, unit-scale initial direction d0 (seed + 1),
    LQR gain K)."""
    nx = 3 * nu
    seed = nu if seed is None else seed
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((nx, nx)) / np.sqrt(nx)
    A = A * (1.05 / np.max(np.abs(np.linalg.eigvals(A))))
    B = rng.standard_normal((nx, nu)) / np.sqrt(nx)
    Q, R = np.eye(nx), 0.1 * np.eye(nu)
    K, P = ihlqr(A, B, Q, R, Q)
    ocp = OCP(A, B, Q, R, N, P, -np.ones(nu), np.ones(nu))
    d0 = np.random.default_rng(seed + 1).standard_normal(nx)
    return ocp, d0, K


def condensed_lqr(ocp, K):
    """Port of ReLU-QP gen_condensed_mpc_qp (LQR-preconditioned condensing).
    z = [u0; x1; ...; u_{N-1}; x_N] = F zb + G x0 with zb = [du_0..du_{N-1}],
    u_k = -K x_k + du_k.  QP at state x: H, q = g_x0 x, A, l + lu_x0 x,
    u + lu_x0 x; first input u_0 = Fu0 zb + Gu0 x."""
    A, B, Q, R, Qf, N = ocp.A, ocp.B, ocp.Q, ocp.R, ocp.QN, ocp.N
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
    return dict(H=H, g_x0=F.T @ Hs @ G, A=sel @ F, l=np.tile(ocp.umin, N),
                u=np.tile(ocp.umax, N), lu_x0=-sel @ G, Fu0=F[:nu], Gu0=G[:nu])


# ---------------------------------------------------------------------------
# methods: make(ocp, K, x0, opts) -> step(x, u_prev, k) -> (u_0, info dict)
#   step.reset(x)  cold start, run before step 0 only (outside the timing)
#   step(...)      data update for state x, then -- for k > 0 and a warm
#                  start -- the shifted warm start, then the solve; all timed
#   step.info      setup facts
# Shifted warm start: stage k+1 of the last solution becomes stage k, the
# last stage is repeated ("hold"), as CUDA MPC (warm = 2) and piMPC do.  For
# the condensed QP a stage is n_u entries of du and n_u bound rows (z, lam).
# ---------------------------------------------------------------------------
def shift_hold(v, size):
    """Shift a stage-major vector one stage forward, repeating the last stage."""
    out = v.clone() if hasattr(v, "clone") else np.array(v, copy=True)
    if len(v) > size:
        out[:-size] = v[size:]
    return out


def m_fame(ocp, K, x0, opts):
    """Setup -- bank, setup test, penalty probe -- on a HELD-OUT random system
    of the same size (seed HELDOUT_SEED + n_u, initial state of the same
    scale; --fame-setup heldout), then the timed system's matrices are loaded
    through update_matrices, keeping the probed rho, scaling and graph.
    --fame-setup x0: set up on the timed system at x0 (paper Table II)."""
    from fame_qp import QP, Solver
    c = condensed_lqr(ocp, K)
    if opts.get("fame_setup", "heldout") == "heldout":
        ho, d_ho, K_ho = build(ocp.nu, seed=HELDOUT_SEED + ocp.nu)
        cs, xs = condensed_lqr(ho, K_ho), opts["x0_scale"] * d_ho
    else:
        cs, xs = c, x0
    sh = cs["lu_x0"] @ xs
    qp = QP.create(cs["H"], cs["g_x0"] @ xs, cs["A"], cs["l"] + sh, cs["u"] + sh, device="cuda")
    s = Solver(qp, precision="float64", eps_abs=CONV_TOL, eps_rel=CONV_TOL,
               max_iter=CAP_ITERS, check_interval=CHECK["fame-qp"])
    s.solve()
    probed = float(s.eng.rho_value)
    if cs is not c:
        s.update_matrices(P=c["H"], G=c["A"])
    seg = [(0, ocp.nu, ocp.N)]                       # du stages / bound-row stages
    warm = opts["start"] == "warm"

    def step(x, u_prev, k):
        sh = c["lu_x0"] @ x
        s.update(q=c["g_x0"] @ x, l=c["l"] + sh, h=c["u"] + sh)
        if warm and k > 0:
            s.warm_start_shifted(z_segments=seg, lam_segments=seg, tail="hold")
        r = s.solve()
        dz = r.z.cpu().numpy().astype(float)
        return c["Fu0"] @ dz + c["Gu0"] @ x, dict(iters=int(r.iters), status=str(r.status),
                                                  rho_end=float(r.info["rho"]))
    step.reset = lambda x: s.reset()

    def init(x):                                     # untimed cold solve at x
        s.reset()
        sh = c["lu_x0"] @ x
        s.update(q=c["g_x0"] @ x, l=c["l"] + sh, h=c["u"] + sh)
        s.solve()
    step.init = init
    step.info = dict(mode=str(s.mode), precision=str(s.dtype).replace("torch.", ""),
                     structure=str(s.structure), setup_s=float(s.setup_time),
                     probed_rho=probed, setup_problem=opts.get("fame_setup", "heldout"),
                     setup_seed=HELDOUT_SEED + ocp.nu if cs is not c else ocp.nu)
    if not warm:
        step.prepare = step.reset
    return step


def m_reluqp(ocp, K, x0, opts):
    """ReLU-QP iterates on self.output = [x; z; lam] (its warm_start() sets
    x, z, lam but not output), so the shifted start is written to output."""
    import torch
    from reluqp.reluqpth import ReLU_QP
    c = condensed_lqr(ocp, K)
    sh = c["lu_x0"] @ x0
    t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float64)  # noqa: E731
    m = ReLU_QP()
    m.setup(t(c["H"]), t(c["g_x0"] @ x0), t(c["A"]), t(c["l"] + sh), t(c["u"] + sh),
            eps_abs=CONV_TOL, max_iter=CAP_ITERS, check_interval=CHECK["reluqp"],
            device=torch.device("cuda"))
    m.solve()
    rho0 = m.rho_ind
    n, nc, nu = c["H"].shape[0], c["A"].shape[0], ocp.nu
    warm = opts["start"] == "warm"

    def step(x, u_prev, k):
        sh = c["lu_x0"] @ x
        m.update(g=c["g_x0"] @ x, l=c["l"] + sh, u=c["u"] + sh)
        if warm and k > 0:
            o = m.output
            m.output = torch.cat([shift_hold(o[:n], nu), shift_hold(o[n:n + nc], nu),
                                  shift_hold(o[n + nc:], nu)]).contiguous()
        r = m.solve()
        dz = r.x.detach().cpu().numpy().astype(float)
        return c["Fu0"] @ dz + c["Gu0"] @ x, dict(iters=int(r.info.iter),
                                                  status=str(r.info.status))

    def reset(x):
        m.output = torch.zeros_like(m.output)
        m.x = torch.zeros_like(m.x)
        m.z = torch.zeros_like(m.z)
        m.lam = torch.zeros_like(m.lam)
        m.rho_ind = rho0
    step.reset = reset

    def init(x):                                     # untimed cold solve at x
        reset(x)
        sh = c["lu_x0"] @ x
        m.update(g=c["g_x0"] @ x, l=c["l"] + sh, u=c["u"] + sh)
        m.solve()
    step.init = init
    if not warm:
        step.prepare = reset
    return step


def m_cuosqp(ocp, K, x0, opts):
    """cuOSQP keeps one global CUDA handle: a cold start destroys the old
    instance and sets up a new one (outside the timing).  Warm: the last
    primal / dual solution shifted, through warm_start(x, y)."""
    import cuosqp
    import scipy.sparse as sp
    c = condensed_lqr(ocp, K)
    Hs, As = sp.csc_matrix(c["H"]), sp.csc_matrix(c["A"])
    nu = ocp.nu
    warm = opts["start"] == "warm"
    box = {}

    def new(x):
        box["p"] = None
        gc.collect()
        sh = c["lu_x0"] @ x
        p = cuosqp.OSQP()
        p.setup(Hs, c["g_x0"] @ x, As, c["l"] + sh, c["u"] + sh, eps_abs=CONV_TOL,
                eps_rel=CONV_TOL, max_iter=CAP_ITERS, warm_start=warm, verbose=False)
        box["p"] = p
        box["last"] = None
    new(x0)
    box["p"].solve()

    def step(x, u_prev, k):
        sh = c["lu_x0"] @ x
        box["p"].update(q=c["g_x0"] @ x, l=c["l"] + sh, u=c["u"] + sh)
        if warm and k > 0 and box["last"] is not None:
            xp, yp = box["last"]
            box["p"].warm_start(x=shift_hold(xp, nu), y=shift_hold(yp, nu))
        r = box["p"].solve()
        if "solved" not in str(r.info.status).lower():
            raise RuntimeError(f"status {r.info.status}")
        dz = np.asarray(r.x, float)
        box["last"] = (dz, np.asarray(r.y, float))
        return c["Fu0"] @ dz + c["Gu0"] @ x, dict(iters=int(r.info.iter),
                                                  status=str(r.info.status))
    step.reset = new

    def init(x):                                     # untimed cold solve at x; the
        new(x)                                       # solver keeps it as its start
        box["p"].solve()
    step.init = init
    if not warm:
        step.prepare = new
    step._keep = box
    return step


def m_qpax(ocp, K, x0, opts):
    """float64 interior point, solved to CONV_TOL every step.  No warm start:
    qpax's solve_qp takes no starting point (interior point)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import qpax
    c = condensed_lqr(ocp, K)
    n = c["H"].shape[0]
    f = jnp.float64
    Q = jnp.asarray(c["H"], f)
    G = jnp.asarray(np.vstack([c["A"], -c["A"]]), f)
    Aeq, beq = jnp.zeros((0, n), f), jnp.zeros((0,), f)

    def _solve(q, h):
        out = qpax.solve_qp(Q, q, Aeq, beq, G, h, solver_tol=CONV_TOL)
        return out[0], out[4], out[5]
    solve = jax.jit(_solve)

    def args(x):
        sh = c["lu_x0"] @ x
        return (jnp.asarray(c["g_x0"] @ x, f),
                jnp.asarray(np.hstack([c["u"] + sh, -(c["l"] + sh)]), f))
    jax.block_until_ready(solve(*args(x0)))          # compile + initial solve

    def step(x, u_prev, k):
        z, conv, it = solve(*args(x))
        dz = np.asarray(z, float)
        return c["Fu0"] @ dz + c["Gu0"] @ x, dict(
            iters=int(it), status="solved" if bool(conv) else "not_converged")
    step.reset = lambda x: None
    step.warm = False
    return step


def m_cupiqp(ocp, K, x0, opts):
    """CuPIQP dense, float64, whole solve as one CUDA graph; each step updates
    c, h_l, h_u and solves from the solver's own initial point (interior
    point; no warm start input).  CUBLAS workspace off: the default one
    allocates inside the captured loop, which a conditional graph node
    rejects at some sizes."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":0:0")
    import warp as wp
    from cupiqp import DenseSolver, Status
    if hasattr(wp.config, "log_level"):
        wp.config.log_level = 30                    # logging.WARNING
    c = condensed_lqr(ocp, K)

    def arr(v):
        return wp.array(np.ascontiguousarray(np.asarray(v, float)), dtype=wp.float64,
                        device="cuda")
    s = DenseSolver(dtype=wp.float64)
    s.settings.verbose = False
    s.settings.eps_abs = CONV_TOL
    s.settings.eps_rel = CONV_TOL
    s.settings.max_iter = 1000
    sh = c["lu_x0"] @ x0
    s.setup(P=arr(c["H"]), c=arr(c["g_x0"] @ x0), G=arr(c["A"]),
            h_l=arr(c["l"] + sh), h_u=arr(c["u"] + sh))
    s.solve()
    s.result.x.numpy()                               # compile + capture + initial solve

    def step(x, u_prev, k):
        sh = c["lu_x0"] @ x
        s.update(c=arr(c["g_x0"] @ x), h_l=arr(c["l"] + sh), h_u=arr(c["u"] + sh))
        s.solve()
        dz = np.asarray(s.result.x.numpy(), float).reshape(-1)
        st = int(s.result.info.status.numpy()[0])
        if st != Status.CUPIQP_SOLVED:
            raise RuntimeError(f"status {Status(st).name}")
        return c["Fu0"] @ dz + c["Gu0"] @ x, dict(
            iters=int(s.result.info.iter.numpy()[0]), status=Status(st).name)
    step.reset = lambda x: None
    step.warm = False
    step._keep = s
    return step


def m_cudampc(ocp, K, x0, opts):
    """CUDA MPC (baselines/cudampc), float64, CUDAMPC_ITERS per step, no
    preconditioning (none in the paper).  Its kernel does the shifted warm
    start itself (warm = 2: shifted one stage, last stage repeated); step 0
    and --start cold use warm = 0."""
    if BASELINES not in sys.path:
        sys.path.insert(0, BASELINES)
    from cudampc import CudaMPC
    cm = CudaMPC.from_ocp(ocp, rho=1.0, dtype="float64", precond=False)
    cm.solve(x0, 5000)                               # initial solve at x0
    warm = opts["start"] == "warm"
    st = {"init": False}

    def step(x, u_prev, k):
        w = 2 if warm and k > 0 else (1 if warm and st["init"] else 0)
        U = cm.solve(x, CUDAMPC_ITERS, u_prev=u_prev, warm=w)
        return U[0], dict(iters=CUDAMPC_ITERS, status="fixed_iterations")

    def reset(x):                                    # warm = 0 at step 0
        st["init"] = False

    def init(x):                                     # untimed solve at x (5000
        cm.solve(x, 5000)                            # iterations, cold); step 0
        st["init"] = True                            # then uses it as is (warm = 1)
    step.reset, step.init = reset, init
    step._keep = cm
    return step


def pimpc_check_every(module, every):
    """Make piMPC test convergence every ``every`` iterations instead of every
    iteration, by recompiling its single-problem ``_solve`` in memory with the
    stopping test guarded by ``it % every == 0`` (external/ is not modified).
    The residual is still computed every iteration: the Nesterov restart
    (``res < eta * res_prev``) uses it."""
    import inspect
    import textwrap
    src = textwrap.dedent(inspect.getsource(module._solve))
    old = "        if res < tol:\n"
    if src.count(old) != 1:
        raise RuntimeError("piMPC _solve changed; cannot set its check interval")
    src = src.replace(old, f"        if it % {int(every)} == 0 and res < tol:\n")
    exec(compile(src, module.__file__, "exec"), module.__dict__)


def pimpc_unshift(m):
    """piMPC shifts its stored solution one stage at the start of every
    solve() (receding horizon).  After a solve at x, store the variables
    shifted the other way, so that the next solve at the same x starts from
    that solution: shift(back(Y)) = Y except the last stage (the second to
    last repeated), and column 0 of X is reset to x by the solve itself.
    Valid with precond = False (no state scaling), as used here."""
    import torch
    back = lambda Y: torch.cat([Y[:, :1], Y[:, :-1]], dim=1)  # noqa: E731
    m.warm_vars = tuple(back(Y) for Y in m.warm_vars)


def m_pimpc(ocp, K, x0, opts):
    """piMPC PyTorch port, float64, tol 1e-6, acceleration on, precond off
    (the port scales the state but not Wu, which would change the problem).
    Native stopping test (every iteration; CHECK["pimpc"] = 1, so its code is
    not patched -- pimpc_check_every only acts for an interval > 1).  Its
    solve shifts the stored variables itself (one stage, last repeated)."""
    import torch
    root = os.path.join(EXTERNAL, "PiMPC.jl")
    if root not in sys.path:
        sys.path.insert(0, root)
    import pytorch.pimpc as pimpc_mod
    from pytorch.pimpc import Model
    if CHECK["pimpc"] > 1:
        pimpc_check_every(pimpc_mod, CHECK["pimpc"])
    nx, nu = ocp.nx, ocp.nu
    m = Model()
    m.setup(A=ocp.A, B=ocp.B, Np=ocp.N, C=np.eye(nx), Wy=ocp.Q, Wu=ocp.R,
            Wdu=np.zeros((nu, nu)), Wf=ocp.QN, umin=ocp.umin, umax=ocp.umax,
            rho=1.0, tol=PIMPC_TOL, maxiter=CAP_ITERS, precond=False, accel=True,
            device="cuda", dtype=torch.float64)
    zy, zu = np.zeros(nx), np.zeros(nu)
    m.solve(x0, zu, zy, zu)                          # initial solve at x0

    def step(x, u_prev, k):
        r = m.solve(x, u_prev, zy, zu)
        return r.u[:, 0].detach().cpu().numpy().astype(float), dict(
            iters=int(r.iterations), status="converged" if r.converged else "not_converged")

    def reset(x):
        m.warm_vars = None
    step.reset = reset

    def init(x):                                     # untimed cold solve at x,
        m.warm_vars = None                           # stored shifted back so step 0
        m.solve(x, zu, zy, zu)                       # (at x, u_prev = 0) starts
        pimpc_unshift(m)                             # from it
    step.init = init
    if opts["start"] != "warm":
        step.prepare = reset
    return step


MAKERS = {"fame-qp": m_fame, "reluqp": m_reluqp, "cuosqp": m_cuosqp, "qpax": m_qpax,
          "cupiqp": m_cupiqp, "cudampc": m_cudampc, "pimpc": m_pimpc}


# ---------------------------------------------------------------------------
# one method, one size, in its own process
# ---------------------------------------------------------------------------
def _child(conn, method, nu, scale, steps, opts):
    try:
        ocp, d0, K = build(nu)
        x0 = scale * d0
        step = MAKERS[method](ocp, K, x0, dict(opts, x0_scale=scale))
        x, u_prev = x0.copy(), np.zeros(ocp.nu)
        rec, J, viol = [], 0.0, 0.0
        prepare = getattr(step, "prepare", None)
        init = getattr(step, "init", None) \
            if opts["start"] == "warm" and opts.get("step0", "init") == "init" else None
        for k in range(steps):
            if k == 0:
                (init or step.reset)(x)              # untimed: solve at x0, or cold
            elif prepare is not None:
                prepare(x)                           # --start cold: every step
            t0 = time.perf_counter()
            u, info = step(x, u_prev, k)
            dt = time.perf_counter() - t0
            if not np.all(np.isfinite(u)):
                conn.send(dict(error=f"non-finite input at step {k}", steps=rec))
                return
            v = float(np.max(np.maximum(np.abs(u) - ocp.umax, 0)))
            viol = max(viol, v)
            uc = np.clip(u, ocp.umin, ocp.umax)
            x = ocp.A @ x + ocp.B @ uc
            cost = float(x @ ocp.Q @ x + uc @ ocp.R @ uc)
            J += cost
            u_prev = uc
            rec.append(dict(k=k, ms=dt * 1e3, viol=v, stage_cost=cost,
                            x_norm=float(np.linalg.norm(x)),
                            bound_active=bool(np.any(np.abs(uc) >= 0.999 * ocp.umax)),
                            u0=uc.tolist(), **info))
        ts = np.array([r["ms"] for r in rec])
        it = [r["iters"] for r in rec if r.get("iters") is not None]
        warm = opts["start"] == "warm" and getattr(step, "warm", True)
        active = float(np.mean([r["bound_active"] for r in rec]))
        conn.send(dict(median_ms=float(np.median(ts)), p99_ms=float(np.percentile(ts, 99)),
                       mean_ms=float(ts.mean()), std_ms=float(ts.std(ddof=1)),
                       min_ms=float(ts.min()), max_ms=float(ts.max()), J=J, viol=viol,
                       final_x_norm=float(np.linalg.norm(x)), active_fraction=active,
                       iters_mean=float(np.mean(it)) if it else None,
                       iters_max=int(np.max(it)) if it else None,
                       statuses=sorted({str(r.get("status")) for r in rec}),
                       warm_started=bool(warm), setup=getattr(step, "info", {}), steps=rec))
    except Exception as exc:  # noqa: BLE001
        import traceback
        conn.send(dict(error=f"{type(exc).__name__}: {exc}"[:300],
                       trace=traceback.format_exc()[-1500:]))


def run_one(method, nu, scale, steps, opts, cap):
    ctx = mp.get_context("spawn")
    a, b = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(b, method, nu, scale, steps, opts))
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
        info["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], text=True).strip()
    except Exception:
        pass
    vers = {}
    for m in ("torch", "scipy", "jax", "qpax", "warp-lang"):
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


def write_csvs(payload, proc, tag):
    rows, steps = [], []
    for rec in payload["results"]:
        base = dict(nu=rec["nu"], nx=rec["nx"], N=rec["N"], n=rec["n"], m=rec["m"])
        for name, r in rec["methods"].items():
            kind, prec = INFO.get(name, ("", ""))
            b = dict(method=name, kind=kind, precision=prec, **base)
            if "median_ms" not in r:
                rows.append(dict(b, status="error" if "error" in r else "over_cap",
                                 error=r.get("error", "")))
                continue
            rows.append(dict(b, status="ok", warm_started=r.get("warm_started"),
                             median_ms=r["median_ms"], p99_ms=r["p99_ms"],
                             mean_ms=r["mean_ms"], std_ms=r["std_ms"], min_ms=r["min_ms"],
                             max_ms=r["max_ms"], J=r["J"], max_viol=r["viol"],
                             active_fraction=r.get("active_fraction"),
                             iters_mean=r["iters_mean"], iters_max=r["iters_max"],
                             setup_s=r["setup"].get("setup_s", ""),
                             mode=r["setup"].get("mode", "")))
            for s in r["steps"]:
                steps.append(dict(b, k=s["k"], ms=s["ms"], iters=s.get("iters", ""),
                                  status=s.get("status", ""), viol=s["viol"],
                                  stage_cost=s["stage_cost"], x_norm=s["x_norm"],
                                  bound_active=s["bound_active"]))
    for fname, data in ((f"summary_{tag}.csv", rows), (f"steps_{tag}.csv", steps)):
        if data:
            keys = list(dict.fromkeys(k for r in data for k in r))
            with open(os.path.join(proc, fname), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(data)


def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nus", type=int, nargs="+", default=NUS)
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=ALL_METHODS)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--cap", type=float, default=1800.0,
                    help="seconds per method per size")
    ap.add_argument("--start", choices=["warm", "cold"], default="warm",
                    help="warm: shifted warm start from step 1 on (default); "
                         "cold: every step from scratch (Table II)")
    ap.add_argument("--step0", choices=["init", "cold"], default="init",
                    help="init: untimed solve at x0 before the loop, step 0 starts "
                         "from it (default); cold: step 0 from a cold start")
    ap.add_argument("--fame-setup", choices=["heldout", "x0"], default="heldout",
                    help="problem FAME-QP is set up / probed on: a held-out random "
                         "system of the same size and scale (default) or the timed "
                         "system at x0 (Table II)")
    ap.add_argument("--out", default=os.path.join(EXP, "data", "random_lmpc_gpu"))
    ap.add_argument("--redo", action="store_true", help="rerun pairs already saved")
    args = ap.parse_args()

    raw, proc, probdir = (os.path.join(args.out, s) for s in ("raw", "processed", "problems"))
    for d in (raw, proc, probdir):
        os.makedirs(d, exist_ok=True)
    step0 = args.step0 if args.start == "warm" else "cold"
    tag = (args.start + ("_init" if step0 == "init" else "")
           + ("_fame-x0" if args.fame_setup == "x0" else ""))
    LOG = os.path.join(raw, f"run_{tag}.log")
    path = os.path.join(raw, f"random_lmpc_gpu_{tag}.json")
    set_environment()
    opts = dict(fame_setup=args.fame_setup, start=args.start, step0=step0)

    payload = json.load(open(path)) if os.path.exists(path) else dict(results=[])
    payload.update(
        kind="random_lmpc_gpu", horizon=HORIZON, steps=args.steps, conv_tol=CONV_TOL,
        pimpc_tol=PIMPC_TOL, cudampc_iters=CUDAMPC_ITERS, cap_iters=CAP_ITERS,
        check_interval=CHECK, start=args.start, step0=step0, shift_tail="hold",
        x0_scale=X0_SCALE, comparison="tolerance-matched (no reference solution)",
        fame_setup=args.fame_setup,
        fame_setup_seed="1000 + nu" if args.fame_setup == "heldout" else "nu (timed system)",
        method_info=INFO, no_warm_start=["qpax", "cupiqp"],
        problem="ReLU-QP Fig. 1 random linear MPC, nx = 3 nu, Q = I, R = 0.1 I, |u| <= 1, "
                "LQR terminal cost; QP methods: LQR-preconditioned condensed QP",
        machine=machine_info())
    log(f"\n##### {time.strftime('%Y-%m-%d %H:%M:%S')}  nus {args.nus}  methods "
        f"{args.methods}  steps {args.steps}  start {args.start}  step0 {step0}  "
        f"fame setup {args.fame_setup}")

    for nu in args.nus:
        if nu not in X0_SCALE:
            raise SystemExit(f"no x0 scale for n_u = {nu}; known: {sorted(X0_SCALE)}")
        scale = float(X0_SCALE[nu])
        ocp, d0, K = build(nu)
        np.savez(os.path.join(probdir, f"problem_nu{nu}.npz"), A=ocp.A, B=ocp.B, K=K,
                 QN=ocp.QN, x0=scale * d0, x0_scale=scale, N=HORIZON)
        rec = next((r for r in payload["results"] if r["nu"] == nu), None)
        if rec is None or rec.get("steps") != args.steps:
            payload["results"] = [r for r in payload["results"] if r["nu"] != nu]
            rec = dict(nu=nu, nx=3 * nu, N=HORIZON, n=HORIZON * nu, m=HORIZON * nu,
                       steps=args.steps, methods={})
            payload["results"].append(rec)
        rec.update(x0_scale=scale)
        for k in ("J_exact", "active_fraction"):            # from older runs
            rec.pop(k, None)
        log(f"\n=== nu={nu} nx={3 * nu} n={HORIZON * nu}: x0 scale {scale:g}")
        for m in args.methods:
            if m in rec["methods"] and not args.redo:
                r = rec["methods"][m]
                log(f"  {m:8s} (saved) " + (f"{r['median_ms']:.3f} ms/step"
                                            if "median_ms" in r else "no result"))
                continue
            t0 = gpu_temp()
            log(f"  {m:8s} running ...")
            r = run_one(m, nu, scale, args.steps, opts, args.cap)
            r.update(gpu_temp_start_C=t0, gpu_temp_end_C=gpu_temp(),
                     timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
            if "median_ms" in r:
                it = f"  it {r['iters_mean']:.0f} (max {r['iters_max']})" \
                    if r["iters_mean"] is not None else ""
                log(f"  {m:8s} {r['median_ms']:9.3f} ms/step  p99 {r['p99_ms']:9.3f}  "
                    f"J {r['J']:.6g}  active {r['active_fraction']:.0%}  "
                    f"viol {r['viol']:.1e}{it}  "
                    f"{','.join(r['statuses'])}  [{r['wall_s']:.0f} s]")
            else:
                why = r.get("error") or f"over cap {args.cap:.0f} s"
                log(f"  {m:8s} -- {why}")
            rec["methods"][m] = r
            payload["results"].sort(key=lambda r: r["nu"])
            payload["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save(path, payload)
            write_csvs(payload, proc, tag)
    log(f"\ndone {time.strftime('%Y-%m-%d %H:%M:%S')}; data in {args.out}")


if __name__ == "__main__":
    main()
