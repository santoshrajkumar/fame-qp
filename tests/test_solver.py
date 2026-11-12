"""Tests: structure detection, accuracy of every mode and precision against
Clarabel, kernels per iteration, automatic selection, updates, LMPC."""

import os
import tempfile

import numpy as np
import pytest
import scipy.sparse as sp
import torch

os.environ.setdefault("FAME_QP_CACHE", os.path.join(tempfile.gettempdir(), "fame_qp_test_cache"))

from fame_qp import QP, Solver  # noqa: E402
from fame_qp.mpc import LinearMPC  # noqa: E402

clarabel = pytest.importorskip("clarabel")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def reference(P, q, G, l, h):
    G = sp.csc_matrix(G)
    fu = np.isfinite(h) & (h < 1e20)
    fl = np.isfinite(l) & (l > -1e20)
    eq = fu & fl & (np.abs(h - l) < 1e-12)
    fu, fl = fu & ~eq, fl & ~eq
    A = sp.vstack([G[eq], G[fu], -G[fl]], format="csc")
    b = np.concatenate([h[eq], h[fu], -l[fl]])
    cones = ([clarabel.ZeroConeT(int(eq.sum()))] if eq.any() else []) + \
        [clarabel.NonnegativeConeT(int(fu.sum() + fl.sum()))]
    s = clarabel.DefaultSettings()
    s.verbose = False
    s.tol_gap_abs = s.tol_gap_rel = s.tol_feas = 1e-10
    return np.array(clarabel.DefaultSolver(sp.csc_matrix(P), q, A, b, cones, s).solve().x)


def random_qp(n, kind, seed=0, mc=None):
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((n, n))
    P = R.T @ R / n + 0.1 * np.eye(n)
    q = rng.standard_normal(n)
    if kind == "box":
        return P, q, None, -0.3 * np.ones(n), 0.3 * np.ones(n)
    mc = n // 2 if mc is None else mc
    C = rng.standard_normal((mc, n))
    G = {"split_top": np.vstack([np.eye(n), C]),
         "split_bot": np.vstack([C, 2.0 * np.eye(n)]),
         "general": rng.standard_normal((n + mc, n))}[kind]
    m = G.shape[0]
    return P, q, G, -0.5 * np.ones(m), 0.5 * np.ones(m)


def dense_G(G, n):
    return np.eye(n) if G is None else G


def solve(P, q, G, l, h, **kw):
    s = Solver(QP.create(P, q, G, l, h), **kw)
    return s, s.solve()


# ---------------------------------------------------------------------------
def test_structure_detection():
    n = 6
    P, q, C = np.eye(n), np.zeros(n), np.ones((3, n))
    assert QP.create(P, q).structure == "box"
    assert QP.create(P, q, np.diag(np.arange(1.0, n + 1))).structure == "box"
    assert QP.create(P, q, np.vstack([np.eye(n), C])).structure == "split"
    qp = QP.create(P, q, np.vstack([C, np.eye(n)]))                 # LMPC layout
    assert qp.structure == "split"
    assert qp.perm.tolist() == list(range(3, 9)) + [0, 1, 2]
    assert QP.create(P, q, np.ones((n, n))).structure == "general"
    assert QP.create(P, q, np.vstack([np.eye(n), C]),
                     detect_structure=False).structure == "general"


CASES = [("box", "persistent"), ("box", "single_kernel"),
         ("split_top", "single_kernel"), ("split_top", "two_kernel"),
         ("split_bot", "single_kernel"), ("split_bot", "two_kernel"),
         ("general", "single_kernel"), ("general", "two_kernel")]


@pytest.mark.parametrize("kind,mode", CASES)
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_accuracy_every_mode(kind, mode, precision):
    n = 48 if mode == "persistent" else 150
    P, q, G, l, h = random_qp(n, kind, seed=n)
    x = reference(P, q, dense_G(G, n), l, h)
    s, r = solve(P, q, G, l, h, mode=mode, precision=precision, eps_abs=1e-5, eps_rel=1e-5)
    assert r.info["mode"] == mode
    assert r.status in ("solved", "solved_inaccurate")
    assert np.abs(r.z.double().cpu().numpy() - x).max() < 1e-3


@pytest.mark.parametrize("kind", ["box", "split_bot", "general"])
def test_kkt_in_user_row_order(kind):
    n = 60
    P, q, G, l, h = random_qp(n, kind, seed=3)
    Gd = dense_G(G, n)
    _, r = solve(P, q, G, l, h, precision="float64", eps_abs=1e-8, eps_rel=1e-8)
    z, y, s = (t.cpu().numpy() for t in (r.z, r.lam, r.s))
    assert np.abs(P @ z + q + Gd.T @ y).max() < 1e-5
    assert np.abs(Gd @ z - s).max() < 1e-5
    assert np.all(s >= l - 1e-6) and np.all(s <= h + 1e-6)


@pytest.mark.parametrize("kind,mode,per_iter", [("box", "single_kernel", 1),
                                                ("split_bot", "single_kernel", 1),
                                                ("split_bot", "two_kernel", 2),
                                                ("general", "single_kernel", 1),
                                                ("general", "two_kernel", 2)])
def test_kernels_per_iteration(kind, mode, per_iter):
    """A graph block launches exactly per_iter x check_interval iteration
    kernels; the rest of the block is the once-per-block stopping test."""
    from torch.profiler import ProfilerActivity, profile
    n = 200
    P, q, G, l, h = random_qp(n, kind)
    s = Solver(QP.create(P, q, G, l, h), mode=mode, check_interval=20)
    s.eng.prime()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        s.eng.block()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
    it = [x for x in names if "_single_kernel_iter" in x or "_two_kernel_pass" in x]
    assert len(it) == 20 * per_iter
    assert len(names) - len(it) <= 8


def test_persistent_is_one_launch():
    from torch.profiler import ProfilerActivity, profile
    P, q, G, l, h = random_qp(40, "box", seed=4)
    s = Solver(QP.create(P, q, G, l, h), mode="persistent", use_graph=False,
               eps_abs=1e-5, eps_rel=1e-5)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        r = s.solve()
    names = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
    assert r.status == "solved"
    # one launch per persistent_budget iterations (rho adapts between them),
    # and no per-iteration kernels at all
    launches = sum("_persistent_solve" in x for x in names)
    assert launches == -(-r.iters // s.st.persistent_budget)
    assert not any("_single_kernel_iter" in x or "_two_kernel" in x for x in names)


def test_auto_mode_selection():
    P, q, G, l, h = random_qp(64, "box")
    assert solve(P, q, G, l, h)[1].info["mode"] == "persistent"
    P, q, G, l, h = random_qp(300, "box")
    assert solve(P, q, G, l, h)[1].info["mode"] == "single_kernel"
    P, q, G, l, h = random_qp(300, "split_bot", mc=1500)       # m_c >> n
    assert solve(P, q, G, l, h)[1].info["mode"] == "two_kernel"
    P, q, G, l, h = random_qp(200, "split_bot", mc=40)          # few dense rows
    assert solve(P, q, G, l, h)[1].info["mode"] == "single_kernel"


def test_auto_precision():
    P, q, G, l, h = random_qp(100, "split_bot", seed=11)
    s, r = solve(P, q, G, l, h, precision="auto", eps_abs=1e-3, eps_rel=1e-3)
    assert r.info["precision"] == "float32"            # well conditioned, loose eps
    assert s.kappa is not None and s.kappa < 1e6
    s, r = solve(P, q, G, l, h, precision="auto", eps_abs=1e-9, eps_rel=1e-9)
    assert r.info["precision"] == "float64"            # beyond float32
    x = reference(P, q, G, l, h)
    assert np.abs(r.z.cpu().numpy() - x).max() < 1e-6


@pytest.mark.parametrize("kind", ["box", "split_bot", "general"])
def test_update_vectors(kind):
    n = 80
    P, q, G, l, h = random_qp(n, kind, seed=7)
    s, _ = solve(P, q, G, l, h, precision="float64", eps_abs=1e-7, eps_rel=1e-7)
    rng = np.random.default_rng(1)
    q2 = rng.standard_normal(n)
    l2, h2 = l - 0.1 * rng.random(l.size), h + 0.1 * rng.random(h.size)
    s.update(q=q2, l=l2, h=h2)
    x = reference(P, q2, dense_G(G, n), l2, h2)
    assert np.abs(s.solve().z.cpu().numpy() - x).max() < 1e-4


@pytest.mark.parametrize("kind,mode", [("box", "single_kernel"), ("split_bot", "two_kernel"),
                                       ("general", "single_kernel")])
def test_update_matrices(kind, mode):
    n = 80
    P, q, G, l, h = random_qp(n, kind, seed=8)
    s, _ = solve(P, q, G, l, h, mode=mode, precision="float64", eps_abs=1e-7, eps_rel=1e-7)
    rng = np.random.default_rng(2)
    R = rng.standard_normal((n, n))
    P2 = R.T @ R / n + 0.2 * np.eye(n)
    G2 = None
    if G is not None:
        G2 = G.copy()
        rows = slice(0, G.shape[0] - n) if kind == "split_bot" else slice(None)
        G2[rows] += 0.1 * rng.standard_normal(G2[rows].shape)
    s.update_matrices(P=P2, G=G2)
    x = reference(P2, q, dense_G(G2, n), l, h)
    assert np.abs(s.solve().z.cpu().numpy() - x).max() < 1e-4


def test_warm_start_helps():
    P, q, G, l, h = random_qp(150, "split_bot", seed=9)
    s, r = solve(P, q, G, l, h, eps_abs=1e-5, eps_rel=1e-5)
    s.update(q=q + 1e-3)
    assert s.solve().iters < r.iters


# ---------------------------------------------------------------------------
def _mpc(state_bounds=True, K=False, seed=0):
    rng = np.random.default_rng(seed)
    nx, nu, N = 6, 2, 20
    A = rng.standard_normal((nx, nx))
    A = A / np.max(np.abs(np.linalg.eigvals(A))) * 0.98
    B = rng.standard_normal((nx, nu))
    Q, R = np.eye(nx), 0.1 * np.eye(nu)
    Kg = None
    if K:
        import scipy.linalg as sla
        Pr = sla.solve_discrete_are(A, B, Q, R)
        Kg = np.linalg.solve(R + B.T @ Pr @ B, B.T @ Pr @ A)
    xb = 2.0 if state_bounds else None
    return LinearMPC(A, B, Q, R, Q, N, u_lo=-0.5, u_hi=0.5,
                     x_lo=None if xb is None else -xb, x_hi=xb, K=Kg), nx, nu


@pytest.mark.parametrize("state_bounds,K,structure",
                         [(False, False, "box"), (True, False, "split"), (True, True, "general")])
def test_lmpc_condensed_matches_sparse(state_bounds, K, structure):
    mpc, nx, nu = _mpc(state_bounds, K)
    x_hat = np.linspace(-1.5, 1.5, nx)
    qp = mpc.qp(x_hat)
    assert qp.structure == structure
    r = Solver(qp, precision="float64", eps_abs=1e-8, eps_rel=1e-8).solve()
    u0 = mpc.first_input(r.z.cpu().numpy(), x_hat)
    zs = reference(*mpc.sparse_qp(x_hat))
    assert np.abs(u0 - zs[nx:nx + nu]).max() < 1e-4


def test_lmpc_controller_loop():
    mpc, nx, nu = _mpc(True)
    x = np.linspace(-1.5, 1.5, nx)
    ctrl = mpc.controller(x, precision="float64", eps_abs=1e-7, eps_rel=1e-7)
    for _ in range(10):
        u = ctrl.step(x)
        zs = reference(*mpc.sparse_qp(x))
        assert np.abs(u - zs[nx:nx + nu]).max() < 1e-4
        x = mpc.A @ x + mpc.B @ u


def test_calibration_cached(tmp_path, monkeypatch):
    from fame_qp import calibration as cm
    monkeypatch.setenv("FAME_QP_CACHE", str(tmp_path))
    cm._mem.clear()
    c1 = cm.get_calibration(torch.float32)
    assert (tmp_path / "calibration.json").exists()
    cm._mem.clear()                       # new process: read from disk
    assert cm.get_calibration(torch.float32) == c1
    cm._mem.clear()


# ---------------------------------------------------------------------------
# shifted warm start
# ---------------------------------------------------------------------------
def test_shift_stages():
    from fame_qp.warmstart import shift_stages
    v = torch.arange(10.0)
    # z: 3 stages of 2 at 0; rows: 2 stages of 1 at 7; entries 6, 9 untouched
    out = shift_stages(v, [(0, 2, 3), (7, 1, 2)])
    assert out.tolist() == [2, 3, 4, 5, 4, 5, 6, 8, 8, 9]
    out = shift_stages(v, [(0, 2, 3)], tail="zero")
    assert out[:6].tolist() == [2, 3, 4, 5, 0, 0]
    assert v.tolist() == list(range(10))               # input unchanged
    with pytest.raises(ValueError):
        shift_stages(v, [(8, 2, 2)])


def test_warm_start_shifted_roundtrip():
    """No segments: the warm start is the last solution itself, unscaled and
    rescaled exactly, so the next solve stops at its first check."""
    P, q, G, l, h = random_qp(120, "split_bot", seed=4)
    s, r = solve(P, q, G, l, h, eps_abs=1e-6, eps_rel=1e-6)
    z0 = r.z.clone()
    s.warm_start_shifted()
    r2 = s.solve()
    assert r2.iters <= s.st.check_interval
    assert (r2.z - z0).abs().max().item() < 1e-5


@pytest.mark.parametrize("K", [False, True])
def test_lmpc_controller_shift(K):
    """Shifted and unshifted warm starts give the same closed loop; the
    shifted one needs no more iterations in total."""
    mpc, nx, nu = _mpc(True, K)
    x0 = np.linspace(-1.5, 1.5, nx)
    us, its = {}, {}
    for shift in (True, False):
        ctrl = mpc.controller(x0, precision="float64", eps_abs=1e-7, eps_rel=1e-7,
                              shift=shift)
        x, u_log, n_it = x0.copy(), [], 0
        for _ in range(15):
            u = ctrl.step(x)
            n_it += ctrl.last.iters
            u_log.append(u)
            x = mpc.A @ x + mpc.B @ u
        us[shift], its[shift] = np.array(u_log), n_it
    assert np.abs(us[True] - us[False]).max() < 1e-4
    assert its[True] <= its[False]
