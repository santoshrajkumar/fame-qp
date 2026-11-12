"""Random linear MPC problems with LQR-preconditioned condensing.

The systems follow Bishop et al. (ReLU-QP, ICRA 2024, Sec. V-B): n_x = 3 n_u,
A random with spectral radius 1.05, Q = I, R = 0.1 I, |u| <= 1, and the
infinite-horizon LQR cost-to-go as terminal cost.

The QP uses the pre-stabilised input u_k = -K x_k + v_k (Jerez et al., 2011)
with decision z = col(v_0, ..., v_{N-1}), as in ReLU-QP's
``gen_condensed_mpc_qp``. The input-bound rows of G are block lower
triangular, so FAME-QP treats the problem as the general structure.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["ihlqr", "random_linear_system", "RandomLMPC", "random_lmpc", "X0_SCALE"]

# Initial-state scales used in the paper: the smallest power of two for which
# exact MPC keeps an input bound active on at least 30 % of 50 closed-loop steps.
X0_SCALE = {10: 16.0, 20: 8.0, 30: 16.0, 40: 8.0, 50: 8.0}


def ihlqr(A, B, Q, R, Qf, iters: int = 5000, tol: float = 1e-10):
    """Infinite-horizon LQR by Riccati iteration. Returns (K, P)."""
    P = Qf.copy()
    K = np.zeros((B.shape[1], A.shape[0]))
    for _ in range(iters):
        Pp = P
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P = Q + A.T @ P @ (A - B @ K)
        if np.linalg.norm(P - Pp, 2) < tol:
            break
    return K, 0.5 * (P + P.T)


def random_linear_system(n_u: int, seed: Optional[int] = None,
                         spectral_radius: float = 1.05):
    """Random (A, B) with n_x = 3 n_u; ``seed`` defaults to ``n_u``."""
    nx = 3 * n_u
    seed = n_u if seed is None else seed
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((nx, nx)) / np.sqrt(nx)
    A = A * (spectral_radius / np.max(np.abs(np.linalg.eigvals(A))))
    B = rng.standard_normal((nx, n_u)) / np.sqrt(nx)
    return A, B


class RandomLMPC:
    """Condensed, LQR-preconditioned LMPC with input bounds.

    Attributes ``A, B, Q, R, Q_N, K, N`` define the problem; ``d0`` is a
    random unit-scale initial direction. For a state x, :meth:`qp_data`
    returns (P, q, G, l, h) and :meth:`first_input` maps a solution z back to
    the applied input u_0.
    """

    def __init__(self, A, B, Q, R, Q_N, K, N: int, u_lo, u_hi, d0=None):
        self.A, self.B = np.asarray(A, float), np.asarray(B, float)
        self.Q, self.R, self.Q_N = (np.asarray(M, float) for M in (Q, R, Q_N))
        self.K = np.asarray(K, float)
        self.N = int(N)
        self.nx, self.nu = self.B.shape
        self.u_lo = np.broadcast_to(np.asarray(u_lo, float), (self.nu,)).copy()
        self.u_hi = np.broadcast_to(np.asarray(u_hi, float), (self.nu,)).copy()
        self.d0 = None if d0 is None else np.asarray(d0, float)
        self._condense()

    def _condense(self):
        A, B, Q, R, Qf, K, N = self.A, self.B, self.Q, self.R, self.Q_N, self.K, self.N
        nx, nu = self.nx, self.nu
        blk = nu + nx

        # Stage-wise cost on w = col(u_0, x_1, ..., u_{N-1}, x_N).
        Hs = np.zeros((N * blk, N * blk))
        for k in range(N):
            Hs[k * blk:k * blk + nu, k * blk:k * blk + nu] = R
            Hs[k * blk + nu:(k + 1) * blk, k * blk + nu:(k + 1) * blk] = Qf if k == N - 1 else Q

        # w = F z + E x_hat under the closed loop A - B K.
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
        E = np.vstack([np.vstack([-K @ Apow[k], Apow[k + 1]]) for k in range(N)])

        sel = np.zeros((N * nu, N * blk))
        for k in range(N):
            sel[k * nu:(k + 1) * nu, k * blk:k * blk + nu] = np.eye(nu)

        P = F.T @ Hs @ F
        self.P = 0.5 * (P + P.T)
        self.q_x = F.T @ Hs @ E
        self.G = sel @ F
        self.l0 = np.tile(self.u_lo, N)
        self.h0 = np.tile(self.u_hi, N)
        self.bound_x = -sel @ E
        self.u0_z, self.u0_x = F[:nu], E[:nu]
        self.n, self.m = N * nu, N * nu

    def qp_data(self, x):
        """Return (P, q, G, l, h) of the QP at state ``x``."""
        x = np.asarray(x, float).reshape(-1)
        sh = self.bound_x @ x
        return self.P, self.q_x @ x, self.G, self.l0 + sh, self.h0 + sh

    def qp(self, x, device="cuda"):
        from ..problem import QP
        return QP.create(*self.qp_data(x), device=device)

    def first_input(self, z, x):
        """Applied input u_0 for the solution ``z`` at state ``x``."""
        return self.u0_z @ np.asarray(z, float) + self.u0_x @ np.asarray(x, float)

    def stage_segments(self):
        """Segments for ``Solver.warm_start_shifted``: N stages of n_u in z and G."""
        seg = [(0, self.nu, self.N)]
        return seg, seg

    def initial_state(self, scale: Optional[float] = None):
        """``scale * d0``; the scale defaults to the paper's value for n_u."""
        if self.d0 is None:
            raise ValueError("no initial direction d0 for this problem")
        scale = X0_SCALE.get(self.nu, 1.0) if scale is None else scale
        return scale * self.d0

    def step(self, x, u):
        """Plant update with u clipped to the bounds."""
        return self.A @ x + self.B @ np.clip(u, self.u_lo, self.u_hi)


def random_lmpc(n_u: int, N: int = 40, seed: Optional[int] = None) -> RandomLMPC:
    """Random LMPC benchmark problem; ``seed`` defaults to ``n_u``.

    The initial direction ``d0`` is drawn with ``seed + 1``.
    """
    seed = n_u if seed is None else seed
    A, B = random_linear_system(n_u, seed)
    nx = 3 * n_u
    Q, R = np.eye(nx), 0.1 * np.eye(n_u)
    K, Q_N = ihlqr(A, B, Q, R, Q)
    d0 = np.random.default_rng(seed + 1).standard_normal(nx)
    return RandomLMPC(A, B, Q, R, Q_N, K, N, -np.ones(n_u), np.ones(n_u), d0=d0)
