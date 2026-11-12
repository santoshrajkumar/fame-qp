"""Linear MPC as a condensed QP.

    minimize    1/2 sum_k (x_k'Q x_k + u_k'R u_k) + 1/2 x_N'Q_N x_N
    subject to  x_0 = x_hat,  x_{k+1} = A x_k + B u_k,
                u_lo <= u_k <= u_hi,  x_lo <= x_k <= x_hi

Eliminating the states gives z = col(u_0, ..., u_{N-1}) and

    P = B_cal' Q_cal B_cal + R_cal,   q = B_cal' Q_cal A_cal x_hat,
    G = [B_cal; I],   l = [X_lo - A_cal x_hat; U_lo],   h = [X_hi - A_cal x_hat; U_hi].

State rows for x_0 and rows without a finite bound are dropped. With input
bounds only, G = I (box); with state bounds, G is split. Optional LQR
pre-stabilisation u_k = -K x_k + v_k (Jerez et al., 2011) makes the input
rows non-diagonal, giving the general structure.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .problem import QP
from .solver import Settings, Solver

__all__ = ["LinearMPC", "MPCController"]


def _vec(v, k, default):
    if v is None:
        return np.full(k, default, dtype=float)
    v = np.asarray(v, dtype=float).reshape(-1)
    return np.full(k, float(v[0])) if v.size == 1 else v.copy()


class LinearMPC:
    def __init__(self, A, B, Q, R, Q_N, N: int, u_lo=None, u_hi=None,
                 x_lo=None, x_hi=None, K: Optional[np.ndarray] = None):
        self.A, self.B = np.asarray(A, float), np.asarray(B, float)
        self.Q, self.R, self.Q_N = (np.asarray(M, float) for M in (Q, R, Q_N))
        self.N = int(N)
        self.nx, self.nu = self.B.shape
        self.u_lo, self.u_hi = _vec(u_lo, self.nu, -np.inf), _vec(u_hi, self.nu, np.inf)
        self.x_lo, self.x_hi = _vec(x_lo, self.nx, -np.inf), _vec(x_hi, self.nx, np.inf)
        self.K = None if K is None else np.asarray(K, float).reshape(self.nu, self.nx)
        self._condense()

    def _condense(self):
        A, B, N, nx, nu = self.A, self.B, self.N, self.nx, self.nu
        Ak = A if self.K is None else A - B @ self.K
        Apow = [np.eye(nx)]
        for _ in range(N):
            Apow.append(Ak @ Apow[-1])
        Acal = np.vstack(Apow)
        Bcal = np.zeros(((N + 1) * nx, N * nu))
        for i in range(1, N + 1):
            for j in range(i):
                Bcal[i * nx:(i + 1) * nx, j * nu:(j + 1) * nu] = Apow[i - 1 - j] @ B
        if self.K is None:
            Ux, Uv = np.zeros((N * nu, nx)), np.eye(N * nu)
        else:
            Kcal = np.kron(np.eye(N), self.K)
            Ux = -Kcal @ Acal[: N * nx]
            Uv = np.eye(N * nu) - Kcal @ Bcal[: N * nx]
        Qcal = np.zeros(((N + 1) * nx, (N + 1) * nx))
        for k in range(N):
            Qcal[k * nx:(k + 1) * nx, k * nx:(k + 1) * nx] = self.Q
        Qcal[N * nx:, N * nx:] = self.Q_N
        Rcal = np.kron(np.eye(N), self.R)
        self.Acal, self.Bcal, self.Qcal, self.Rcal, self.Ux, self.Uv = Acal, Bcal, Qcal, Rcal, Ux, Uv
        P = Bcal.T @ Qcal @ Bcal + Uv.T @ Rcal @ Uv
        self.P = 0.5 * (P + P.T)
        self.F = Bcal.T @ Qcal @ Acal + Uv.T @ Rcal @ Ux

        Xlo, Xhi = np.tile(self.x_lo, N + 1), np.tile(self.x_hi, N + 1)
        keep = np.isfinite(Xlo) | np.isfinite(Xhi)
        keep[:nx] = False
        S = np.flatnonzero(keep)
        self.state_rows = S
        self.G = np.vstack([Bcal[S], Uv])
        self.l0 = np.concatenate([Xlo[S], np.tile(self.u_lo, N)])
        self.h0 = np.concatenate([Xhi[S], np.tile(self.u_hi, N)])
        self.Lx = np.vstack([Acal[S], Ux])
        self.n, self.m = N * nu, self.G.shape[0]

    def stage_segments(self):
        """Segments of z and of the rows of G for ``Solver.warm_start_shifted``."""
        N, nu = self.N, self.nu
        nb = self.state_rows.size // N
        z_seg = [(0, nu, N)]
        lam_seg = [(0, nb, N), (nb * N, nu, N)] if nb else [(0, nu, N)]
        return z_seg, lam_seg

    def qp_data(self, x_hat):
        """Return (P, q, G, l, h) of the condensed QP at ``x_hat``."""
        x_hat = np.asarray(x_hat, float).reshape(-1)
        off = self.Lx @ x_hat
        return self.P, self.F @ x_hat, self.G, self.l0 - off, self.h0 - off

    def qp(self, x_hat, device="cuda") -> QP:
        return QP.create(*self.qp_data(x_hat), device=device)

    def first_input(self, z, x_hat):
        """Recover u_0 from the condensed solution ``z``."""
        nu = self.nu
        return self.Ux[:nu] @ np.asarray(x_hat, float) + self.Uv[:nu] @ np.asarray(z, float)

    def controller(self, x_hat0, device="cuda", settings: Optional[Settings] = None,
                   **kw) -> "MPCController":
        return MPCController(self, x_hat0, device, settings, **kw)

    def sparse_qp(self, x_hat):
        """Sparse QP in z = col(x_0, u_0, ..., x_N), for reference solutions."""
        import scipy.sparse as sp
        A, B, N, nx, nu = self.A, self.B, self.N, self.nx, self.nu
        P = sp.block_diag([M for _ in range(N) for M in (self.Q, self.R)] + [self.Q_N],
                          format="csc")
        nz = (N + 1) * nx + N * nu
        E = sp.lil_matrix(((N + 1) * nx, nz))
        E[:nx, :nx] = np.eye(nx)
        step = nx + nu
        for k in range(N):
            r, c = (k + 1) * nx, k * step
            E[r:r + nx, c:c + nx] = -A
            E[r:r + nx, c + nx:c + step] = -B
            E[r:r + nx, c + step:c + step + nx] = np.eye(nx)
        G = sp.vstack([E.tocsc(), sp.eye(nz, format="csc")], format="csc")
        b = np.zeros((N + 1) * nx)
        b[:nx] = np.asarray(x_hat, float)
        zlo = np.concatenate([np.concatenate([self.x_lo, self.u_lo])] * N + [self.x_lo])
        zhi = np.concatenate([np.concatenate([self.x_hi, self.u_hi])] * N + [self.x_hi])
        zlo[:nx], zhi[:nx] = -np.inf, np.inf
        return P, np.zeros(nz), G, np.concatenate([b, zlo]), np.concatenate([b, zhi])


class MPCController:
    """Receding-horizon controller.

    Each step updates q and the bounds. With ``shift=True`` the previous
    solution is shifted one stage as the warm start; otherwise the last
    iterate is reused.
    """

    def __init__(self, mpc: LinearMPC, x_hat0, device="cuda",
                 settings: Optional[Settings] = None, shift: bool = True, **kw):
        self.mpc, self.shift = mpc, shift
        self.solver = Solver(mpc.qp(x_hat0, device=device), settings, **kw)
        t = lambda a: torch.as_tensor(a, device=device, dtype=torch.float64)  # noqa: E731
        self._F, self._Lx, self._l0, self._h0 = t(mpc.F), t(mpc.Lx), t(mpc.l0), t(mpc.h0)
        self._u0z, self._u0x = t(mpc.Uv[: mpc.nu]), t(mpc.Ux[: mpc.nu])
        self.device, self.last = device, None

    def step(self, x_hat) -> np.ndarray:
        x = torch.as_tensor(np.asarray(x_hat, float), device=self.device, dtype=torch.float64)
        off = self._Lx @ x
        self.solver.update(q=self._F @ x, l=self._l0 - off, h=self._h0 - off)
        if self.shift and self.last is not None:
            self.solver.warm_start_shifted(*self.mpc.stage_segments())
        self.last = self.solver.solve()
        u0 = self._u0x @ x + self._u0z @ self.last.z.to(torch.float64)
        return u0.cpu().numpy()
