"""Python front-end for the CUDA MPC re-implementation (``cudampc.cu``).

Builds the parallel-in-horizon ADMM data of piMPC / CUDA MPC from a linear MPC
problem, hands it to the persistent kernel, and provides a NumPy reference of
the identical iteration for validation.

Formulation (CUDA MPC Sec. IV-B, piMPC):  augmented state xb_k = [x_k; u_{k-1}],
input du_k = u_k - u_{k-1},

    xb_{k+1} = Ab xb_k + Bb du_k + e,     Ab = [[A, B], [0, I]],  Bb = [[B], [I]]

    min  sum_{k=0}^{N-1} 1/2 xb_{k+1}' Qb_k xb_{k+1} - qb_k' xb_{k+1}
                         + 1/2 du_k' R du_k
    s.t. lo <= xb_{k+1} <= hi

A standard MPC problem (cost x'Qx + u'Ru, input and state boxes) is the special
case Qb = blkdiag(Q, R), Qb_N = blkdiag(QN, R), R = 0: the u_{k-1} block of
xb_{k+1} carries u_k, so the two problems have the same minimiser.

Diagonal preconditioning follows piMPC, E = sqrt(diag(Ab'Ab)), but the cost is
transformed consistently (Qb -> E^-1 Qb E^-1); piMPC's reference code leaves the
input-weight block unscaled, which changes the problem whenever Wu != 0.
"""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "libcudampc.so")
BIG = 1e20
NVCC_CANDIDATES = [
    os.environ.get("NVCC", ""),
    os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc"),
    os.path.join(os.path.dirname(os.path.dirname(HERE)), "external", "cuda11", "toolkit", "bin", "nvcc"),
]

__all__ = ["CudaMPC", "PiADMMReference", "build_library"]


def build_library(force=False, arch="sm_86"):
    src = os.path.join(HERE, "cudampc.cu")
    if (not force and os.path.exists(LIB)
            and os.path.getmtime(LIB) >= os.path.getmtime(src)):
        return LIB
    nvcc = next((c for c in NVCC_CANDIDATES if c and os.path.exists(c)), "nvcc")
    subprocess.check_call([nvcc, "-O3", f"-arch={arch}", "-shared",
                           "-Xcompiler", "-fPIC", "-o", LIB, src])
    return LIB


_lib = None


def _load():
    global _lib
    if _lib is None:
        build_library()
        L = ctypes.CDLL(LIB)
        dp = ctypes.POINTER(ctypes.c_double)
        L.cmpc_create.restype = ctypes.c_void_p
        L.cmpc_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_double] + [dp] * 9 + [
                                      ctypes.c_int, ctypes.c_int]
        L.cmpc_solve.restype = ctypes.c_int
        L.cmpc_solve.argtypes = [ctypes.c_void_p, dp, ctypes.c_int, ctypes.c_int,
                                 dp, ctypes.POINTER(ctypes.c_float)]
        L.cmpc_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        L.cmpc_set_q.argtypes = [ctypes.c_void_p, dp]
        L.cmpc_destroy.argtypes = [ctypes.c_void_p]
        _lib = L
    return _lib


def _dp(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


# ---------------------------------------------------------------------------
class _Data:
    """Scaled piMPC matrices, shared by the kernel and the NumPy reference."""

    def __init__(self, A, B, Qb, QbN, R, qb, lo, hi, e=None, rho=1.0,
                 precond=True):
        A = np.asarray(A, float)
        B = np.asarray(B, float)
        nx, nu = B.shape
        nb = nx + nu
        N = qb.shape[0]
        Ab = np.block([[A, B], [np.zeros((nu, nx)), np.eye(nu)]])
        Bb = np.vstack([B, np.eye(nu)])
        eb = np.zeros(nb) if e is None else np.hstack([np.asarray(e, float),
                                                       np.zeros(nu)])
        E = np.sqrt(np.diag(Ab.T @ Ab)) if precond else np.ones(nb)
        Ei = 1.0 / E
        self.E = E
        self.Ab = (E[:, None] * Ab) * Ei[None, :]
        self.Bb = E[:, None] * Bb
        self.e = E * eb
        Qs = Ei[:, None] * Qb * Ei[None, :]
        QsN = Ei[:, None] * QbN * Ei[None, :]
        self.q = qb * Ei[None, :]                       # N x nb
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        self.lo = np.where(np.isfinite(lo) & (np.abs(lo) < BIG), E * lo, -BIG)
        self.hi = np.where(np.isfinite(hi) & (np.abs(hi) < BIG), E * hi, BIG)
        I = np.eye(nb)
        R = np.asarray(R, float)
        self.J = np.linalg.solve(R + rho * self.Bb.T @ self.Bb, rho * self.Bb.T)
        self.H = np.linalg.inv(Qs + rho * I + rho * self.Ab.T @ self.Ab)
        self.HN = np.linalg.inv(QsN + rho * I)
        self.nx, self.nu, self.nb, self.N, self.rho = nx, nu, nb, N, rho

    def xbar0(self, x0, u_prev=None):
        u_prev = np.zeros(self.nu) if u_prev is None else np.asarray(u_prev, float)
        return self.E * np.hstack([np.asarray(x0, float).reshape(-1), u_prev])

    def unscale_u(self, Xu):
        """Scaled u-block of xb_{1..N} (N x nu) -> inputs u_0..u_{N-1}."""
        return Xu / self.E[self.nx:][None, :]

    # -- construction from a standard linear MPC problem ---------------------
    @classmethod
    def from_ocp(cls, ocp, rho=1.0, precond=True):
        nx, nu, N = ocp.nx, ocp.nu, ocp.N
        Z = np.zeros((nx, nu))
        Qb = np.block([[ocp.Q, Z], [Z.T, ocp.R]])
        QbN = np.block([[ocp.QN, Z], [Z.T, ocp.R]])
        xr = np.asarray(ocp.xref, float)
        qb = np.tile(np.hstack([ocp.Q @ xr, np.zeros(nu)]), (N, 1))
        qb[-1] = np.hstack([ocp.QN @ xr, np.zeros(nu)])
        lo = np.hstack([ocp.xmin if ocp.xmin is not None else np.full(nx, -np.inf),
                        ocp.umin if ocp.umin is not None else np.full(nu, -np.inf)])
        hi = np.hstack([ocp.xmax if ocp.xmax is not None else np.full(nx, np.inf),
                        ocp.umax if ocp.umax is not None else np.full(nu, np.inf)])
        return cls(ocp.A, ocp.B, Qb, QbN, np.zeros((nu, nu)), qb, lo, hi,
                   rho=rho, precond=precond)


# ---------------------------------------------------------------------------
class CudaMPC:
    """Persistent-kernel CUDA MPC solver for one linear MPC problem.

    ``solve`` runs exactly ``iters`` ADMM iterations (CUDA MPC uses a fixed
    iteration budget; there is no grid-wide residual test inside the kernel).
    """

    def __init__(self, data: _Data, dtype="float32", force_global=False,
                 max_stages=0):
        L = _load()
        d = data
        self.d = d
        self.dtype = dtype
        arrs = [np.ascontiguousarray(a, dtype=np.float64) for a in
                (d.Ab, d.Bb, d.J, d.H, d.HN, d.lo, d.hi, d.e, d.q)]
        self._keep = arrs
        self.h = L.cmpc_create(1 if dtype == "float64" else 0, d.nx, d.nu, d.N,
                               float(d.rho), *[_dp(a) for a in arrs],
                               int(force_global), int(max_stages))
        if not self.h:
            raise RuntimeError("cudampc: setup failed (see stderr)")
        info = (ctypes.c_int * 5)()
        L.cmpc_info(self.h, info)
        self.M, self.B, self.mats_shared, self.smem, self.threads = list(info)
        self._u = np.zeros(d.N * d.nu)
        self._kms = ctypes.c_float(0.0)

    @classmethod
    def from_ocp(cls, ocp, rho=1.0, precond=True, **kw):
        return cls(_Data.from_ocp(ocp, rho=rho, precond=precond), **kw)

    def set_q(self, qb):
        q = np.ascontiguousarray(np.asarray(qb, float) / self.d.E[None, :])
        _load().cmpc_set_q(self.h, _dp(q))

    def solve(self, x0, iters, u_prev=None, warm=False, kernel_time=False):
        xb = np.ascontiguousarray(self.d.xbar0(x0, u_prev))
        rc = _load().cmpc_solve(self.h, _dp(xb), int(iters), int(warm),
                                _dp(self._u),
                                ctypes.byref(self._kms) if kernel_time else None)
        if rc != 0:
            raise RuntimeError("cudampc: solve failed")
        U = self.d.unscale_u(self._u.reshape(self.d.N, self.d.nu))
        return U

    @property
    def last_kernel_ms(self):
        return float(self._kms.value)

    def __del__(self):
        try:
            if getattr(self, "h", None):
                _load().cmpc_destroy(self.h)
                self.h = None
        except Exception:
            pass


# ---------------------------------------------------------------------------
class PiADMMReference:
    """NumPy float64 implementation of the *same* iteration (validation)."""

    def __init__(self, data: _Data):
        self.d = data

    def solve(self, x0, iters, u_prev=None, dtype=np.float64):
        d = self.d
        N, nb = d.N, d.nb
        c = lambda a: np.asarray(a, dtype)  # noqa: E731
        Ab, Bb, J, H, HN, e = map(c, (d.Ab, d.Bb, d.J, d.H, d.HN, d.e))
        lo, hi, q, rho = c(d.lo), c(d.hi), c(d.q), dtype(d.rho)
        x0b = c(d.xbar0(x0, u_prev))
        X = np.zeros((N, nb), dtype)
        Z, V, Th, Be, La = (np.zeros((N, nb), dtype) for _ in range(5))
        for _ in range(iters):
            DU = (V - Be) @ J.T
            T1 = Z[1:] - V[1:] + La[1:] - e
            rhs = q.copy()
            rhs[:-1] += rho * (Z[:-1] - Th[:-1] + T1 @ Ab)
            rhs[-1] += rho * (Z[-1] - Th[-1])
            Xn = np.empty_like(X)
            Xn[:-1] = rhs[:-1] @ H.T
            Xn[-1] = HN @ rhs[-1]
            Xk = np.vstack([x0b[None, :], Xn[:-1]])
            BU = DU @ Bb.T
            AX = Xk @ Ab.T
            Zn = np.clip((2 * (Xn + Th) + BU + Be + AX + e - La) / 3, lo, hi)
            Vn = 0.5 * (Zn + BU + Be - AX - e + La)
            Th = Th + Xn - Zn
            Be = Be + BU - Vn
            La = La + Zn - AX - Vn - e
            X, Z, V = Xn, Zn, Vn
        return d.unscale_u(X[:, d.nx:].astype(float))
