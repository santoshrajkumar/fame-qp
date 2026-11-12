"""Pre-inverted ADMM operators for a bank of penalties.

With M = (P + sigma I + G' diag(rho) G)^-1 and G split into diag(d) and C,
the iterate is collected into r = [r_z; r_c] with

    r_z = sigma z + d o (rho_d o s_d - lam_d),     r_c = rho_c o s_c - lam_c.

Two operator forms are stored:

    single_kernel   W = S M S',   S = [I; C],   size (n + m_c)^2
    two_kernel      W = [M, M C'] plus C,       n (n + 2 m_c) values

so that z~ = W[:n] r + b and, in the stacked form, C z~ = W[n:] r + b[n:],
with b = -W[:, :n] q. In the box structure both forms reduce to W = M.
Inverses are computed in float64 and stored in the working precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch

__all__ = ["Bank", "build_bank", "fill_bank", "rho_grid", "rho_base_vector",
           "EQ_BOOST", "operator_rows"]

EQ_BOOST = 1.0e3
EQ_TOL = 1.0e-10


def rho_grid(rho0: float, n_rho: int, ratio: float = 10.0) -> List[float]:
    """Geometric grid of ``n_rho`` penalties centred on ``rho0``."""
    if n_rho <= 1:
        return [float(rho0)]
    k0 = (n_rho - 1) // 2
    return [float(rho0) * ratio ** (k - k0) for k in range(n_rho)]


def rho_base_vector(l, h, n_d: int, gamma: float, eq_boost: float = EQ_BOOST):
    """Per-row penalty weights: ``eq_boost`` on equality rows and, in the split
    structure, an extra factor ``gamma`` on the dense rows."""
    base = torch.where((h - l) < EQ_TOL, torch.full_like(l, eq_boost),
                       torch.ones_like(l))
    if n_d > 0 and base.numel() > n_d and gamma != 1.0:
        base[n_d:] *= gamma
    return base


def operator_rows(form: str, n: int, m_c: int) -> int:
    return n + m_c if form == "single_kernel" else n


@dataclass
class Bank:
    """Operators ``W_bank[k]`` for the penalties ``rhos[k]``."""

    W_bank: torch.Tensor
    rhos: List[float]
    rho_base: torch.Tensor
    gamma: float
    form: str

    @property
    def n_rho(self) -> int:
        return len(self.rhos)

    def nearest(self, rho_value: float) -> int:
        lv = math.log(max(rho_value, 1e-300))
        return min(range(self.n_rho), key=lambda i: abs(math.log(self.rhos[i]) - lv))

    def nbytes(self) -> int:
        return self.W_bank.numel() * self.W_bank.element_size()


def _inverses(P, d, C, base, rhos, sigma):
    """Batched float64 inverses of P + sigma I + rho_k G' diag(base) G."""
    dev, n = P.device, P.shape[0]
    nd = 0 if d is None else n
    Pw = P.to(torch.float64)
    bw = base.to(torch.float64)
    Gram = torch.zeros(n, n, device=dev, dtype=torch.float64)
    if d is not None:
        Gram += torch.diag(bw[:nd] * d.to(torch.float64) ** 2)
    Cw = None
    if C is not None and C.shape[0]:
        Cw = C.to(torch.float64)
        Gram += Cw.T @ (bw[nd:].unsqueeze(1) * Cw)
    r = torch.tensor(rhos, device=dev, dtype=torch.float64).view(-1, 1, 1)
    eye = torch.eye(n, device=dev, dtype=torch.float64)
    K = (Pw + sigma * eye).unsqueeze(0) + r * Gram.unsqueeze(0)
    L, info = torch.linalg.cholesky_ex(K)
    if bool(info.any()):
        jit = 1e-9 * torch.diagonal(K, dim1=-2, dim2=-1).abs().amax(-1).clamp(min=1.0)
        L = torch.linalg.cholesky(K + jit.view(-1, 1, 1) * eye)
    return torch.cholesky_inverse(L), Cw


def fill_bank(W_bank, P, d, C, base, rhos, sigma, form, only=None):
    """Recompute the bank in place, either fully or only the entries in ``only``."""
    n = P.shape[0]
    ks = list(range(len(rhos))) if only is None else [int(k) for k in only]
    M, Cw = _inverses(P, d, C, base, [rhos[k] for k in ks], sigma)
    for i, k in enumerate(ks):
        if Cw is None:
            W_bank[k].copy_(M[i])
            continue
        top = torch.cat([M[i], M[i] @ Cw.T], dim=1)
        W_bank[k, :n].copy_(top)
        if form == "single_kernel":
            W_bank[k, n:].copy_(Cw @ top)


def build_bank(P, d, C, l, h, sigma, rhos, gamma, form, dtype,
               eq_boost: float = EQ_BOOST) -> Bank:
    """Build a bank for scaled data in internal row order."""
    n = P.shape[0]
    nd = 0 if d is None else n
    mc = 0 if C is None else C.shape[0]
    base = rho_base_vector(l, h, nd, gamma, eq_boost)
    W_bank = torch.empty((len(rhos), operator_rows(form, n, mc), n + mc),
                         device=P.device, dtype=dtype)
    fill_bank(W_bank, P, d, C, base, rhos, sigma, form)
    return Bank(W_bank=W_bank, rhos=[float(r) for r in rhos],
                rho_base=base.to(dtype), gamma=float(gamma), form=form)
