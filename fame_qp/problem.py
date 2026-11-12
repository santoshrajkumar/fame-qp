"""QP data and constraint-structure detection.

The rows of G are stored as a diagonal block diag(d) and a dense block C:

    box      G = diag(d)                     n_d = n, m_c = 0
    split    G = [diag(d); C] or [C; diag(d)]  n_d = n, m_c > 0
    general  anything else                    n_d = 0, C = G

Internally the diagonal rows come first; ``perm`` maps internal row k to the
user's row ``perm[k]``, so all inputs and outputs use the user's row order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

__all__ = ["QP", "BIG"]

# Finite stand-in for an infinite bound.
BIG = float(torch.finfo(torch.float32).max) / 4


def to_tensor(M, device, dtype=torch.float64):
    """Convert numpy, scipy.sparse or torch input to a dense tensor."""
    if M is None:
        return None
    if hasattr(M, "toarray"):
        M = M.toarray()
    if isinstance(M, torch.Tensor):
        return M.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(M, dtype=np.float64), device=device, dtype=dtype)


def _diagonal_of(B: torch.Tensor) -> Optional[torch.Tensor]:
    """Return diag(B) if B is square, diagonal and nonsingular, else None."""
    r, c = B.shape
    if r != c:
        return None
    d = torch.diagonal(B)
    if bool(torch.any(d == 0)) or bool(torch.any(B - torch.diag(d) != 0)):
        return None
    return d.clone()


@dataclass
class QP:
    """A QP stored on the device in float64. Build it with :meth:`QP.create`."""

    P: torch.Tensor
    q: torch.Tensor
    d: Optional[torch.Tensor]
    C: Optional[torch.Tensor]
    l: torch.Tensor
    h: torch.Tensor
    perm: torch.Tensor

    @property
    def n(self) -> int:
        return self.P.shape[0]

    @property
    def n_d(self) -> int:
        return 0 if self.d is None else self.n

    @property
    def m_c(self) -> int:
        return 0 if self.C is None else self.C.shape[0]

    @property
    def m(self) -> int:
        return self.n_d + self.m_c

    @property
    def structure(self) -> str:
        if self.d is None:
            return "general"
        return "box" if self.C is None else "split"

    @property
    def device(self):
        return self.P.device

    @staticmethod
    def create(P, q, G=None, l=None, h=None, device="cuda",
               detect_structure: bool = True) -> "QP":
        """Build a QP.

        ``G=None`` means G = I. With ``detect_structure=False`` all rows are
        treated as dense (general structure).
        """
        dev = torch.device(device)
        P = to_tensor(P, dev)
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError(f"P must be square, got {tuple(P.shape)}")
        P = 0.5 * (P + P.T)
        n = P.shape[0]
        q = to_tensor(q, dev).reshape(-1)
        if q.numel() != n:
            raise ValueError(f"q must have {n} entries, got {q.numel()}")

        G = to_tensor(G, dev)
        if G is None:
            m = n
            d = torch.ones(n, device=dev, dtype=torch.float64)
            C, perm = None, torch.arange(n, device=dev)
        else:
            if G.ndim != 2 or G.shape[1] != n:
                raise ValueError(f"G must be (m, {n}), got {tuple(G.shape)}")
            m = G.shape[0]
            d, C, perm = None, G, torch.arange(m, device=dev)
            if detect_structure:
                d, C, perm = QP._split(G)

        l = QP._bound(l, m, -np.inf, dev)
        h = QP._bound(h, m, np.inf, dev)
        if bool(torch.any(l > h)):
            raise ValueError("infeasible bounds: some l > h")
        return QP(P=P, q=q, d=d, C=C, l=l[perm].contiguous(),
                  h=h[perm].contiguous(), perm=perm)

    @staticmethod
    def _split(G: torch.Tensor):
        """Look for a diagonal n-row block at the top or bottom of G."""
        m, n = G.shape
        dev = G.device
        if m == n:
            d = _diagonal_of(G)
            if d is not None:
                return d, None, torch.arange(n, device=dev)
        if m > n:
            d = _diagonal_of(G[:n])
            if d is not None:
                return d, G[n:].contiguous(), torch.arange(m, device=dev)
            d = _diagonal_of(G[m - n:])
            if d is not None:
                perm = torch.cat([torch.arange(m - n, m, device=dev),
                                  torch.arange(0, m - n, device=dev)])
                return d, G[: m - n].contiguous(), perm
        return None, G.contiguous(), torch.arange(m, device=dev)

    @staticmethod
    def _bound(v, m, default, dev):
        if v is None:
            v = torch.full((m,), default, device=dev, dtype=torch.float64)
        else:
            v = to_tensor(v, dev).reshape(-1)
            if v.numel() == 1:
                v = v.expand(m).clone()
            if v.numel() != m:
                raise ValueError(f"bound must have {m} entries, got {v.numel()}")
        return torch.clamp(v, -BIG, BIG)

    def G_user(self) -> torch.Tensor:
        """Dense G in the user's row order."""
        parts = []
        if self.d is not None:
            parts.append(torch.diag(self.d))
        if self.C is not None:
            parts.append(self.C)
        Gi = torch.cat(parts, 0)
        G = torch.empty_like(Gi)
        G[self.perm] = Gi
        return G

    def objective(self, z: torch.Tensor) -> float:
        z = z.to(torch.float64)
        return float(0.5 * z @ (self.P @ z) + self.q @ z)
