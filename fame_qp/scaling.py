"""Modified Ruiz equilibration (Stellato et al., 2020, Sec. 5.1).

The scaled problem is

    P -> c D P D,   q -> c D q,   G -> E G D,   l -> E l,   h -> E h,

and its solution maps back as z = D z', s = E^-1 s', lam = c^-1 E lam'.
A diagonal block of G stays diagonal, so the detected structure is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

__all__ = ["Scaling", "ruiz_equilibrate"]

_MIN_SCAL = 1e-4
_MAX_SCAL = 1e4


@dataclass
class Scaling:
    D: torch.Tensor
    E: torch.Tensor
    c: torch.Tensor
    Dinv: torch.Tensor
    Einv: torch.Tensor
    cinv: torch.Tensor


def _limit(v):
    return torch.clamp(v, _MIN_SCAL, _MAX_SCAL)


def _limit_norm(v):
    # As in OSQP, near-zero norms are left unscaled rather than clamped.
    v = torch.where(v < _MIN_SCAL, torch.ones_like(v), v)
    return torch.clamp(v, max=_MAX_SCAL)


def ruiz_equilibrate(P, q, d: Optional[torch.Tensor], C: Optional[torch.Tensor],
                     l, h, iters: int = 10, cost_scaling: bool = True):
    """Return the scaled ``(P, q, d, C, l, h)`` and the :class:`Scaling`."""
    dev, dt = P.device, P.dtype
    n = P.shape[0]
    nd = 0 if d is None else n
    mc = 0 if C is None else C.shape[0]
    m = nd + mc
    P, q, l, h = P.clone(), q.clone(), l.clone(), h.clone()
    d = None if d is None else d.clone()
    C = None if C is None else C.clone()
    D = torch.ones(n, device=dev, dtype=dt)
    E = torch.ones(m, device=dev, dtype=dt)
    c = torch.ones((), device=dev, dtype=dt)

    for _ in range(iters):
        col = torch.amax(torch.abs(P), dim=0)
        rows = []
        if d is not None:
            col = torch.maximum(col, torch.abs(d))
            rows.append(torch.abs(d))
        if mc:
            col = torch.maximum(col, torch.amax(torch.abs(C), dim=0))
            rows.append(torch.amax(torch.abs(C), dim=1))
        dz = _limit(1.0 / torch.sqrt(_limit_norm(col)))
        P.mul_(dz.unsqueeze(0)).mul_(dz.unsqueeze(1))
        q.mul_(dz)
        D.mul_(dz)
        if rows:
            ds = _limit(1.0 / torch.sqrt(_limit_norm(torch.cat(rows))))
            if d is not None:
                d.mul_(ds[:nd]).mul_(dz)
            if mc:
                C.mul_(dz.unsqueeze(0)).mul_(ds[nd:].unsqueeze(1))
            l.mul_(ds)
            h.mul_(ds)
            E.mul_(ds)
        if cost_scaling:
            mean_col = torch.mean(torch.amax(torch.abs(P), dim=0))
            g = 1.0 / _limit_norm(torch.maximum(mean_col, torch.amax(torch.abs(q))))
            P.mul_(g)
            q.mul_(g)
            c.mul_(g)

    sc = Scaling(D=D, E=E, c=c, Dinv=1.0 / D, Einv=1.0 / E, cinv=1.0 / c)
    return P, q, d, C, l, h, sc
