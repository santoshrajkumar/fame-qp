"""Random dense QPs (Bishop et al., ReLU-QP, ICRA 2024).

The generator matches ``reluqp.utils.rand_qp`` with the same seeding:
P = 2(R'R + I), ``n_eq`` dense equality rows followed by ``n_ineq`` dense
one-sided inequality rows, of which about 30 % are active at the solution.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["random_qp"]


def random_qp(n: int, seed: int = 0, n_eq: Optional[int] = None,
              n_ineq: Optional[int] = None) -> dict:
    """Return a random QP as a dict with keys ``P, q, G, l, h``.

    ``n_eq`` and ``n_ineq`` default to ``n // 4``. The equality rows come
    first in G; the inequality rows have ``h = +inf``. The dict can be passed
    directly to :meth:`fame_qp.QP.create`::

        qp = QP.create(**random_qp(200, seed=0))
    """
    n_eq = int(n / 4) if n_eq is None else int(n_eq)
    n_ineq = int(n / 4) if n_ineq is None else int(n_ineq)
    randn = np.random.RandomState(seed).randn

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

    return dict(P=H, q=g, G=np.vstack((Aeq, C)), l=np.concatenate((b, d)),
                h=np.concatenate((b, np.full(n_ineq, np.inf))))
