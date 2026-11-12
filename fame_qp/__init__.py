"""FAME-QP: fused ADMM with multimode execution for dense QPs on the GPU.

Solves

    minimize    1/2 z'Pz + q'z
    subject to  l <= Gz <= h

Example
-------
>>> from fame_qp import QP, Solver
>>> qp = QP.create(P, q, G, l, h, device="cuda")
>>> res = Solver(qp, eps_abs=1e-4, eps_rel=1e-4).solve()
>>> res.z, res.lam, res.status, res.info["mode"]

Linear MPC in condensed form:

>>> from fame_qp.mpc import LinearMPC
>>> mpc = LinearMPC(A, B, Q, R, Q_N, N=30, u_lo=-1, u_hi=1)
>>> ctrl = mpc.controller(x_hat)
>>> u0 = ctrl.step(x_hat)
"""

__version__ = "0.2.0"

from .problem import QP
from .solver import Result, Settings, Solver
from .calibration import clear_cache, get_calibration

__all__ = ["QP", "Solver", "Settings", "Result", "get_calibration", "clear_cache"]
