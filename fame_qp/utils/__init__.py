"""Benchmark problem generators.

>>> from fame_qp import QP, Solver
>>> from fame_qp.utils import random_qp, random_lmpc
>>> qp = QP.create(**random_qp(200, seed=0))
>>> prob = random_lmpc(n_u=10, N=40)
>>> x = prob.initial_state()
>>> res = Solver(prob.qp(x)).solve()
>>> u0 = prob.first_input(res.z.cpu().numpy(), x)
"""

from .random_lmpc import X0_SCALE, RandomLMPC, ihlqr, random_linear_system, random_lmpc
from .random_qp import random_qp

__all__ = ["random_qp", "random_lmpc", "RandomLMPC", "random_linear_system",
           "ihlqr", "X0_SCALE"]
