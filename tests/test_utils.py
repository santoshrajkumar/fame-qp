import numpy as np
import pytest
import torch

from fame_qp import QP, Solver
from fame_qp.utils import random_lmpc, random_qp

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_random_qp_shape_and_seed():
    a, b = random_qp(80, seed=3), random_qp(80, seed=3)
    assert a["P"].shape == (80, 80) and a["G"].shape == (40, 80)
    assert np.array_equal(a["P"], b["P"]) and np.array_equal(a["q"], b["q"])
    assert np.allclose(a["P"], a["P"].T)
    assert np.array_equal(a["l"][:20], a["h"][:20])
    assert np.all(np.isinf(a["h"][20:]))


def test_random_lmpc_structure():
    p = random_lmpc(n_u=4, N=10)
    P, q, G, l, h = p.qp_data(p.initial_state())
    assert P.shape == (40, 40) and G.shape == (40, 40)
    assert np.allclose(np.triu(G, 1), 0)
    assert not np.allclose(G, np.diag(np.diag(G)))
    assert np.max(np.abs(np.linalg.eigvals(p.A - p.B @ p.K))) < 1.0


@cuda
def test_random_qp_solves():
    s = Solver(QP.create(**random_qp(100, seed=0)), precision="float64",
               eps_abs=1e-6, eps_rel=1e-6)
    assert s.structure == "general"
    assert s.solve().status == "solved"


@cuda
def test_random_lmpc_closed_loop():
    p = random_lmpc(n_u=4, N=10)
    x = p.initial_state(4.0)
    s = Solver(p.qp(x), precision="float64", eps_abs=1e-4, eps_rel=1e-4)
    for k in range(5):
        P, q, G, l, h = p.qp_data(x)
        s.update(q=q, l=l, h=h)
        if k:
            s.warm_start_shifted(*p.stage_segments())
        r = s.solve()
        assert r.status == "solved"
        u = p.first_input(r.z.cpu().numpy(), x)
        assert np.all(u <= p.u_hi + 1e-3) and np.all(u >= p.u_lo - 1e-3)
        x = p.step(x, u)
