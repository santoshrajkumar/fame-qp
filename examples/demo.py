"""Minimal FAME-QP usage.

    python examples/demo.py

1. A random dense QP with n = 1000.
2. Closed-loop MPC on a random linear system, with the LQR-preconditioned
   condensed QP and shifted warm starts.
3. The same MPC with A and B changing at every step: the QP matrices are
   replaced in place with ``update_matrices``.

Requires a CUDA GPU and FAME-QP installed from the repository root
(``pip install -e .``).
"""

import time

import numpy as np

from fame_qp import QP, Solver
from fame_qp.utils import RandomLMPC, random_lmpc, random_qp


def random_qp_demo(n=1000):
    print(f"1. Random dense QP, n = {n}")
    data = random_qp(n, seed=0)                     # n/4 equalities, n/4 inequalities
    qp = QP.create(**data, device="cuda")
    solver = Solver(qp, precision="float64", eps_abs=1e-4, eps_rel=1e-4)
    print(f"   setup {solver.setup_time:.2f} s: {qp.structure} structure, "
          f"{solver.mode} mode, rho = {solver.eng.rho_value:g}")

    t0 = time.perf_counter()
    res = solver.solve()
    z = res.z.cpu().numpy()
    ms = (time.perf_counter() - t0) * 1e3
    print(f"   solve {ms:.2f} ms: {res.status} in {res.iters} iterations, "
          f"objective {qp.objective(res.z):.6g}")

    # A new linear term: the matrices, scaling and penalty are kept, and the
    # last solution is the starting point.
    solver.update(q=data["q"] + 0.1 * np.random.default_rng(1).standard_normal(n))
    t0 = time.perf_counter()
    res = solver.solve()
    res.z.cpu()
    print(f"   re-solve after changing q: {(time.perf_counter() - t0) * 1e3:.2f} ms, "
          f"{res.iters} iterations")


def closed_loop(prob, solver, x, steps, model=None):
    """Run the MPC loop; ``model(k)`` returns (A, B) when the system changes."""
    times, active = [], 0
    for k in range(steps):
        if model is not None:
            A, B = model(k)
            prob = RandomLMPC(A, B, prob.Q, prob.R, prob.Q_N, prob.K, prob.N,
                              prob.u_lo, prob.u_hi)
        P, q, G, l, h = prob.qp_data(x)

        t0 = time.perf_counter()
        if model is not None:
            solver.update_matrices(P=P, G=G)
        solver.update(q=q, l=l, h=h)
        if k > 0:
            solver.warm_start_shifted(*prob.stage_segments())
        res = solver.solve()
        u = prob.first_input(res.z.cpu().numpy(), x)
        times.append((time.perf_counter() - t0) * 1e3)

        active += bool(np.any(np.abs(u) >= 0.999 * prob.u_hi))
        x = prob.step(x, u)
    return np.array(times), active, x


def lmpc_demo(n_u=10, steps=50):
    prob = random_lmpc(n_u=n_u, N=40)
    print(f"\n2. Random LMPC: n_x = {prob.nx}, n_u = {n_u}, N = {prob.N}, QP n = {prob.n}")
    x = prob.initial_state()
    solver = Solver(prob.qp(x), precision="float64", eps_abs=1e-3, eps_rel=1e-3)
    print(f"   setup {solver.setup_time:.2f} s: {solver.structure} structure, {solver.mode} mode")

    solver.solve()                                  # first solve at x0, untimed
    times, active, x = closed_loop(prob, solver, x, steps)
    print(f"   {steps} steps: {times.mean():.2f} ± {times.std(ddof=1):.2f} ms per step, "
          f"input bound active on {active} steps, final |x| = {np.linalg.norm(x):.2e}")


def time_varying_lmpc_demo(n_u=10, steps=50):
    nominal = random_lmpc(n_u=n_u, N=40)
    A0, B0 = nominal.A, nominal.B

    def model(k):
        # Smoothly varying dynamics, starting from the nominal model; the LQR
        # pre-stabilising gain stays the nominal one.
        s = np.sin(2 * np.pi * k / steps)
        return A0 * (1 + 0.03 * s), B0 * (1 + 0.1 * s)

    print(f"\n3. Random LMPC with A and B changing every step (QP n = {nominal.n})")
    x = nominal.initial_state()
    # The stopping test is relative, and this QP's scales are large, so a
    # tighter tolerance than in demo 2 keeps the first input accurate when the
    # warm start comes from a QP with different matrices.
    solver = Solver(nominal.qp(x), precision="float64", eps_abs=1e-5, eps_rel=1e-5,
                    lazy_update=True)                # re-invert only the active penalty
    solver.solve()
    times, active, x = closed_loop(nominal, solver, x, steps, model=model)
    print(f"   {steps} steps: {times.mean():.2f} ± {times.std(ddof=1):.2f} ms per step "
          f"(matrix update and solve), input bound active on {active} steps, "
          f"final |x| = {np.linalg.norm(x):.2e}")


if __name__ == "__main__":
    random_qp_demo()
    lmpc_demo()
    time_varying_lmpc_demo()
