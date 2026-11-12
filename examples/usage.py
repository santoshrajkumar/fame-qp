"""Basic FAME-QP usage.

    python examples/usage.py

Solves  minimize 1/2 z'Pz + q'z  subject to  l <= Gz <= h  on the GPU, then
updates the problem between solves and runs a small linear MPC loop.
"""

import numpy as np

from fame_qp import QP, Solver
from fame_qp.mpc import LinearMPC

rng = np.random.default_rng(0)

# A QP with n variables and m constraint rows.
n, m = 50, 30
M = rng.standard_normal((n, n))
P = M.T @ M + np.eye(n)
q = rng.standard_normal(n)
G = rng.standard_normal((m, n))
l, h = -np.ones(m), np.ones(m)

qp = QP.create(P, q, G, l, h, device="cuda")
solver = Solver(qp, precision="float64", eps_abs=1e-6, eps_rel=1e-6)
res = solver.solve()

z = res.z.cpu().numpy()          # solution
lam = res.lam.cpu().numpy()      # multipliers of l <= Gz <= h
print(f"QP: {res.status} in {res.iters} iterations "
      f"({qp.structure} structure, {res.info['mode']} mode, {res.info['precision']})")
print(f"    objective {qp.objective(res.z):.6f}, "
      f"stationarity |Pz + q + G'lam| = {np.abs(P @ z + q + G.T @ lam).max():.1e}")

# New q and bounds: the setup is kept and the last solution is the starting point.
solver.update(q=q + 0.1, l=l - 0.5, h=h + 0.5)
res = solver.solve()
print(f"    after update: {res.status} in {res.iters} iterations")

# New P (and/or G) with the same constraint structure.
solver.update_matrices(P=2 * P)
res = solver.solve()
print(f"    after update_matrices: {res.status} in {res.iters} iterations")

# Without G the constraints are bounds on z: l <= z <= h.
box = Solver(QP.create(P, q, l=-0.1, h=0.1), precision="float64")
res = box.solve()
print(f"Box QP: {res.status}, max |z| = {res.z.abs().max().item():.3f}")

# Linear MPC: a double integrator driven to the origin, |u| <= 1, |velocity| <= 1.
dt = 0.1
A = np.array([[1.0, dt], [0.0, 1.0]])
B = np.array([[0.5 * dt**2], [dt]])
mpc = LinearMPC(A, B, Q=np.eye(2), R=0.1 * np.eye(1), Q_N=10 * np.eye(2), N=30,
                u_lo=-1.0, u_hi=1.0, x_lo=[-10.0, -1.0], x_hi=[10.0, 1.0])

x = np.array([3.0, 0.0])
ctrl = mpc.controller(x, precision="float64", eps_abs=1e-5, eps_rel=1e-5)
for k in range(100):
    u = ctrl.step(x)              # updates the QP for x, warm starts and solves
    x = A @ x + B @ u
print(f"MPC: after 100 steps x = {np.round(x, 4)}, last solve {ctrl.last.iters} iterations")
