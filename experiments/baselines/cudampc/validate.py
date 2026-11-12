#!/usr/bin/env python
"""Validate the CUDA MPC kernel (float32 and float64) on the systems of the
paper's closed-loop table (bench/run_random_mpc.py: n_x = 3 n_u, N = 40).

  1. kernel == NumPy reference of eqs. (8)-(13), for several partitions of the
     horizon into blocks (automatic M from eqs. (14)-(16), M = 1, M = 7);
  2. kernel == piMPC's own PyTorch code (external/PiMPC.jl/pytorch, no
     preconditioning, no acceleration: the iteration CUDA MPC describes);
  3. both precisions, with and without piMPC's diagonal preconditioning.

    source bench/env.sh
    python experiments/baselines/cudampc/validate.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(EXP)
sys.path[:0] = [ROOT, os.path.dirname(HERE), os.path.join(EXP, "external", "PiMPC.jl"),
                 os.path.join(EXP, "random_lmpc")]

from cudampc import CudaMPC, PiADMMReference, _Data  # noqa: E402
from random_lmpc_gpu import build  # noqa: E402

ITERS = 300


def pimpc_u(ocp, x0, iters):
    import torch
    from pytorch.pimpc import Model
    nx, nu = ocp.nx, ocp.nu
    m = Model()
    m.setup(A=ocp.A, B=ocp.B, Np=ocp.N, C=np.eye(nx), Wy=ocp.Q, Wu=ocp.R,
            Wdu=np.zeros((nu, nu)), Wf=ocp.QN, umin=ocp.umin, umax=ocp.umax,
            rho=1.0, tol=-1.0, maxiter=iters, precond=False, accel=False,
            device="cuda", dtype=torch.float64)
    r = m.solve(x0, np.zeros(nu), np.zeros(nx), np.zeros(nu))
    return r.u.detach().cpu().numpy().T          # N x nu


def main():
    worst = {}
    for nu in (10, 30, 50):
        ocp, d0, K = build(nu)
        x0 = 4.0 * d0
        for precond in (False, True):
            d = _Data.from_ocp(ocp, rho=1.0, precond=precond)
            ref = PiADMMReference(d).solve(x0, ITERS)
            scale = max(1.0, np.max(np.abs(ref)))
            for dtype in ("float64", "float32"):
                for ms in (0, 1, 7):
                    cm = CudaMPC(d, dtype=dtype, max_stages=ms)
                    U = cm.solve(x0, ITERS)
                    err = np.max(np.abs(U - ref)) / scale
                    key = (dtype, precond)
                    worst[key] = max(worst.get(key, 0.0), err)
                    print(f"nu={nu:2d} precond={precond!s:5s} {dtype}  M={cm.M:2d} B={cm.B:2d} "
                          f"threads={cm.threads:4d}  |kernel - numpy| / |u| = {err:.1e}",
                          flush=True)
                    del cm
            if not precond:
                up = pimpc_u(ocp, x0, ITERS)
                e = np.max(np.abs(up - ref)) / scale
                worst[("pimpc", False)] = max(worst.get(("pimpc", False), 0.0), e)
                print(f"nu={nu:2d} numpy reference vs piMPC PyTorch (float64): {e:.1e}",
                      flush=True)
    print("\nworst relative differences:")
    for k, v in worst.items():
        print(f"  {k}: {v:.1e}")
    ok = (worst[("float64", False)] < 1e-10 and worst[("float64", True)] < 1e-10
          and worst[("pimpc", False)] < 1e-10 and worst[("float32", False)] < 1e-3)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
