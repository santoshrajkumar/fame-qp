# Third-party baselines

No third-party code is stored in this repository. `setup_externals.sh` fetches
each item below at the exact version used in the paper into
`experiments/external/`, which git ignores. None of them is modified.

| Baseline | Source | Version | License | Used in |
|---|---|---|---|---|
| cuOSQP | [osqp/cuosqp](https://github.com/osqp/cuosqp) | `2a045f8`, submodule `osqp_sources` at `837c7a3` | Apache-2.0 | random QP, random LMPC |
| CuPIQP | [PREDICT-EPFL/cupiqp](https://github.com/PREDICT-EPFL/cupiqp) | `d538dda` | BSD-2-Clause | random QP, random LMPC, quadrotor |
| πMPC (PyTorch port) | [SOLARIS-JHU/PiMPC.jl](https://github.com/SOLARIS-JHU/PiMPC.jl) | `6aa0950` | MIT | random LMPC, CUDA MPC validation |
| ReLU-QP | [RoboticExplorationLab/ReLUQP-py](https://github.com/RoboticExplorationLab/ReLUQP-py) | `83aa082` (pip) | MIT | random QP, random LMPC, G1 |
| KQ-LMPC quadrotor | [santoshrajkumar/kq-lmpc-quadrotor-updated](https://github.com/santoshrajkumar/kq-lmpc-quadrotor-updated) | `60fb4c0` | MIT | quadrotor |
| Unitree G1 model | [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie), `unitree_g1/` | `cda5236` | BSD-3-Clause | G1 |
| CUDA 11.8 toolkit (nvcc, cudart, CCCL, cuBLAS 11.11, cuSPARSE 11.7) | [NVIDIA redistributables](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_11.8.0.json), SHA-256 checked | 11.8.89 | NVIDIA CUDA EULA | building and running cuOSQP |
| `helper_cuda.h`, `helper_string.h` | [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) `Common/` | tag `v12.5`, SHA-256 checked | BSD-3-Clause | building cuOSQP |

The CPU solvers (OSQP, PIQP, ProxQP, Clarabel, DAQP), qpax and MuJoCo are
installed from PyPI at the versions in `requirements-bench.txt`.

## cuOSQP

The official cuOSQP build is single precision (its `setup.py` passes
`-DDFLOAT=ON`) and is used as is. Its CUDA code calls cuSPARSE functions that
CUDA 12 removed (`cusparseCsrmvEx`, `cusparseAlgMode_t`), so it is built with
the CUDA 11.8 toolkit rather than patched. The build options are passed through
environment variables; no file of cuOSQP is changed. At run time `env.sh` adds
the CUDA 11.8 libraries to `LD_LIBRARY_PATH`. They have CUDA 11 sonames, so
they load alongside PyTorch's CUDA 13 libraries without conflict. CUDA 11.8
supports GPUs up to compute capability 9.0.

## CUDA MPC

The authors of CUDA MPC (Akbari & Greeff) have not released their code.
`baselines/cudampc/` is our implementation from the paper: one cooperative
persistent kernel, templated on the floating-point type, following Sec. V of
the paper (eqs. (8)–(13) per stage, an M × n̄x thread block per sub-horizon with
M from eqs. (14)–(16), iterates in shared memory, synchronisation only between
neighbouring blocks). Differences from the paper:

* Blocks wait for their neighbours' current boundary values, so the iterates
  are those of synchronous πMPC; the paper allows a delay of one iteration.
* The problems are linear and time-invariant, so each block stores one copy of
  the system matrices instead of one per stage.
* There is no SCP outer loop, since the problems are linear. Each MPC step is
  one kernel launch and one copy of the solution.
* No preconditioning, as in the paper (`precond=True` is available).

`python experiments/baselines/cudampc/validate.py` checks the kernel against a
NumPy implementation of the same iteration and against πMPC's PyTorch code on
the random-LMPC systems. `float64` agrees to 1.5e-14 and `float32` to 7e-6,
relative to |u|.
