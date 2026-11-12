# Experiments

Scripts that reproduce the results in the paper. The solver itself does not
need anything in this folder.

## Setup

Requires Linux, an NVIDIA GPU with a CUDA 13 driver, Python 3.10, `git`,
`curl` and, for cuOSQP, `cmake`. From the repository root:

```bash
bash experiments/setup_externals.sh   # baselines at pinned versions, plus all pip packages
source experiments/env.sh             # CUDA library paths; run in every new shell
```

The script installs FAME-QP, the packages in `requirements-bench.txt`, CuPIQP
and cuOSQP into the active Python environment, and downloads the other
baselines into `experiments/external/`. Use `--no-cuosqp` to skip the CUDA 11.8
toolkit and cuOSQP. `EXTERNALS.md` lists every third-party component with its
source, version and license.

## Experiments

| Folder | Script | Paper |
|---|---|---|
| `random_qp/` | `random_qp_cpu_gpu.py` | random dense QPs, all solvers (Table I) |
| | `random_qp_fame_f32.py` | the same QPs, FAME-QP in `float32` |
| | `viz_tab_random_qp.py` | figure and table from the saved data |
| `random_lmpc/` | `random_lmpc_gpu.py` | closed-loop MPC on random linear systems (Table II) |
| | `viz_tab_random_lmpc.py` | figure and table from the saved data |
| `g1_balance/` | `g1_balance_gpu.py` | Unitree G1 balance MPC in MuJoCo |
| | `g1_lqr_balance.py` | linearisation and LQR design behind `g1_lqr_matrices.npz` |
| `quadrotor/` | `kqlmpc_gpu.py` | KQ-LMPC trajectory tracking on the Skydio X2 in MuJoCo |
| `baselines/cudampc/` | `validate.py` | checks our CUDA MPC implementation against πMPC |

Each script documents its protocol and options at the top of the file, and
`--help` lists the options. For example:

```bash
python experiments/random_qp/random_qp_cpu_gpu.py --sizes 50 100 --solvers fame-qp osqp
python experiments/random_lmpc/random_lmpc_gpu.py --nus 10 20 --methods fame-qp reluqp
python experiments/g1_balance/g1_balance_gpu.py
```

Results are written to `experiments/data/<experiment>/`: raw results in
`raw/`, and the processed CSV files, tables and figures made from them. The
raw results behind the paper are included, and `quick_demo.ipynb` in the
repository root regenerates the paper's figures and tables from them. Timings
depend on the GPU; the paper used an RTX 3080 Laptop GPU.

FAME-QP keeps its device calibration in `~/.cache/fame_qp`. Set
`FAME_QP_CACHE` to use a fresh cache.
