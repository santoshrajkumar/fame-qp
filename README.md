# fame_qp — GPU QP solver for MPC with automatic execution mode

```
min_z  ½ zᵀPz + qᵀz     s.t.   l ≤ G z ≤ h          (P ∈ S₊ⁿ, G ∈ ℝ^{m×n})
```

## Installation

```bash
pip install -e .            # requires a CUDA GPU, PyTorch and Triton
```

## Usage

```python
from fame_qp import QP, Solver
qp  = QP.create(P, q, G, l, h, device="cuda")      # G=None means G = I
sol = Solver(qp, eps_abs=1e-4, eps_rel=1e-4)        # calibrates / selects at setup
res = sol.solve()
res.z, res.s, res.lam, res.status, res.info["mode"], res.info["precision"]

sol.update(q=q2, l=l2, h=h2)       # MPC loop: last iterate and rho carry over
sol.update_matrices(P=P2, G=G2)    # same structure; bank re-inverted in place
sol.warm_start_shifted([(0, n_u, N)],                   # z = col(u_0..u_{N-1})
                       [(0, n_b, N), (n_b * N, n_u, N)])  # row groups, by stage
```

**Shifted warm start.** In receding-horizon control the next QP is the last one
moved one stage along the horizon, so the best starting point is the last
solution shifted by one stage (stage k+1 becomes stage k, the last stage is
held). `warm_start_shifted(z_segments, lam_segments)` does this for the primal
and the duals; a segment `(start, size, count)` is `count` stages of `size`
entries. Call it after `update` / `update_matrices`: the copy `s` is rebuilt
from the current `G`, `l`, `h`. `MPCController` uses it by default
(`shift=True`, segments from `LinearMPC.stage_segments()`).

LMPC (condensed, `G = [𝓑; I]`, optional LQR pre-stabilisation, sparse form for references):

```python
from fame_qp.mpc import LinearMPC
mpc  = LinearMPC(A, B, Q, R, Q_N, N=40, u_lo=-1, u_hi=1, x_lo=-2, x_hi=2)
ctrl = mpc.controller(x_hat)
u0   = ctrl.step(x_hat)
```

## Benchmark problems

`fame_qp.utils` generates the random problems used in the paper, seeded as in ReLU-QP:

```python
from fame_qp.utils import random_qp, random_lmpc

qp = QP.create(**random_qp(n=400, seed=0))   # n/4 dense equalities, n/4 one-sided inequalities

prob = random_lmpc(n_u=10, N=40)             # n_x = 3 n_u, LQR-preconditioned condensed QP
x = prob.initial_state()                     # paper's initial-state scale for n_u
sol = Solver(prob.qp(x), precision="float64", eps_abs=1e-3, eps_rel=1e-3)
u0 = prob.first_input(sol.solve().z.cpu().numpy(), x)
x = prob.step(x, u0)
```

## Example

[`examples/demo.py`](examples/demo.py) solves a random QP with n = 1000, runs
closed-loop MPC on a random linear system, and runs the same MPC with A and B
changing at every step, using `update_matrices`:

```bash
python examples/demo.py
```

## Notation in the code

| Symbol | Code | Meaning |
|---|---|---|
| z, P, q, G, l, h | `z, P, q, G, l, h` | the QP above |
| s = G z | `s` | ADMM copy of the constrained quantities |
| λ | `lam` | Lagrange multiplier of l ≤ G z ≤ h (dual of s = G z); `lambda` is reserved in Python |
| ρ, σ, α | `rho, sigma, alpha` | penalty, regularisation, relaxation |
| K = P + σI + Gᵀdiag(ρ)G, M = K⁻¹ | | pre-inverted for a bank of ρ |
| diag(d), C | `d, C` | diagonal rows and dense rows of G (n_d = n or 0, m_c rows) |
| 𝒜, 𝓑, 𝒬, ℛ | `Acal, Bcal, Qcal, Rcal` | condensing matrices (`mpc.py`) |

The ADMM paragraph of the paper still uses x, z, A, u for z, s, G, h.

Structures, detected from G: **box** `G = diag(d)`; **split** `G = [C; diag(d)]`
or `[diag(d); C]`; **general** otherwise (`C = G`).

## Execution modes

| Mode | Role of the CUDA graph |
|---|---|
| **Persistent-kernel solve** (`persistent`) | Not used for looping: the whole solve is one kernel launch (box structure, tile within the register file). A one-node graph only cuts the launch cost; ρ adapts between launches of `persistent_budget` iterations |
| **Single-kernel iteration** (`single_kernel`) | Replays a block of `check_interval` iterations, one kernel each, then the on-device stopping test |
| **Two-kernel iteration** (`two_kernel`) | Replays a block of iterations, two dependent kernels each, then the on-device stopping test |

Operators (`operator.py`): single-kernel `W = S M Sᵀ`, `S = [I; C]`, size (n+m_c)²;
two-kernel `[M, M Cᵀ]` plus `C`, n(n+2m_c) values. They differ by exactly m_c²
values per iteration, against one kernel launch.

## Automatic selection (`mode="auto"`, default)

1. **Calibration, once per device and precision** (`calibration.py`, ~1 s with
   Triton's compile cache warm): graph-node cost τ, host-read cost, streaming
   time vs operator size (captures L2 vs DRAM), and for the persistent kernel
   the registers/spills reported by the compiler per tile, giving the largest
   spill-free tile on this GPU.
2. **Setup test, once per problem shape**: predict single vs two kernels from
   the calibration, drop candidates > 1.5× the best prediction or whose bank
   does not fit in memory, time the rest on placeholder operators of the real
   shape (tiles tuned), keep the single kernel unless two kernels win by > 10 %.
3. **Persistent vs graph replay**: predicted solve time from the iteration count
   of the ρ probe. If the persistent mode wins, its starting ρ is re-probed for
   its schedule (ρ can only change between launches) and the comparison redone.

Force a mode with `Solver(qp, mode="persistent" | "single_kernel" | "two_kernel")`.

## Precision

`precision="float32"` (default), `"float64"`, or `"auto"` (optional, two stages):

1. κ(K) by power iterations on K and K⁻¹ (float64). `u₃₂κ ≤ 0.01ε` → float32;
   `u₃₂κ ≥ ε` → float64; otherwise stage 2.
2. Probe in both precisions; drop one that does not reach ε; take the shorter
   predicted solve, preferring float64 within `precision_margin` (20 %).

Mode is selected per precision (float64 changes the persistent limit and the
crossover). The bank is always inverted in float64.

## Penalty selection

Probe trials adapt ρ as the solve does and record when the test passes at
`probe_tighten`×, √`probe_tighten`× and 1× the tolerance; candidates are compared
at the tightest level some candidate reached. The bank is re-centred when the
best start is at an end. In the split structure γ = ρ_dense/ρ_diag is searched
at the probed ρ, then ρ is probed again.

## Cache

`$FAME_QP_CACHE`, else `$XDG_CACHE_HOME/fame_qp`, else `~/.cache/fame_qp`:
`calibration.json` (per device: GPU, compute capability, driver, CUDA, Triton,
package version) and `modes.json` (per device, precision and problem shape).
Read once per process at the first setup; never touched by `solve`. Atomic,
file-locked writes; in memory only if the directory is not writable.
`fame_qp.clear_cache()` forgets everything; `Settings(retune=True)` re-measures.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests     # 48 tests
```

## Reproducing the paper

[`quick_demo.ipynb`](quick_demo.ipynb) regenerates the figures and tables of the
experimental results from the saved benchmark results, and runs short versions
of the experiments live on your GPU. The benchmark
scripts and the setup of the baselines are in [`experiments/`](experiments/README.md).
