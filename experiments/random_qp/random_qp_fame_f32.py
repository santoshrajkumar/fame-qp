#!/usr/bin/env python
"""Table I random dense QPs, FAME-QP only, in float32.

Same problems and protocol as random_qp_cpu_gpu.py, with FAME-QP's
working precision set to float32 (the bank is still inverted in float64, as
FAME-QP always does).  Self-contained: imports only fame_qp, numpy and
clarabel (for the reference).

Problems
    ReLU-QP's random dense QPs (``reluqp.utils.rand_qp``, same seeding):
    H = 2(R'R + I), n/4 dense equality and n/4 dense inequality rows.
    Five timed instances per size (seeds 0-4).

Protocol
    Tolerance eps swept over 1e-3 ... 1e-8, loose to tight; reported at the
    first eps whose solution is within ``--target`` (infinity norm, default
    1e-3) of a Clarabel 1e-10 reference on all five instances.  Per instance:
    one untimed accuracy check, warm-up solves, then 1-30 timed cold solves
    (wall clock, solution copied to the host).  Setup -- bank inversion, setup
    test, penalty probe -- runs on a held-out instance (seed 5) and is not
    timed; each timed instance is loaded through update_matrices/update.
    Each size runs in its own spawned process with a wall-clock cap.

Output (``--out``, default data/random_qp_fame_f32/)
    raw/random_qp_fame_f32.json   settings, machine, and per n the full sweep:
                                  every eps tried, each instance's accuracy
                                  check, and every timed solve (time,
                                  iterations, final rho, residuals, status)
    raw/run.log                   console log
    refs/ref_n{n}.npz             Clarabel references (copied from
                                  data/random_qp_cpu_gpu/refs when present,
                                  else computed)
    processed/summary.csv         one row per n: reported eps, mean / std /
                                  median / min / max over instances, the five
                                  per-instance medians, iterations, setup
    processed/solves.csv          one row per timed solve

The JSON has the same layout as random_qp_cpu_gpu.json (solver key
``fame-qp-f32``).  It is rewritten after every size; a rerun skips sizes
already done (``--redo`` to repeat them).

    python experiments/random_qp/random_qp_fame_f32.py
    python experiments/random_qp/random_qp_fame_f32.py --sizes 20 50
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # experiments/
ROOT = os.path.dirname(EXP)                                      # repository root
sys.path.insert(0, ROOT)                       # fame_qp package

KEY = "fame-qp-f32"
PRECISION = "float32"
SIZES = [20, 50, 100, 150, 200, 220, 250, 300, 350, 400, 600, 800, 1000,
         1100, 1200, 1400]
EPSS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
CHECK = 10                # iterations between convergence checks (as Table I)
N_SEEDS = 5               # timed instances: seeds 0..4
HELD_OUT = N_SEEDS        # setup / penalty probe instance: seed 5
SHARED_REFS = os.path.join(EXP, "data", "random_qp_cpu_gpu", "refs")
LOG = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOG:
        with open(LOG, "a") as f:
            f.write(s + "\n")


# ---------------------------------------------------------------------------
# problems and reference
# ---------------------------------------------------------------------------
def rand_qp(n, seed):
    """reluqp.utils.rand_qp(nx=n, n_eq=n//4, n_ineq=n//4, seed), as
    l <= A x <= u with the n_eq equality rows first."""
    n_eq = n_ineq = int(n / 4)
    np.random.seed(seed)
    randn = np.random.randn
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
    return dict(P=H, q=g, A=np.vstack((Aeq, C)), l=np.concatenate((b, d)),
                u=np.concatenate((b, np.full(n_ineq, np.inf))), n=n, n_eq=n_eq)


def reference_solution(d, eps=1e-10):
    """Clarabel interior point at 1e-10 on [A; -A] x <= [u; -l] (infinite
    rows dropped)."""
    import clarabel
    import scipy.sparse as sp
    A = sp.csc_matrix(d["A"])
    ru = np.isfinite(d["u"]) & (np.abs(d["u"]) < 1e20)
    rl = np.isfinite(d["l"]) & (np.abs(d["l"]) < 1e20)
    G = sp.vstack([A[ru], -A[rl]], format="csc")
    h = np.hstack([d["u"][ru], -d["l"][rl]])
    st = clarabel.DefaultSettings()
    st.verbose = False
    st.tol_gap_abs = st.tol_gap_rel = st.tol_feas = eps
    sol = clarabel.DefaultSolver(sp.csc_matrix(d["P"]), d["q"], G, h,
                                 [clarabel.NonnegativeConeT(G.shape[0])], st).solve()
    return np.array(sol.x), str(sol.status)


def references(n, refdir):
    """(solutions, seconds, source): own cache, else the shared cache of
    random_qp_cpu_gpu.py (same problems, same reference), else computed."""
    path = os.path.join(refdir, f"ref_n{n}.npz")
    shared = os.path.join(SHARED_REFS, f"ref_n{n}.npz")
    src = "cached"
    if not os.path.exists(path) and os.path.exists(shared):
        shutil.copy2(shared, path)
        src = "copied from random_qp_cpu_gpu"
    if os.path.exists(path):
        z = np.load(path)
        if int(z["n_seeds"]) == N_SEEDS:
            return [z[f"x{s}"] for s in range(N_SEEDS)], 0.0, src
    t0 = time.perf_counter()
    xs, sts = zip(*(reference_solution(rand_qp(n, s)) for s in range(N_SEEDS)))
    np.savez(path, n_seeds=N_SEEDS, statuses=np.array(sts),
             **{f"x{s}": x for s, x in enumerate(xs)})
    dt = time.perf_counter() - t0
    return list(xs), dt, f"computed {dt:.1f} s"


# ---------------------------------------------------------------------------
# FAME-QP, float32
# ---------------------------------------------------------------------------
def make_fame():
    """Setup (bank, setup test, penalty probe) on the held-out instance, once
    per (n, eps); timed instances loaded via update_matrices/update, keeping
    the probed rho, the scaling and the captured graph."""
    import torch
    from fame_qp import QP, Solver
    cache = {}

    def setup(d, eps):
        n = d["n"]
        if cache.get("key") != (n, eps):
            cache.clear()
            torch.cuda.empty_cache()
            h = rand_qp(n, HELD_OUT)
            qp = QP.create(h["P"], h["q"], h["A"], h["l"], h["u"], device="cuda")
            s = Solver(qp, precision=PRECISION, eps_abs=eps, eps_rel=eps, max_iter=20000,
                       check_interval=CHECK, probe_rho=True)
            cache.update(key=(n, eps), s=s, setup=dict(
                setup_s=float(s.setup_time), setup_seed=HELD_OUT,
                probed_rho=float(s.eng.rho_value), gamma=float(s.gamma),
                mode=str(s.mode), structure=str(s.structure),
                precision=str(s.dtype).replace("torch.", "")))
        s = cache["s"]
        s.update_matrices(P=d["P"], G=d["A"])
        s.update(q=d["q"], l=d["l"], h=d["u"])
        return s

    def solve(s):
        r = s.solve()
        i = r.info
        return (r.z.double().cpu().numpy(), r.status,
                dict(iters=int(r.iters), rho_end=float(i["rho"]), pri_res=float(i["pri_res"]),
                     dua_res=float(i["dua_res"]), mode=str(i["mode"]),
                     precision=str(i["precision"])))

    def cold(s):
        s.reset()                                   # zero iterate, probed rho

    return setup, solve, cold, (lambda: cache.get("setup", {}))


def n_reps(t):
    """(timed repeats, warm-up solves) from the first solve time [s]."""
    if t > 3:
        return 1, 0
    if t > 0.3:
        return 3, 1
    if t > 0.03:
        return 10, 2
    return 30, 5


def _child(conn, n, refs, target):
    try:
        setup, solve, cold, info = make_fame()
        data = [rand_qp(n, s) for s in range(N_SEEDS)]
        sweep = []
        for eps in EPSS:
            step = dict(eps=eps, passed=True, instances=[])
            for seed, (d, xr) in enumerate(zip(data, refs)):
                st = setup(d, eps)
                cold(st)
                t0 = time.perf_counter()
                x, status, extra = solve(st)
                t1 = time.perf_counter() - t0
                err = float(np.max(np.abs(x - xr))) if np.all(np.isfinite(x)) else float("inf")
                inst = dict(seed=seed, check_ms=t1 * 1e3, err=err, status=status, **extra)
                step["instances"].append(inst)
                if err > target:
                    step["passed"] = False
                    break
                reps, warm = n_reps(t1)
                solves = []
                for k in range(warm + reps):
                    cold(st)
                    a = time.perf_counter()
                    _, s_k, ex_k = solve(st)
                    dt = time.perf_counter() - a
                    if k >= warm:
                        solves.append(dict(ms=dt * 1e3, status=s_k, **ex_k))
                inst.update(warmup=warm, reps=reps, solves=solves,
                            median_ms=float(np.median([s["ms"] for s in solves])))
            step["setup"] = info()
            sweep.append(step)
            if step["passed"]:
                conn.send(dict(reached=True, eps=eps, sweep=sweep))
                return
        conn.send(dict(reached=False, sweep=sweep))
    except Exception as exc:  # noqa: BLE001
        import traceback
        conn.send(dict(reached=False, error=f"{type(exc).__name__}: {exc}"[:300],
                       trace=traceback.format_exc()[-1500:]))


def run_one(n, refs, target, cap):
    ctx = mp.get_context("spawn")
    a, b = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(b, n, refs, target))
    t0 = time.perf_counter()
    p.start()
    out = a.recv() if a.poll(cap) else dict(reached=False, over_cap_s=cap)
    p.join(10)
    if p.is_alive():
        p.kill()
        p.join()
    out["wall_s"] = time.perf_counter() - t0
    return out


def summarise(res):
    if not res.get("reached"):
        return None
    step = res["sweep"][-1]
    med = np.array([i["median_ms"] for i in step["instances"]])
    return dict(eps=step["eps"], mean_ms=float(med.mean()), std_ms=float(med.std(ddof=1)),
                median_ms=float(np.median(med)), min_ms=float(med.min()),
                max_ms=float(med.max()), inst_ms=med.tolist(),
                max_err=max(i["err"] for i in step["instances"]),
                statuses=sorted({i["status"] for i in step["instances"]}),
                iters=[i["iters"] for i in step["instances"]], setup=step.get("setup", {}))


# ---------------------------------------------------------------------------
# bookkeeping
# ---------------------------------------------------------------------------
def gpu_temp():
    try:
        return int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            text=True).split()[0])
    except Exception:
        return None


def machine_info():
    info = dict(platform=platform.platform(), python=sys.version.split()[0],
                numpy=np.__version__, cpu_threads=os.cpu_count())
    try:
        info["cpu_model"] = subprocess.check_output(
            "lscpu | grep 'Model name' | head -1", shell=True, text=True).split(":", 1)[1].strip()
        info["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], text=True).strip()
    except Exception:
        pass
    try:
        from importlib.metadata import version
        info["versions"] = {m: version(m) for m in ("torch", "clarabel", "scipy")}
    except Exception:
        pass
    return info


def save(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1, default=float)
    os.replace(tmp, path)


def write_csvs(payload, proc):
    rows, solves = [], []
    for rec in payload["results"]:
        res = rec["solvers"].get(KEY)
        if res is None:
            continue
        base = dict(solver=KEY, device="gpu", precision=PRECISION, n=rec["n"], m=rec["m"])
        s = res.get("summary")
        if s is None:
            why = ("error" if "error" in res else
                   "over_cap" if "over_cap_s" in res else "target_not_reached")
            rows.append(dict(base, status=why))
            continue
        row = dict(base, status="ok", eps=s["eps"], mean_ms=s["mean_ms"], std_ms=s["std_ms"],
                   median_ms=s["median_ms"], min_ms=s["min_ms"], max_ms=s["max_ms"],
                   max_err=s["max_err"])
        for k, v in enumerate(s["inst_ms"]):
            row[f"inst{k}_ms"] = v
        row.update(iters_mean=float(np.mean(s["iters"])), setup_s=s["setup"].get("setup_s"),
                   probed_rho=s["setup"].get("probed_rho"), mode=s["setup"].get("mode"))
        rows.append(row)
        for inst in res["sweep"][-1]["instances"]:
            for r, sv in enumerate(inst["solves"]):
                solves.append(dict(base, eps=s["eps"], seed=inst["seed"], rep=r, ms=sv["ms"],
                                   iters=sv.get("iters", ""), status=sv["status"],
                                   err=inst["err"], rho_end=sv.get("rho_end", ""),
                                   pri_res=sv.get("pri_res", ""), dua_res=sv.get("dua_res", "")))
    for fname, data in (("summary.csv", rows), ("solves.csv", solves)):
        if data:
            keys = list(dict.fromkeys(k for r in data for k in r))
            with open(os.path.join(proc, fname), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(data)


def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--target", type=float, default=1e-3,
                    help="accuracy target, inf-norm distance to the reference")
    ap.add_argument("--cap", type=float, default=900.0,
                    help="seconds per size (whole tolerance sweep)")
    ap.add_argument("--out", default=os.path.join(EXP, "data", "random_qp_fame_f32"))
    ap.add_argument("--redo", action="store_true", help="rerun sizes already saved")
    args = ap.parse_args()

    raw, proc, refdir = (os.path.join(args.out, s) for s in ("raw", "processed", "refs"))
    for d in (raw, proc, refdir):
        os.makedirs(d, exist_ok=True)
    LOG = os.path.join(raw, "run.log")
    path = os.path.join(raw, "random_qp_fame_f32.json")

    payload = json.load(open(path)) if os.path.exists(path) else dict(results=[])
    payload.update(
        kind="random_qp_fame_f32", solver=KEY, precision=PRECISION, target=args.target,
        epss=EPSS, cap_s=args.cap, check_interval=CHECK, timed_seeds=list(range(N_SEEDS)),
        fame_setup_seed=HELD_OUT,
        problem="reluqp.utils.rand_qp(nx=n, n_eq=n//4, n_ineq=n//4, seed)",
        reference="Clarabel, tol 1e-10", machine=machine_info())
    log(f"\n##### {time.strftime('%Y-%m-%d %H:%M:%S')}  FAME-QP {PRECISION}  "
        f"sizes {args.sizes}  target {args.target:g}")

    for n in args.sizes:
        xs, _, src = references(n, refdir)
        rec = next((r for r in payload["results"] if r["n"] == n), None)
        if rec is None:
            rec = dict(n=n, m=2 * int(n / 4), n_eq=int(n / 4), n_ineq=int(n / 4), solvers={})
            payload["results"].append(rec)
        if KEY in rec["solvers"] and not args.redo:
            s = rec["solvers"][KEY].get("summary")
            log(f"  n={n:5d} (saved) " + (f"{s['mean_ms']:.3f} ms eps={s['eps']:g}"
                                          if s else "no result"))
            continue
        t0 = gpu_temp()
        res = run_one(n, xs, args.target, args.cap)
        res.update(gpu_temp_start_C=t0, gpu_temp_end_C=gpu_temp(),
                   timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
        res["summary"] = summarise(res)
        rec["solvers"][KEY] = res
        s = res["summary"]
        if s:
            st = s["setup"]
            log(f"  n={n:5d}  {s['mean_ms']:9.3f} +- {s['std_ms']:.3f} ms  "
                f"(median {s['median_ms']:.3f})  eps={s['eps']:g}  err {s['max_err']:.1e}  "
                f"it {np.mean(s['iters']):.0f}  setup {st.get('setup_s', 0):.2f} s  "
                f"rho {st.get('probed_rho', 0):.0e}  {st.get('mode')}  "
                f"{st.get('precision')}  [ref {src}; {res['wall_s']:.0f} s]")
        else:
            why = res.get("error") or (f"over cap {args.cap:.0f} s" if "over_cap_s" in res
                                       else "target not reached at any eps")
            log(f"  n={n:5d}  -- {why}")
        payload["results"].sort(key=lambda r: r["n"])
        payload["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save(path, payload)
        write_csvs(payload, proc)
    log(f"\ndone {time.strftime('%Y-%m-%d %H:%M:%S')}; data in {args.out}")


if __name__ == "__main__":
    main()
