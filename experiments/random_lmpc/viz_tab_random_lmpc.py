#!/usr/bin/env python
"""Closed-loop random linear MPC (Table II problems): one figure of the time
per MPC step against problem size (median over the closed-loop steps with a
band), a LaTeX table of mean +- std time per step, and a companion table of
the control quality behind it: closed-loop cost relative to FAME-QP and the
largest input-bound violation of the returned input, per method and size.

Reads the output of random_lmpc_gpu.py:

    data/random_lmpc_gpu/raw/random_lmpc_gpu_<tag>.json      (--tag, default warm)

Per method and size the statistics are over the closed-loop steps (50 by
default) after the first --skip, each step's time being the wall clock of one
solve with the input copied to the host.  --skip defaults to 0 for runs whose
step 0 starts from an untimed solve at x0 (step0 = init, tag warm_init) and
to 1 otherwise (step 0, a cold start, left out of the figure, table and csv).  Writes, next to the data:

    figures/fig_random_lmpc_<tag>.{pdf,png}      the figure
    tables/tab_random_lmpc_<tag>.tex             the timing table (booktabs; table*)
    tables/tab_random_lmpc_<tag>_quality.tex     cost and bound violation (table*)
    processed/random_lmpc_mean_std_<tag>.csv     the numbers in all of them

J, the bound violation and the share of steps with a bound active are taken
over the same steps as the times (after --skip): J = sum of the stage costs
x_{t+1}'Q x_{t+1} + u_t'R u_t, violation = max_t max_i (|u_{0,i}| - 1)_+ of
the input each method returned (before the plant clips it).

    python experiments/random_lmpc/viz_tab_random_lmpc.py [--tag warm_init|warm|cold|cold_fame-x0]
                                          [--band minmax|p5p95|iqr] [--skip N]

Sizes or methods missing from the data (an unfinished run) are left out of
the figure and shown as -- in the table; a failed method shows as fail.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
import numpy as np  # noqa: E402

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # experiments/
ROOT = os.path.dirname(EXP)                                      # repository root
DEFAULT_DATA = os.path.join(EXP, "data", "random_lmpc_gpu")

# Colours follow the random-QP figure (viz_tab_random_qp.py) for the
# solvers both share; the two MPC methods take the remaining validated slots
# (violet, red).  Line style + marker fill = method kind: QP solver (on the
# condensed QP) solid / filled, MPC method (on the OCP) dashed / hollow.
# (key, label, kind, colour, marker)
SERIES = [
    ("fame-qp", "FAME-QP", "qp", "#2a78d6", "o"),
    ("reluqp", "ReLU-QP", "qp", "#e87ba4", "X"),
    ("cuosqp", "cuOSQP", "qp", "#eb6834", "s"),
    ("cupiqp", "CuPIQP", "qp", "#1baf7a", "^"),
    ("qpax", "qpax", "qp", "#eda100", ">"),
    ("cudampc", "CUDA MPC", "mpc", "#6250d6", "v"),
    ("pimpc", r"$\pi$MPC", "mpc", "#e34948", "P"),
]
TEX_NAME = {"fame-qp": r"\textbf{FAME-QP}", "reluqp": "ReLU-QP", "cuosqp": r"cuOSQP$^\dagger$",
            "cupiqp": r"CuPIQP$^\ast$", "qpax": r"qpax$^\ast$",
            "cudampc": r"CUDA MPC$^\ddagger$", "pimpc": r"$\pi$MPC"}
BANDS = {"minmax": (0, 100, "min--max"), "p5p95": (5, 95, "5th--95th percentile"),
         "iqr": (25, 75, "interquartile range")}
INK, GRID = "#2b2b29", "#e2e2dd"


# ---------------------------------------------------------------------------
def collect(d, skip=1):
    """{method: [dict(nu, nx, n, ms=per-step times)]} and {method: {nu: error}}.
    Steps k < skip are dropped (skip = 1: the cold-start step 0)."""
    out, failed = {}, {}
    for rec in d["results"]:
        for k, r in rec.get("methods", {}).items():
            if r.get("steps") and "median_ms" in r:
                st = [s for s in r["steps"] if s["k"] >= skip]
                it = [s["iters"] for s in st if s.get("iters") is not None]
                has = all(x in st[0] for x in ("stage_cost", "viol", "bound_active")) if st else False
                out.setdefault(k, []).append(dict(
                    nu=rec["nu"], nx=rec["nx"], n=rec["n"],
                    ms=np.array([s["ms"] for s in st], float),
                    J=float(sum(s["stage_cost"] for s in st)) if has else r.get("J"),
                    viol=float(max(s["viol"] for s in st)) if has else r.get("viol"),
                    active=float(np.mean([s["bound_active"] for s in st])) if has
                    else r.get("active_fraction"),
                    iters=float(np.mean(it)) if it else r.get("iters_mean")))
            else:
                failed.setdefault(k, {})[rec["nu"]] = r.get("error") or "over cap"
    for k in out:
        out[k].sort(key=lambda p: p["n"])
    return out, failed


def write_csv(data, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "kind", "nu", "nx", "n", "steps", "mean_ms", "std_ms",
                    "median_ms", "p5_ms", "p25_ms", "p75_ms", "p95_ms", "p99_ms",
                    "min_ms", "max_ms", "J", "max_viol", "active_fraction", "iters_mean"])
        for key, _, kind, _, _ in SERIES:
            for p in data.get(key, []):
                v = p["ms"]
                q = np.percentile(v, [5, 25, 75, 95, 99])
                w.writerow([key, kind, p["nu"], p["nx"], p["n"], len(v), f"{v.mean():.6g}",
                            f"{v.std(ddof=1):.6g}", f"{np.median(v):.6g}"]
                           + [f"{x:.6g}" for x in q]
                           + [f"{v.min():.6g}", f"{v.max():.6g}"]
                           + ["" if p[k] is None else f"{p[k]:.6g}"
                              for k in ("J", "viol", "active", "iters")])
    print("wrote", path)


# ---------------------------------------------------------------------------
def plot(data, out, band):
    lo_q, hi_q, _ = BANDS[band]
    # one column of a two-column paper (IEEE 3.5 in); Times-like STIX text and
    # math, TrueType fonts embedded -- the same style as fig_random_qp
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 9, "axes.labelsize": 10, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "xtick.direction": "out",
        "ytick.direction": "out", "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.45, "ytick.minor.width": 0.45,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2})
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    handles = {}
    for i, (key, label, kind, color, marker) in enumerate(SERIES):
        pts = data.get(key)
        if not pts:
            continue
        n = [p["n"] for p in pts]
        med = [np.median(p["ms"]) for p in pts]
        lo = [np.percentile(p["ms"], lo_q) for p in pts]
        hi = [np.percentile(p["ms"], hi_q) for p in pts]
        fame = key == "fame-qp"
        qp = kind == "qp"
        ax.fill_between(n, lo, hi, color=color, alpha=0.22 if fame else 0.13, lw=0,
                        zorder=5 if fame else 2)
        (h,) = ax.plot(n, med, color=color, ls="-" if qp else (0, (3, 1.5)),
                       lw=1.7 if fame else 0.9, marker=marker, ms=3.6 if fame else 2.8,
                       mfc=color if qp else "white", mec=color, mew=0.7,
                       zorder=10 if fame else 4 + i * 0.1, label=label)
        handles[key] = h
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of decision variables $n$")
    ax.set_ylabel("median solve time per step (ms)")
    ax.grid(True, which="major", lw=0.4, color=GRID)
    ax.set_axisbelow(True)
    sizes = sorted({p["n"] for v in data.values() for p in v})
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.xaxis.set_minor_formatter(NullFormatter())
    if len(sizes) > 1:
        ax.set_xlim(sizes[0] / 1.12, sizes[-1] * 1.12)

    # legend above the axes, 4 columns, filled column by column: the
    # QP-solver / MPC-method key first, then the QP solvers, then the MPC methods
    lead = [Line2D([], [], color=INK, ls="-", lw=0.9, marker="o", ms=2.8, mfc=INK,
                   label="QP solver"),
            Line2D([], [], color=INK, ls=(0, (3, 1.5)), lw=0.9, marker="o", ms=2.8,
                   mfc="white", mec=INK, mew=0.7, label="MPC method")]
    entries = lead + [handles[k] for k, _, kd, _, _ in SERIES if kd == "qp" and k in handles] \
        + [handles[k] for k, _, kd, _, _ in SERIES if kd == "mpc" and k in handles]
    ax.legend(handles=entries, frameon=False, ncol=3, loc="lower left",
              bbox_to_anchor=(-0.02, 1.0), borderaxespad=0.15, handlelength=1.7,
              handletextpad=0.3, columnspacing=0.6, labelspacing=0.2)
    fig.tight_layout(pad=0.1)
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out + ".png", dpi=400, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print("wrote", out + ".pdf")


# ---------------------------------------------------------------------------
def fmt(mean, std):
    """3 significant digits on the mean; std to the same decimal place."""
    dec = 0 if mean >= 100 else 1 if mean >= 10 else 2 if mean >= 1 else 3
    return f"{mean:.{dec}f}", f"{std:.{dec}f}"


def table(d, data, failed, path, tag, skip=1):
    cols = [k for k, _, _, _, _ in SERIES if k in data or k in failed]
    qp = [k for k in cols if dict((s[0], s[2]) for s in SERIES)[k] == "qp"]
    mpc = [k for k in cols if k not in qp]
    cols = qp + mpc
    sizes = sorted({(r["nu"], r["nx"], r["n"]) for r in d["results"]})
    look = {k: {p["nu"]: p for p in data.get(k, [])} for k in cols}
    steps = d.get("steps", 50)
    warm = d.get("start", "warm") == "warm"
    held = d.get("fame_setup", "heldout") == "heldout"
    steps_txt = (f"steps {skip}--{steps - 1} of {steps}" + (r" (step 0, the cold start, "
                 r"excluded)" if skip == 1 else "") if skip else f"{steps} steps")
    init = d.get("step0") == "init"
    step0_txt = (r"Before step 0 every method solves the problem at $x_0$ once (untimed) "
                 r"and step 0 starts from that solution; " if init else
                 r"Step 0 starts cold; ")
    start_txt = (step0_txt + r"from step 1 on every method starts from its "
                 r"previous solution shifted one stage (last stage repeated). "
                 r"$^\ast$Interior point, no warm start (every step cold). "
                 if warm else r"Every step solved from a cold start. ")
    ci = d.get("check_interval", {})
    names = {"fame-qp": "FAME-QP", "reluqp": "ReLU-QP", "pimpc": r"$\pi$MPC"}
    check_txt = ""
    if isinstance(ci, dict):
        multi = [k for k in ci if k in cols and ci[k] > 1]
        every = [k for k in ci if k in cols and ci[k] <= 1]
        parts = []
        if multi:
            parts.append("convergence checked every " + ", ".join(
                f"{ci[k]} ({names.get(k, k)})" for k in multi) + " iterations")
        if every:
            parts.append(" and ".join(names.get(k, k) for k in every)
                         + " every iteration (native)")
        check_txt = ", ".join(parts) + "; " if parts else ""
    setup_txt = (r"FAME-QP is set up (including the penalty probe) on a held-out random "
                 r"system of the same size. " if held else
                 r"FAME-QP is set up (including the penalty probe) at the initial state. ")
    L = [r"% generated by experiments/random_lmpc/viz_tab_random_lmpc.py -- do not edit by hand",
         r"\begin{table*}[t]",
         r"\centering",
         rf"\caption{{Closed-loop MPC on random linear systems (problems of ReLU-QP), "
         rf"{steps_txt}: solve time per MPC step [ms], mean $\pm$ standard deviation over "
         rf"these steps. Tolerance-matched: each step is solved to the method's own "
         rf"stopping test at $10^{{-3}}$ ($\pi$MPC $10^{{-6}}$ on its own residual), "
         + check_txt + r"no iteration budget. " + start_txt + setup_txt
         + r"Fastest per row in bold. $^\dagger$float32 (official build); all others "
           rf"float64. $^\ddagger$No stopping test: {d.get('cudampc_iters', 2000)} "
           r"iterations per step.}",
         rf"\label{{tab:random_lmpc_{tag.replace('-', '_')}}}",
         r"\scriptsize",
         r"\setlength{\tabcolsep}{3pt}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{lr" + "r" * len(qp) + ("|" + "r" * len(mpc) if mpc else "") + "}",
         r"\toprule",
         rf" & & \multicolumn{{{len(qp)}}}{{c{'|' if mpc else ''}}}{{QP solvers (condensed QP)}}"
         + (rf" & \multicolumn{{{len(mpc)}}}{{c}}{{MPC methods}}" if mpc else "") + r" \\",
         rf"\cmidrule(lr){{3-{2 + len(qp)}}}"
         + (rf"\cmidrule(lr){{{3 + len(qp)}-{2 + len(cols)}}}" if mpc else ""),
         "$n_u/n_x$ & $n$ & " + " & ".join(TEX_NAME.get(k, k) for k in cols) + r" \\",
         r"\midrule"]
    for nu, nx, n in sizes:
        means = {k: look[k][nu]["ms"].mean() for k in cols if nu in look[k]}
        best = min(means, key=means.get) if means else None
        cells = []
        for k in cols:
            p = look[k].get(nu)
            if p is None:
                cells.append("fail" if nu in failed.get(k, {}) else "--")
                continue
            m, s = fmt(p["ms"].mean(), p["ms"].std(ddof=1))
            cells.append(rf"{{\boldmath${m} \pm {s}$}}" if k == best else rf"${m} \pm {s}$")
        L.append(f"{nu}/{nx} & {n} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    with open(path, "w") as f:
        f.write("\n".join(L))
    print("wrote", path)


def _sci(v):
    """Compact scientific notation for a table cell: 2.7e-06 -> 2.7e-6, 0 -> 0."""
    if v is None:
        return "--"
    if v == 0:
        return "0"
    m, e = f"{v:.1e}".split("e")
    return f"{m}e{int(e)}"


def quality_table(d, data, failed, path, tag, skip=1):
    """Per size and method: closed-loop cost J relative to FAME-QP [%] and the
    largest input-bound violation of the returned input; FAME-QP's own J and
    the share of steps with an input bound active as reference columns."""
    cols = [k for k, _, _, _, _ in SERIES if k in data or k in failed]
    qp = [k for k in cols if dict((s[0], s[2]) for s in SERIES)[k] == "qp"]
    mpc = [k for k in cols if k not in qp]
    cols = qp + mpc
    others = [k for k in cols if k != "fame-qp"]
    sizes = sorted({(r["nu"], r["nx"], r["n"]) for r in d["results"]})
    look = {k: {p["nu"]: p for p in data.get(k, [])} for k in cols}
    steps = d.get("steps", 50)
    steps_txt = (f"steps {skip}--{steps - 1} of {steps}" if skip else f"all {steps} steps")
    L = [r"% generated by experiments/random_lmpc/viz_tab_random_lmpc.py -- do not edit by hand",
         r"\begin{table*}[t]",
         r"\centering",
         rf"\caption{{Control quality behind Table~\ref{{tab:random_lmpc_{tag.replace('-', '_')}}}, "
         rf"over {steps_txt}. $J$: closed-loop cost "
         r"$\sum_t x_{t+1}^\top Q x_{t+1}+u_t^\top R u_t$ with FAME-QP's inputs, and "
         r"$\Delta J$ the cost of each other method relative to it [\%]; active: share of "
         r"steps with an input bound active (FAME-QP); viol.: largest bound violation "
         r"$\max_t\max_i(|u_{0,i}|-1)_+$ of the input each method returned, before the "
         r"plant clips it. Same runs and settings as the timing table.}",
         rf"\label{{tab:random_lmpc_{tag.replace('-', '_')}_quality}}",
         r"\scriptsize",
         r"\setlength{\tabcolsep}{3pt}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{lrrr r" + "rr" * len(others) + "}",
         r"\toprule",
         r" & & \multicolumn{3}{c}{" + TEX_NAME.get("fame-qp", "FAME-QP") + "}"
         + "".join(rf" & \multicolumn{{2}}{{c}}{{{TEX_NAME.get(k, k)}}}" for k in others)
         + r" \\",
         r"\cmidrule(lr){3-5}" + "".join(rf"\cmidrule(lr){{{6 + 2 * i}-{7 + 2 * i}}}"
                                        for i in range(len(others))),
         r"$n_u/n_x$ & $n$ & $J$ & active & viol." + r" & $\Delta J$ [\%] & viol." * len(others)
         + r" \\",
         r"\midrule"]
    rows = []
    for nu, nx, n in sizes:
        f = look.get("fame-qp", {}).get(nu)
        cells = []
        if f is None:
            cells += ["--", "--", "--"]
        else:
            cells += [f"{f['J']:.6g}", f"{100 * f['active']:.0f}\\%", _sci(f["viol"])]
        row = dict(nu=nu, nx=nx, n=n, J_fame=None if f is None else f["J"],
                   active_fame=None if f is None else f["active"],
                   viol_fame=None if f is None else f["viol"])
        for k in others:
            p = look[k].get(nu)
            if p is None:
                st = "fail" if nu in failed.get(k, {}) else "--"
                cells += [st, st]
                continue
            dj = None if f is None or not f["J"] else 100.0 * (p["J"] / f["J"] - 1.0)
            djs = "--" if dj is None else ("0.000" if abs(dj) < 5e-4 else f"{dj:+.3f}")
            cells += [djs, _sci(p["viol"])]
            row[f"dJ_pct_{k}"], row[f"viol_{k}"], row[f"J_{k}"] = dj, p["viol"], p["J"]
        rows.append(row)
        L.append(f"{nu}/{nx} & {n} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    with open(path, "w") as f:
        f.write("\n".join(L))
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(path[:-4] + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="dataset folder holding raw/random_lmpc_gpu_<tag>.json")
    ap.add_argument("--tag", default="warm_init",
                    help="run to plot: warm_init (default), warm, cold, cold_fame-x0, ...")
    ap.add_argument("--band", choices=list(BANDS), default="minmax",
                    help="band around the median over the steps (default min-max)")
    ap.add_argument("--skip", type=int, default=None,
                    help="leave out the first SKIP steps (default: 0 if step 0 starts "
                         "from an untimed solve at x0, else 1 -- the cold step 0)")
    a = ap.parse_args()
    data_dir = os.path.abspath(a.data)
    src = os.path.join(data_dir, "raw", f"random_lmpc_gpu_{a.tag}.json")
    figs, tabs, proc = (os.path.join(data_dir, s) for s in ("figures", "tables", "processed"))
    for d in (figs, tabs, proc):
        os.makedirs(d, exist_ok=True)
    print("reading", src)
    d = json.load(open(src))
    if a.skip is None:
        a.skip = 0 if d.get("step0") == "init" else 1
    print("skipping the first", a.skip, "step(s)")
    data, failed = collect(d, a.skip)
    for k, v in failed.items():
        print(f"  {k}: no result at n_u = {sorted(v)}")
    write_csv(data, os.path.join(proc, f"random_lmpc_mean_std_{a.tag}.csv"))
    plot(data, os.path.join(figs, f"fig_random_lmpc_{a.tag}"), a.band)
    table(d, data, failed, os.path.join(tabs, f"tab_random_lmpc_{a.tag}.tex"), a.tag, a.skip)
    quality_table(d, data, failed, os.path.join(tabs, f"tab_random_lmpc_{a.tag}_quality.tex"),
                  a.tag, a.skip)


if __name__ == "__main__":
    main()
