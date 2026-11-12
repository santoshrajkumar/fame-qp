#!/usr/bin/env python
"""Table I from the random dense QP data: one figure (CPU and GPU solvers
together, median over the five instances with a min-max band) and a LaTeX
table of mean +- std solve time across the five instances (seeds).

Reads the output of random_qp_cpu_gpu.py (all solvers, FAME-QP in
float64) and of random_qp_fame_f32.py (FAME-QP in float32):

    data/random_qp_cpu_gpu/raw/random_qp_cpu_gpu.json
    data/random_qp_fame_f32/raw/random_qp_fame_f32.json     (--extra)

Each instance's time is the median of its repeated cold solves, at the
tolerance reported for that solver and size.  Writes, next to the data:

    figures/fig_random_qp.{pdf,png}       the figure
    tables/tab_random_qp.tex              the table (booktabs; table*)
    processed/random_qp_mean_std.csv      the numbers in both

    python experiments/random_qp/viz_tab_random_qp.py [--data data/random_qp_cpu_gpu]
                                        [--extra data/random_qp_fame_f32 ...]

The older per-instance file of data/random_qp_table1 (written by
bench/run_random_qp.py) is read too, with --data data/random_qp_table1.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # experiments/
ROOT = os.path.dirname(EXP)                                      # repository root
DEFAULT_DATA = os.path.join(EXP, "data", "random_qp_cpu_gpu")
DEFAULT_EXTRA = [os.path.join(EXP, "data", "random_qp_fame_f32")]

# Colour = algorithm (validated 8-slot categorical palette, dataviz skill);
# a GPU port shares its CPU solver's hue (cuOSQP ~ OSQP, CuPIQP ~ PIQP).
# Line style + marker fill = device: GPU solid / filled, CPU dashed / hollow.
# FAME-QP float32: a darker step of FAME-QP's blue (#104281, blue 650 --
# below the lightness band meant for fills, but the step that clears the
# normal-vision separation from #2a78d6) with its own marker.
# (key, label, device, colour, marker)
SERIES = [
    ("fame-qp", "FAME-QP", "gpu", "#2a78d6", "o"),
    ("fame-qp-f32", "FAME-QP (f32)", "gpu", "#104281", "h"),
    ("reluqp", "ReLU-QP", "gpu", "#e87ba4", "X"),
    ("cuosqp", "cuOSQP", "gpu", "#eb6834", "s"),
    ("cupiqp", "CuPIQP", "gpu", "#1baf7a", "^"),
    ("qpax", "qpax", "gpu", "#eda100", ">"),
    ("osqp", "OSQP", "cpu", "#eb6834", "s"),
    ("piqp", "PIQP", "cpu", "#1baf7a", "^"),
    ("proxqp", "ProxQP", "cpu", "#008300", "D"),
    ("clarabel", "Clarabel", "cpu", "#6250d6", "v"),
    ("daqp", "DAQP", "cpu", "#e34948", "P"),
]
TEX_NAME = {"fame-qp": r"\textbf{FAME-QP} (f64)", "fame-qp-f32": r"\textbf{FAME-QP} (f32)",
            "reluqp": "ReLU-QP", "cuosqp": r"cuOSQP$^\dagger$",
            "cupiqp": "CuPIQP", "qpax": "qpax", "osqp": "OSQP", "piqp": "PIQP",
            "proxqp": "ProxQP", "clarabel": "Clarabel", "daqp": "DAQP"}
OLD_KEYS = {"fame-qp-f64": "fame-qp", "reluqp-f64": "reluqp", "qpax-f64": "qpax"}
PLOT_EXCLUDE = {"fame-qp-f32"}      # in the table (and CSV) only, not in the figure
INK, MUTED, GRID = "#2b2b29", "#5a5a55", "#e2e2dd"


# ---------------------------------------------------------------------------
def find_json(data):
    for pat in ("raw/random_qp_cpu_gpu.json", "raw/random_qp_fame_f32.json",
                "raw/random_qp_*perinst.json"):
        hits = sorted(glob.glob(os.path.join(data, pat)))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no random-QP JSON under {data}/raw")


def collect(d):
    """{solver: [dict(n, eps, inst=[5 per-instance medians])]}, sorted by n."""
    out = {}
    for rec in d["results"]:
        if "solvers" in rec:                                  # random_qp_cpu_gpu.py
            items = ((k, (r.get("summary") or {})) for k, r in rec["solvers"].items())
        else:                                                 # bench/run_random_qp.py
            items = ((OLD_KEYS.get(k, k), r) for k, r in rec["results"].items())
        for k, s in items:
            if s.get("inst_ms"):
                out.setdefault(k, []).append(dict(n=rec["n"], eps=s["eps"],
                                                  inst=np.asarray(s["inst_ms"], float)))
    for k in out:
        out[k].sort(key=lambda p: p["n"])
    return out


def write_csv(data, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        k_max = max(len(p["inst"]) for v in data.values() for p in v)
        w.writerow(["solver", "device", "n", "eps", "mean_ms", "std_ms", "median_ms",
                     "min_ms", "max_ms"] + [f"seed{i}_ms" for i in range(k_max)])
        for key, _, dev, _, _ in SERIES:
            for p in data.get(key, []):
                v = p["inst"]
                w.writerow([key, dev, p["n"], p["eps"], f"{v.mean():.6g}",
                            f"{v.std(ddof=1):.6g}", f"{np.median(v):.6g}",
                            f"{v.min():.6g}", f"{v.max():.6g}"] + [f"{x:.6g}" for x in v])
    print("wrote", path)


# ---------------------------------------------------------------------------
def plot(data, out):
    # publication style for one column of a two-column paper (IEEE: 3.5 in):
    # Times-like STIX text and math, TrueType fonts embedded in the PDF
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
    for i, (key, label, dev, color, marker) in enumerate(SERIES):
        pts = data.get(key)
        if not pts or key in PLOT_EXCLUDE:
            continue
        n = [p["n"] for p in pts]
        med = [np.median(p["inst"]) for p in pts]
        lo = [p["inst"].min() for p in pts]
        hi = [p["inst"].max() for p in pts]
        fame = key.startswith("fame-qp")
        gpu = dev == "gpu"
        ax.fill_between(n, lo, hi, color=color, alpha=0.22 if fame else 0.13, lw=0,
                        zorder=5 if fame else 2)
        (h,) = ax.plot(n, med, color=color, ls="-" if gpu else (0, (3, 1.5)),
                       lw=1.7 if fame else 0.9, marker=marker, ms=3.6 if fame else 2.8,
                       mfc=color if gpu else "white", mec=color, mew=0.7,
                       zorder=10 if fame else 4 + i * 0.1, label=label)
        handles[key] = h
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of decision variables $n$")
    ax.set_ylabel("median solve time (ms)")
    ax.grid(True, which="major", lw=0.4, color=GRID)
    ax.set_axisbelow(True)
    sizes = sorted({p["n"] for v in data.values() for p in v})
    ticks = [s for s in (20, 50, 100, 200, 400, 1000, 2000) if sizes[0] <= s <= sizes[-1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_on()
    ax.tick_params(axis="x", which="minor", labelbottom=False)

    # legend above the axes, 4 columns x 3 rows, filled column by column:
    # the GPU / CPU line-style key first, then the GPU solvers, then the CPU ones
    lead = [Line2D([], [], color=INK, ls="-", lw=0.9, marker="o", ms=2.8, mfc=INK,
                   label="GPU"),
            Line2D([], [], color=INK, ls=(0, (3, 1.5)), lw=0.9, marker="o", ms=2.8,
                   mfc="white", mec=INK, mew=0.7, label="CPU")]
    entries = lead + [handles[k] for k, _, d, _, _ in SERIES if d == "gpu" and k in handles] \
        + [handles[k] for k, _, d, _, _ in SERIES if d == "cpu" and k in handles]
    ax.legend(handles=entries, frameon=False, ncol=4, loc="lower left",
              bbox_to_anchor=(-0.02, 1.0), borderaxespad=0.15, handlelength=1.7,
              handletextpad=0.3, columnspacing=0.5, labelspacing=0.2)
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


def table(data, path):
    gpu = [k for k, _, d, _, _ in SERIES if d == "gpu" and k in data]
    cpu = [k for k, _, d, _, _ in SERIES if d == "cpu" and k in data]
    cols = gpu + cpu
    sizes = sorted({p["n"] for k in cols for p in data[k]})
    look = {k: {p["n"]: p for p in data[k]} for k in cols}
    k_inst = max(len(p["inst"]) for k in cols for p in data[k])
    L = [r"% generated by experiments/random_qp/viz_tab_random_qp.py -- do not edit by hand",
         r"\begin{table*}[t]",
         r"\centering",
         r"\caption{Solve time [ms] on the random dense QPs, mean $\pm$ standard deviation over "
         rf"{k_inst} instances (seeds) per size; $m = 2\lfloor n/4 \rfloor$ dense rows, half of "
         r"them equalities. Each instance's time is the median of repeated cold solves at the "
         r"loosest tolerance $\epsilon \in \{10^{-3},\dots,10^{-8}\}$ whose solution is within "
         r"$10^{-3}$ ($\infty$-norm) of a Clarabel $10^{-10}$ reference on every instance; setup "
         r"excluded. FAME-QP is set up (including the penalty probe) on a held-out instance. "
         r"Fastest per row in bold. FAME-QP in float64 (f64) and float32 (f32); "
         r"$^\dagger$float32 (official build); all others float64.}",
         r"\label{tab:random_qp}",
         r"\scriptsize",
         r"\setlength{\tabcolsep}{3pt}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{r" + "r" * len(gpu) + "|" + "r" * len(cpu) + "}",
         r"\toprule",
         rf" & \multicolumn{{{len(gpu)}}}{{c|}}{{GPU}} & \multicolumn{{{len(cpu)}}}{{c}}{{CPU}} \\",
         rf"\cmidrule(lr){{2-{1 + len(gpu)}}}\cmidrule(lr){{{2 + len(gpu)}-{1 + len(cols)}}}",
         "$n$ & " + " & ".join(TEX_NAME.get(k, k) for k in cols) + r" \\",
         r"\midrule"]
    for n in sizes:
        cells, means = [], {k: look[k][n]["inst"].mean() for k in cols if n in look[k]}
        best = min(means, key=means.get) if means else None
        for k in cols:
            p = look[k].get(n)
            if p is None:
                cells.append("--")
                continue
            m, s = fmt(p["inst"].mean(), p["inst"].std(ddof=1))
            cells.append(rf"{{\boldmath${m} \pm {s}$}}" if k == best else rf"${m} \pm {s}$")
        L.append(f"{n} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    with open(path, "w") as f:
        f.write("\n".join(L))
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="dataset folder holding raw/<json>; outputs go here "
                         "(default data/random_qp_cpu_gpu)")
    ap.add_argument("--extra", nargs="*", default=DEFAULT_EXTRA,
                    help="further dataset folders whose series are added "
                         "(default data/random_qp_fame_f32; none: --extra)")
    a = ap.parse_args()
    data_dir = os.path.abspath(a.data)
    figs, tabs, proc = (os.path.join(data_dir, s) for s in ("figures", "tables", "processed"))
    for d in (figs, tabs, proc):
        os.makedirs(d, exist_ok=True)
    src = find_json(data_dir)
    print("reading", src)
    data = collect(json.load(open(src)))
    for extra in a.extra:
        if not os.path.isdir(extra):
            print("skipping missing", extra)
            continue
        src = find_json(os.path.abspath(extra))
        print("reading", src)
        for k, v in collect(json.load(open(src))).items():
            data.setdefault(k, v)
    write_csv(data, os.path.join(proc, "random_qp_mean_std.csv"))
    plot(data, os.path.join(figs, "fig_random_qp"))
    table(data, os.path.join(tabs, "tab_random_qp.tex"))


if __name__ == "__main__":
    main()
