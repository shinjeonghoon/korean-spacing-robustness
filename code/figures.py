"""Paper figures (PDF), computed from the same per-example predictions as the tables.

    python code/figures.py --results results --tokdiff results/tokdiff --out results/figures

fig1_boundary_drop.pdf   Set A (k=1): accuracy drop per boundary type and model, 95% bootstrap CI
fig2_matched_change.pdf  Set B (k=1): JOSA vs EOJEOL drop within quartiles of segmentation change
Colors: categorical slots 1-4 of the validated default palette (blue, orange, aqua, yellow);
bars also carry hatching so they stay distinguishable in grayscale print.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import MODEL_NAME, boot, load_preds, order

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
HATCH = ["", "///", "...", "\\\\\\"]
INK, MUTED, GRID = "#222222", "#6b7280", "#e5e7eb"

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def fig1(results, out, rng):
    preds, _ = load_preds(results, "setA_k1")
    types = ["JOSA", "DEP", "EOJEOL", "RANDOM"]
    models = order(preds)
    fig, ax = plt.subplots(figsize=(3.2, 2.1))
    w = 0.19
    x = np.arange(len(models))
    for j, t in enumerate(types):
        stats = [boot((preds[m]["CLEAN"] - preds[m][t]).to_numpy(), rng) for m in models]
        mean = [100 * s[0] for s in stats]
        err = np.array([[100 * (s[0] - s[1]) for s in stats], [100 * (s[2] - s[0]) for s in stats]])
        ax.bar(x + (j - 1.5) * w, mean, w * 0.9, color=SERIES[j], hatch=HATCH[j], edgecolor="white",
               linewidth=0.6, label=t, zorder=2)
        ax.errorbar(x + (j - 1.5) * w, mean, yerr=err, fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.5, zorder=3)
    ax.axhline(0, color=MUTED, linewidth=0.6)
    ax.set_xticks(x, [MODEL_NAME.get(m, m) for m in models])
    ax.set_ylabel("Accuracy drop (pp)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2), handlelength=1.2, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_boundary_drop.pdf"))
    fig.savefig(os.path.join(out, "fig1_boundary_drop.png"), dpi=200)


def fig2(results, tokdiff, out, rng):
    preds, _ = load_preds(results, "setB_k1")
    td = pd.read_csv(os.path.join(tokdiff, "setB_k1.csv"))
    models = order(preds)
    fig, axes = plt.subplots(1, len(models), figsize=(6.3, 1.9), sharey=True)
    for ax, m in zip(axes, models):
        t = td[td["model"].str.split("/").str[-1] == m]
        parts = []
        for typ in ["JOSA", "EOJEOL"]:
            g = t[t["type"] == typ].set_index("idx")
            parts.append(pd.DataFrame({"type": typ, "e": g["edit_norm"], "drop": (preds[m]["CLEAN"] - preds[m][typ]).loc[g.index]}))
        X = pd.concat(parts)
        X["q"] = pd.qcut(X["e"].rank(method="first"), 4, labels=False)
        for j, typ in enumerate(["JOSA", "EOJEOL"]):
            col = SERIES[0] if typ == "JOSA" else SERIES[1]
            st = [boot(X[(X["q"] == q) & (X["type"] == typ)]["drop"].to_numpy(), rng) for q in range(4)]
            mean = [100 * s[0] for s in st]
            lo = [100 * s[1] for s in st]
            hi = [100 * s[2] for s in st]
            xs = np.arange(4) + (j - 0.5) * 0.08
            ax.fill_between(xs, lo, hi, color=col, alpha=0.15, linewidth=0)
            ax.plot(xs, mean, color=col, linewidth=2, marker="o" if typ == "JOSA" else "s", markersize=4,
                    markeredgecolor="white", markeredgewidth=0.8, label=typ)
            ax.annotate(typ, (xs[-1], mean[-1]), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=7, color=INK)
        ax.axhline(0, color=MUTED, linewidth=0.6)
        ax.set_title(MODEL_NAME.get(m, m), fontsize=8, color=INK)
        ax.set_xticks(range(4), ["Q1", "Q2", "Q3", "Q4"])
        ax.set_xlim(-0.3, 3.8)
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    fig.supxlabel("Segmentation change (quartile of normalized token edit distance)", fontsize=8, color=INK, y=0.02)
    axes[0].set_ylabel("Accuracy drop (pp)")
    axes[0].legend(frameon=False, loc="upper left", handlelength=1.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_matched_change.pdf"))
    fig.savefig(os.path.join(out, "fig2_matched_change.png"), dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--tokdiff", default="results/tokdiff")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig1(a.results, a.out, np.random.default_rng(10))
    fig2(a.results, a.tokdiff, a.out, np.random.default_rng(11))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
