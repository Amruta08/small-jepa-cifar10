"""Paper figures from the real sweep results (runs/all_runs.csv, runs/paired_differences.csv).

Fig 1: accuracy vs pre-training epoch (mean, 95% CI over seeds, individual seeds as dots).
Fig 2: paired differences with 95% CIs (forest plot).
Colours: categorical slots 1-4 of the validated reference palette (see dataviz skill), print on white.
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RUNS = sys.argv[1]
OUT = sys.argv[2]

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
C = {"jepa_multiblock": "#2a78d6", "jepa_block": "#eb6834", "jepa_random": "#1baf7a", "pixel_multiblock": "#eda100"}
MK = {"jepa_multiblock": "o", "jepa_block": "s", "jepa_random": "^", "pixel_multiblock": "D"}
NAME = {"jepa_multiblock": "JEPA, multi-block masking", "jepa_block": "JEPA, single-block masking",
        "jepa_random": "JEPA, random-patch masking", "pixel_multiblock": "Pixel targets, multi-block masking"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
    "axes.unicode_minus": False, "axes.formatter.use_mathtext": True,
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.linewidth": 0.8, "pdf.fonttype": 42,
})


def mean_ci(v):
    v = np.asarray(v, float)
    n = len(v)
    return v.mean(), stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)


# ---------------------------------------------------------------- Figure 1
d = pd.read_csv(f"{RUNS}/all_runs.csv")
d = d[d.dataset == "cifar10"]
base = {c: d[(d.cond == c)] for c in ("untrained", "raw_pixels")}
fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.9), sharex=True)
for ax, (metric, title) in zip(axes, [("linear_1.0", "(a) Linear probe, all labels"), ("knn", "(b) $k$-NN classifier")]):
    ref = {c: 100 * base[c][metric].mean() for c in base}
    for c, ls, lab, dy in (("raw_pixels", (0, (5, 3)), "raw pixels", 0.9), ("untrained", (0, (1, 2)), "untrained encoder", 0.9)):
        ax.axhline(ref[c], color=INK2, lw=1.0, ls=ls, zorder=1)
        ax.text(50.8, ref[c] + dy, lab, ha="right", va="bottom", fontsize=7.5, color=INK2)
    for c in C:
        xs, ms, cis = [], [], []
        for ep in (0, 25, 50):
            sel = d[(d.cond == c) & (d.epoch == ep)].sort_values("seed")
            v = 100 * sel[metric].values
            m, ci = mean_ci(v)
            xs.append(ep); ms.append(m); cis.append(ci)
            jit = {"jepa_multiblock": -0.9, "jepa_block": -0.3, "jepa_random": 0.3, "pixel_multiblock": 0.9}[c]
            ax.scatter(np.full(len(v), ep + jit), v, s=7, color=C[c], alpha=0.45, lw=0, zorder=2)
        jit = {"jepa_multiblock": -0.9, "jepa_block": -0.3, "jepa_random": 0.3, "pixel_multiblock": 0.9}[c]
        xj = np.array(xs) + jit
        ax.errorbar(xj, ms, yerr=cis, color=C[c], lw=1.6, capsize=2.2, elinewidth=1.0, zorder=3)
        ax.plot(xj, ms, color=C[c], lw=1.6, marker=MK[c], ms=5.2, mfc=C[c], mec="white", mew=0.9, zorder=4)
    ax.set_title(title, loc="left", fontsize=9)
    ax.set_xlabel("Pre-training epoch")
    ax.set_xticks([0, 25, 50]); ax.set_xlim(-3, 52)
    ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Test accuracy (%)")
axes[0].set_ylim(25, 66); axes[1].set_ylim(25, 66)
handles = [Line2D([0], [0], color=C[c], lw=1.6, marker=MK[c], ms=5.2, mec="white", mew=0.9, label=NAME[c]) for c in C]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01), columnspacing=1.6)
fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.6)
fig.savefig(f"{OUT}/fig_accuracy_vs_epoch.pdf"); fig.savefig(f"{OUT}/fig_accuracy_vs_epoch.png", dpi=220)
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
p = pd.read_csv(f"{RUNS}/paired_differences.csv")
LAB = {"jepa_multiblock": "JEPA multi-block", "jepa_block": "JEPA single-block", "jepa_random": "JEPA random-patch",
       "pixel_multiblock": "pixel-target multi-block", "untrained": "untrained"}
groups = [
    ("Training effect (trained minus untrained encoder)",
     [("jepa_multiblock", "untrained"), ("jepa_block", "untrained"), ("jepa_random", "untrained"), ("pixel_multiblock", "untrained")]),
    ("Recipe comparisons",
     [("jepa_multiblock", "pixel_multiblock"), ("jepa_multiblock", "jepa_block"), ("jepa_multiblock", "jepa_random"), ("jepa_block", "jepa_random")]),
]
metrics = [("linear_1.0", "Linear, all labels"), ("linear_0.1", "Linear, 10% labels"), ("knn", "$k$-NN")]
fig, axes = plt.subplots(2, 3, figsize=(6.2, 4.7), gridspec_kw=dict(height_ratios=[1, 1]))
BLUE = C["jepa_multiblock"]
for r, (gtitle, pairs) in enumerate(groups):
    lo = hi = None
    for cidx, (met, mtitle) in enumerate(metrics):
        ax = axes[r, cidx]
        for i, (a, b) in enumerate(pairs):
            row = p[(p.a == a) & (p.b == b) & (p.metric == met)].iloc[0]
            y = len(pairs) - 1 - i
            ax.plot([row.mean_diff_pp - row.ci95_pp, row.mean_diff_pp + row.ci95_pp], [y, y], color=BLUE, lw=1.6, solid_capstyle="butt", zorder=2)
            ax.plot(row.mean_diff_pp, y, "o", ms=5.5, mec=BLUE, mew=1.3, mfc=BLUE if row.excludes_zero else "white", zorder=3)
        ax.axvline(0, color=INK2, lw=0.9, zorder=1)
        ax.set_ylim(-0.6, len(pairs) - 0.4)
        ax.grid(axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if r == 0:
            ax.set_title(mtitle, fontsize=8.5)
            ax.set_xlim(-2, 36)
        else:
            ax.set_xlim(-12, 17)
        if cidx == 0:
            ax.set_yticks(range(len(pairs)))
            ax.set_yticklabels([f"{LAB[a]}\n$-$ {LAB[b]}" for a, b in reversed(pairs)], fontsize=8.2)
        else:
            ax.set_yticks(range(len(pairs))); ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        if r == 1:
            ax.set_xlabel("Difference (pp)")
handles = [Line2D([0], [0], marker="o", ls="", ms=5.5, mfc=BLUE, mec=BLUE, label="95% CI excludes 0"),
           Line2D([0], [0], marker="o", ls="", ms=5.5, mfc="white", mec=BLUE, mew=1.3, label="95% CI includes 0")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0.0, 0.05, 1, 0.93), h_pad=3.4, w_pad=0.8)
for r, (gtitle, _) in enumerate(groups):
    pos = axes[r, 0].get_position()
    fig.text(0.015, pos.y1 + (0.075 if r == 0 else 0.032), gtitle, fontsize=8.5, fontweight="bold", ha="left", va="bottom")
fig.savefig(f"{OUT}/fig_paired_differences.pdf"); fig.savefig(f"{OUT}/fig_paired_differences.png", dpi=220)
plt.close(fig)
print("figures written to", OUT)
