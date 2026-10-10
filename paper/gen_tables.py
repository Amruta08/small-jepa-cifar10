"""Generate LaTeX tables and a numbers summary directly from the real result CSVs (no hand transcription)."""
import sys, numpy as np, pandas as pd
from scipy import stats
RUNS, OUT = sys.argv[1], sys.argv[2]
d = pd.read_csv(f"{RUNS}/all_runs.csv"); d = d[d.dataset == "cifar10"]
p = pd.read_csv(f"{RUNS}/paired_differences.csv")
tq = stats.t.ppf(0.975, 2)
def mci(v):
    v = np.asarray(v, float); return v.mean(), tq * v.std(ddof=1) / np.sqrt(len(v))
def fmt(m, c, scale=100): return f"{m*scale:.1f} $\\pm$ {c*scale:.1f}"
NAME = {"untrained": "Untrained encoder", "raw_pixels": "Raw pixels (no network)", "jepa_multiblock": "JEPA, multi-block",
        "jepa_block": "JEPA, single block", "jepa_random": "JEPA, random patches", "pixel_multiblock": "Pixel targets, multi-block"}
order = ["raw_pixels", "untrained", "jepa_multiblock", "jepa_block", "jepa_random", "pixel_multiblock"]
final = {c: (0 if c in ("untrained", "raw_pixels") else 50) for c in order}
rows = []
summ = {}
for c in order:
    s = d[(d.cond == c) & (d.epoch == final[c])].sort_values("seed")
    assert len(s) == 3, (c, len(s))
    cells = [NAME[c]] + [fmt(*mci(s[m])) for m in ("linear_1.0", "linear_0.1", "linear_0.01", "knn")]
    er = s.erank_centered.mean()
    cells.append("--" if c == "raw_pixels" else f"{er:.1f}")
    rows.append(" & ".join(cells) + r" \\")
    summ[c] = {m: mci(s[m]) for m in ("linear_1.0", "linear_0.1", "linear_0.01", "knn")} | {"erank": er, "loss": s.train_loss.mean() if c not in ("untrained","raw_pixels") else np.nan}
rows.insert(2, r"\midrule")
open(f"{OUT}/tab_main.tex", "w").write("\n".join(rows) + "\n\\botrule\n")

# paired table (8 comparisons x 3 metrics); bold if 95% CI excludes 0
pairs = [("jepa_multiblock","untrained"),("jepa_block","untrained"),("jepa_random","untrained"),("pixel_multiblock","untrained"),
         ("jepa_multiblock","pixel_multiblock"),("jepa_multiblock","jepa_block"),("jepa_multiblock","jepa_random"),("jepa_block","jepa_random")]
SH = {"jepa_multiblock":"JEPA multi-block","jepa_block":"JEPA single block","jepa_random":"JEPA random patches","pixel_multiblock":"pixel-target multi-block","untrained":"untrained"}
rows = []
for a, b in pairs:
    cells = [f"{SH[a]} $-$ {SH[b]}"]
    for m in ("linear_1.0", "linear_0.1", "knn"):
        r = p[(p.a == a) & (p.b == b) & (p.metric == m)].iloc[0]
        t = f"{r.mean_diff_pp:+.1f} $\\pm$ {r.ci95_pp:.1f}".replace("-", "$-$").replace("+", "$+$")
        cells.append(f"\\textbf{{{t}}}" if r.excludes_zero else t)
    rows.append(" & ".join(cells) + r" \\")
rows.insert(4, r"\midrule")
open(f"{OUT}/tab_paired.tex", "w").write("\n".join(rows) + "\n\\botrule\n")

# per-seed table (appendix)
rows = []
for c in order:
    for sd in (0, 1, 2):
        s = d[(d.cond == c) & (d.epoch == final[c]) & (d.seed == sd)].iloc[0]
        loss = "--" if np.isnan(s.train_loss) else f"{s.train_loss:.3f}"
        rows.append(f"{NAME[c]} & {sd} & {100*s['linear_1.0']:.1f} & {100*s['linear_0.1']:.1f} & {100*s['linear_0.01']:.1f} & {100*s['knn']:.1f} & {'--' if c == 'raw_pixels' else format(s.erank_centered, '.1f')} & {loss} \\\\")
    if c != order[-1]: rows.append(r"\midrule")
open(f"{OUT}/tab_perseed.tex", "w").write("\n".join(rows) + "\n\\botrule\n")

# numbers used in the prose
print("== final-epoch summary (mean, ci) ==")
for c in order:
    s = summ[c]; print(f"{c:18s} lin100 {s['linear_1.0'][0]*100:.2f}+-{s['linear_1.0'][1]*100:.2f}  lin10 {s['linear_0.1'][0]*100:.2f}+-{s['linear_0.1'][1]*100:.2f}  lin1 {s['linear_0.01'][0]*100:.2f}+-{s['linear_0.01'][1]*100:.2f}  knn {s['knn'][0]*100:.2f}+-{s['knn'][1]*100:.2f}  erank {s['erank']:.1f} loss {s['loss']:.4f}")
print("== epoch-25 means ==")
for c in order[2:]:
    for ep in (0, 25, 50):
        s = d[(d.cond == c) & (d.epoch == ep)]
        print(c, ep, f"lin100 {100*s['linear_1.0'].mean():.2f} knn {100*s['knn'].mean():.2f}")
print("== per-seed SD linear100 at 50 ==")
for c in order[2:]:
    s = d[(d.cond == c) & (d.epoch == 50)]; print(c, f"sd {100*s['linear_1.0'].std(ddof=1):.2f}", (100*s['linear_1.0']).round(1).tolist())
print("== 1% label paired (computed from per-seed) ==")
def diff(a,b,m,ep=50):
    A=d[(d.cond==a)&(d.epoch==(0 if a=='untrained' else ep))].sort_values('seed')[m].values
    B=d[(d.cond==b)&(d.epoch==(0 if b=='untrained' else ep))].sort_values('seed')[m].values
    return 100*(A-B)
for a,b in [("jepa_multiblock","pixel_multiblock"),("jepa_multiblock","jepa_random"),("jepa_block","jepa_random"),("jepa_multiblock","jepa_block"),("jepa_multiblock","untrained")]:
    x=diff(a,b,'linear_0.01'); print(a,b,'lin1pct diff %.2f +- %.2f'%(x.mean(), tq*x.std(ddof=1)/np.sqrt(3)), x.round(2).tolist())
