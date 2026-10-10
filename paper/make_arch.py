import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

OUT = sys.argv[1]
INK = "#1a1a1a"; GR = "#9a9a96"
BLUE, BLUE_E = "#cfe2f7", "#2a78d6"
ORG, ORG_E = "#fbe0c8", "#d9722a"
GRN, GRN_E = "#cdeedd", "#1b9a68"
GRY = "#eeeeec"; RED = "#c62828"
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(6.6, 4.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 68); ax.axis("off")
FS = 7.2

def box(x0, y0, x1, y1, fc, ec, txt, fs=FS, lw=1.2):
    ax.add_patch(FancyBboxPatch((x0, y0), x1-x0, y1-y0, boxstyle="round,pad=0,rounding_size=1.0", fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text((x0+x1)/2, (y0+y1)/2, txt, ha="center", va="center", fontsize=fs, color=INK, zorder=3, linespacing=1.35)
def line(xs, ys, ls="-", c=INK, lw=1.2, z=1): ax.plot(xs, ys, ls=ls, color=c, lw=lw, zorder=z, solid_capstyle="butt")
def arrow(x0, y0, x1, y1, ls="-", c=INK, lw=1.2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=lw, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=9), zorder=1)
def label(x, y, t, ha="center", fs=FS-0.3, c=INK): ax.text(x, y, t, ha=ha, va="center", fontsize=fs, color=c, zorder=4, linespacing=1.3)
def stack(x, yc, ec, fc, s=2.5, d=3.7):
    for dy in (d, -d):
        ax.add_patch(Rectangle((x-s/2, yc+dy-s/2), s, s, fc=fc, ec=ec, lw=1.0, zorder=2))
    ax.text(x, yc, "⋮", ha="center", va="center", fontsize=9, zorder=4)

# ---- image with 8x8 grid (schematic)
gx, gy, cs = 1.5, 48.0, 1.85
targets = [(0,0,2,3),(0,5,3,3),(4,0,3,2),(4,5,3,3)]
tcell = {(r,c) for r0,c0,h,w in targets for r in range(r0,r0+h) for c in range(c0,c0+w)}
for r in range(8):
    for c in range(8):
        ax.add_patch(Rectangle((gx+c*cs, gy+(7-r)*cs), cs, cs, fc=GRN if (r,c) in tcell else BLUE, ec="white", lw=0.8, zorder=2))
for r0,c0,h,w in targets:
    ax.add_patch(Rectangle((gx+c0*cs, gy+(8-r0-h)*cs), w*cs, h*cs, fc="none", ec=GRN_E, lw=1.6, zorder=3))
label(gx+4*cs, gy+8*cs+2.2, "Input image 32×32\n(8×8 grid of 4×4 patches)", fs=FS-0.3)
ax.add_patch(Rectangle((gx, 41.2), 1.8, 1.8, fc=BLUE, ec=BLUE_E, lw=0.9)); label(gx+2.5, 42.1, "context (visible)", ha="left", fs=6.8)
ax.add_patch(Rectangle((gx, 38.4), 1.8, 1.8, fc=GRN, ec=GRN_E, lw=0.9)); label(gx+2.5, 39.3, "targets (hidden)", ha="left", fs=6.8)

# ---- top panel
trunk, X0, X1 = 20.0, 28.0, 48.0
line([gx+8*cs+0.4, trunk], [gy+4*cs]*2)
line([trunk, trunk], [36, 58]); arrow(trunk, 58, X0, 58); arrow(trunk, 36, X0, 36)
label((trunk+X0)/2, 60.8, "context\nonly", fs=6.8); label((trunk+X0)/2, 38.8, "full\nimage", fs=6.8)

box(X0, 52.5, X1, 63.5, BLUE, BLUE_E, "Context encoder $f_\\theta$\nViT, 6 layers,\nwidth 192")
box(61, 52.5, 77, 63.5, GRN, GRN_E, "Predictor $g_\\phi$\n4 layers,\nwidth 96")
arrow(X1, 58, 61, 58); label(54.5, 61.0, "context\nembeddings", fs=6.6)
box(61, 42.5, 77, 50, GRY, GR, "positional mask\ntokens for\ntarget block $k$", fs=6.8)
arrow(69, 50, 69, 52.5)
arrow(77, 58, 80.2, 58); stack(81.5, 58, BLUE_E, BLUE)
label(81.5, 50.6, "predicted\n$\\hat{s}_i$", fs=6.6)
line([82.8, 92.5], [58, 58]); arrow(92.5, 58, 92.5, 55)
box(86, 39.5, 99.5, 55, "white", INK, "Loss: MSE\nover target\npatches,\naveraged over\nthe 4 blocks", fs=6.6)

box(X0, 30.5, X1, 41.5, ORG, ORG_E, "Target encoder $f_{\\bar\\theta}$\nEMA copy,\nfull image")
arrow(38, 52.5, 38, 41.5, ls=(0, (4, 3)))
label(39.5, 47, "EMA update,\nmomentum\n0.996 → 1.0", ha="left", fs=6.6)
arrow(X1, 36, 51.5, 36)
box(51.5, 33.3, 63, 38.7, GRY, GR, "LayerNorm", fs=6.6)
arrow(63, 36, 68.5, 36); stack(69.8, 36, ORG_E, ORG)
label(69.8, 29.6, "targets $s_i$ (192-d)", fs=6.8)
line([71.1, 77.8], [36, 36])
ax.add_patch(Circle((79.5, 36), 1.7, fc="white", ec=INK, lw=1.2, zorder=3))
line([78.3, 80.7], [34.8, 37.2], z=4); line([78.3, 80.7], [37.2, 34.8], z=4)
label(79.5, 32.2, "stop-gradient", fs=6.8)
line([81.2, 92.5], [36, 36]); arrow(92.5, 36, 92.5, 39.5)

# gradients: only to trainable parts
line([97, 97], [55, 66], ls=(0, (4, 3)), c=RED); line([97, 38], [66, 66], ls=(0, (4, 3)), c=RED)
arrow(69, 66, 69, 63.9, c=RED); arrow(38, 66, 38, 63.9, c=RED)
label(53, 67.2, "gradients (trainable parts only)", fs=7.4, c=RED)

# ---- bottom panel: pixel-target control
ax.add_patch(FancyBboxPatch((0.5, 0.5), 99, 25, boxstyle="round,pad=0,rounding_size=1.2", fc="none", ec=GR, lw=1.0, ls=(0, (5, 3)), zorder=0))
ax.text(2.5, 22.6, "Pixel-target control (no target encoder)", ha="left", va="center", fontsize=8.6, fontweight="bold", color=INK)
box(2, 10.5, 16.5, 18.5, GRY, GR, "same image,\nsame masks", fs=6.8)
line([16.5, trunk], [14.5]*2); line([trunk, trunk], [4.5, 14.5]); arrow(trunk, 14.5, X0, 14.5); arrow(trunk, 4.5, X0, 4.5)
box(X0, 9, X1, 20, BLUE, BLUE_E, "Context encoder\n(unchanged)", fs=7.0)
box(61, 9, 77, 20, GRN, GRN_E, "Predictor\n(output layer\n48 units)", fs=7.0)
arrow(X1, 14.5, 61, 14.5); arrow(77, 14.5, 80.2, 14.5); stack(81.5, 14.5, BLUE_E, BLUE)
label(81.5, 7.4, "predicted\n$\\hat{s}_i$", fs=6.6)
line([82.8, 92.5], [14.5, 14.5]); arrow(92.5, 14.5, 92.5, 10.2)
box(86, 3.0, 99.5, 10.2, "white", INK, "Loss (same)", fs=7.6)
box(X0, 0.8, X1, 8.2, GRY, GR, "per-patch\nstandardised\npixels (48-d)", fs=6.6)
arrow(X1, 4.5, 85.9, 4.5); label(67, 6.3, "targets $s_i$ (48-d)", fs=6.8)

fig.savefig(f"{OUT}/fig_architecture.pdf", bbox_inches="tight", pad_inches=0.05)
fig.savefig(f"{OUT}/fig_architecture.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
