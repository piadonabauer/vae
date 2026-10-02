"""Architecture sketch of the three fusion operators (DiT Fig.3 style).

Three vertical block diagrams, drawn bottom-up (inputs at the bottom,
fused bottleneck feature at the top), faithful to
DiffSynth-Studio/diffsynth/models/wan_video_vae.py:

  (a) cross_attention (default): ViewAttention = RMSNorm -> 1x1 QKV -> MHA over
      all views' spatial tokens per t' -> zero-init 1x1 proj -> +residual,
      then a carry-forward binary tree of pairwise merges, each merge =
      concat(2C) -> ResBlock(2C->C) -> ResBlock(C->C), V-1 merges total,
      independent weights per merge (lines 2370-2755).
  (b) conv3d: channel concat (V*C) -> 1x1x1 Conv3d -> GN+SiLU ->
      2x symmetric (non-causal) 3D residual blocks (lines 2419-2426, 2774-2784).
  (c) conv4d factorized: stack view axis -> spatial 3x3 Conv2d -> GN+SiLU ->
      temporal 3x3x3 Conv3d -> GN+SiLU -> view-axis Conv3d (V,3,3) V->1 ->
      GN+SiLU -> 2x 3D residual blocks (lines 2434-2452, 2790-2820).

Output: fusion_operators.pdf (full text width).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = os.path.dirname(os.path.abspath(__file__))
W2 = 6.875

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
})

# palette (soft, print-friendly)
C_IO    = "#ffffff"   # inputs / outputs
C_ATTN  = "#fde3c8"   # attention
C_CONV  = "#cfe2f3"   # convolutions
C_NORM  = "#ececec"   # norm / act
C_RES   = "#d7ebd2"   # residual blocks
C_SHAPE = "#f3e6f7"   # reshape / concat / stack ops
EDGE    = "#555555"

BW = 0.80            # box width (axes units)
FS = 6.4             # box label font size
FS_SMALL = 5.4       # annotations


def box(ax, y, label, color, h=0.052, w=BW, x=0.5, fs=FS, lw=0.7, ls="-"):
    b = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.008,rounding_size=0.012",
                       fc=color, ec=EDGE, lw=lw, linestyle=ls, zorder=3)
    ax.add_patch(b)
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, zorder=4)
    return y


def arrow(ax, y0, y1, x=0.5):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                 mutation_scale=7, lw=0.8, color="#333333",
                                 shrinkA=0, shrinkB=0.5, zorder=2))


def oplus(ax, y, x=0.5, r=0.016):
    c = Circle((x, y), r, fc="white", ec=EDGE, lw=0.8, zorder=4)
    ax.add_patch(c)
    ax.plot([x - r * 0.55, x + r * 0.55], [y, y], color=EDGE, lw=0.8, zorder=5)
    ar = r * 0.55 * 0.62  # visually equal arm length (axes aspect != 1)
    ax.plot([x, x], [y - ar, y + ar], color=EDGE, lw=0.8, zorder=5)
    return y


def skip(ax, y0, y1, x_from=0.5, x_side=0.945):
    """Residual skip: right-side elbow from y0 up to the oplus at y1."""
    ax.plot([x_from, x_side, x_side], [y0, y0, y1], color=EDGE, lw=0.7, zorder=1)
    ax.add_patch(FancyArrowPatch((x_side, y1), (x_from + 0.02, y1),
                                 arrowstyle="-|>", mutation_scale=6, lw=0.7,
                                 color=EDGE, shrinkA=0, shrinkB=0, zorder=1))


def setup(ax, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, -0.035, title, ha="center", va="top", fontsize=7.6,
            transform=ax.transAxes)


def panel_attention(ax):
    setup(ax, "(a) Multi-view attention + tree merge (default)")
    H = 0.042
    ys = iter([0.042, 0.112, 0.182, 0.252, 0.322, 0.392, 0.455])
    y_in   = box(ax, next(ys), "per-view features  $z_1,\\dots,z_V$", C_IO, h=H)
    y_tok  = box(ax, next(ys), "stack views $\\to$ tokens  $[T'\\!,\\ V\\!\\cdot\\!N,\\ C]$", C_SHAPE, h=H)
    y_norm = box(ax, next(ys), "RMSNorm", C_NORM, h=H)
    y_qkv  = box(ax, next(ys), "1$\\times$1 QKV", C_ATTN, h=H)
    y_mha  = box(ax, next(ys), "MHA over all views' tokens (per $t'$)", C_ATTN, h=H)
    y_proj = box(ax, next(ys), "1$\\times$1 proj  (zero-init)", C_ATTN, h=H)
    y_add  = oplus(ax, next(ys))
    for a, b_ in zip([y_in, y_tok, y_norm, y_qkv, y_mha],
                     [y_tok, y_norm, y_qkv, y_mha, y_proj]):
        arrow(ax, a + 0.026, b_ - 0.026)
    arrow(ax, y_proj + 0.026, y_add - 0.016)
    skip(ax, (y_tok + y_norm) / 2, y_add)

    y_split = box(ax, 0.525, "split back into $V$ view maps", C_SHAPE, h=H)
    arrow(ax, y_add + 0.016, y_split - 0.026)

    # tree-merge container
    cont_y0, cont_y1 = 0.572, 0.852
    ax.add_patch(FancyBboxPatch((0.045, cont_y0), 0.91, cont_y1 - cont_y0,
                                boxstyle="round,pad=0.008,rounding_size=0.015",
                                fc="none", ec="#999999", lw=0.7,
                                linestyle=(0, (3, 2)), zorder=1))
    ax.text(0.115, 0.712, "$\\times\\,(V\\!-\\!1)$\nmerges\n(binary\ntree),\nown\nweights\neach",
            fontsize=4.9, color="#666666", ha="center", va="center")
    y_cat = box(ax, 0.622, "concat pair  $[2C]$", C_SHAPE, w=0.60, h=H)
    y_rb1 = box(ax, 0.692, "ResBlock  $2C\\!\\to\\!C$", C_RES, w=0.60, h=H)
    y_rb2 = box(ax, 0.762, "ResBlock  $C\\!\\to\\!C$", C_RES, w=0.60, h=H)
    arrow(ax, y_split + 0.026, y_cat - 0.026)
    arrow(ax, y_cat + 0.026, y_rb1 - 0.026)
    arrow(ax, y_rb1 + 0.026, y_rb2 - 0.026)

    y_out = box(ax, 0.91, "fused feature  $[C,\\ T'\\!,\\ H'\\!,\\ W']$", C_IO, h=H)
    arrow(ax, y_rb2 + 0.026, y_out - 0.026)


def panel_conv3d(ax):
    setup(ax, "(b) Channel-concat Conv3d")
    ys = [0.045, 0.175, 0.305, 0.435, 0.565, 0.695, 0.91]
    y_in  = box(ax, ys[0], "per-view features  $z_1,\\dots,z_V$", C_IO)
    y_cat = box(ax, ys[1], "concat on channels  $[V\\!\\cdot\\!C]$", C_SHAPE)
    y_cv  = box(ax, ys[2], "Conv3d $1\\times1\\times1$   $V\\!\\cdot\\!C\\to C$", C_CONV)
    y_gn  = box(ax, ys[3], "GroupNorm + SiLU", C_NORM)
    y_r1  = box(ax, ys[4], "3D ResBlock (symmetric, non-causal)", C_RES)
    y_r2  = box(ax, ys[5], "3D ResBlock (symmetric, non-causal)", C_RES)
    y_out = box(ax, ys[6], "fused feature  $[C,\\ T'\\!,\\ H'\\!,\\ W']$", C_IO)
    for a, b_ in zip(ys[:-1], ys[1:]):
        arrow(ax, a + 0.028, b_ - 0.028)


def panel_conv4d(ax):
    setup(ax, "(c) Factorized 4D convolution")
    ys = [0.045, 0.14, 0.235, 0.33, 0.425, 0.52, 0.615, 0.71, 0.805, 0.91]
    y_in  = box(ax, ys[0], "per-view features  $z_1,\\dots,z_V$", C_IO)
    y_st  = box(ax, ys[1], "stack view axis  $[C,\\ T'\\!,\\ V,\\ H'\\!,\\ W']$", C_SHAPE)
    y_sp  = box(ax, ys[2], "spatial Conv2d $3\\times3$  (per $v,t'$)", C_CONV)
    y_g1  = box(ax, ys[3], "GN + SiLU", C_NORM, w=0.46)
    y_tm  = box(ax, ys[4], "temporal Conv3d $3\\times3\\times3$  (per $v$)", C_CONV)
    y_g2  = box(ax, ys[5], "GN + SiLU", C_NORM, w=0.46)
    y_vw  = box(ax, ys[6], "view Conv3d $(V,3,3)$:  $V\\to1$", C_CONV)
    y_g3  = box(ax, ys[7], "GN + SiLU", C_NORM, w=0.46)
    y_rb  = box(ax, ys[8], "3D ResBlock $\\times\\,2$ (symmetric)", C_RES)
    y_out = box(ax, ys[9], "fused feature  $[C,\\ T'\\!,\\ H'\\!,\\ W']$", C_IO)
    for a, b_ in zip(ys[:-1], ys[1:]):
        arrow(ax, a + 0.024, b_ - 0.024)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(W2, 3.05))
    fig.subplots_adjust(wspace=0.09, left=0.012, right=0.988, top=0.985, bottom=0.085)
    panel_attention(axes[0])
    panel_conv3d(axes[1])
    panel_conv4d(axes[2])
    path = os.path.join(OUT, "fusion_operators.pdf")
    fig.savefig(path, pad_inches=0.01)
    print("Saved", path)


if __name__ == "__main__":
    main()
