"""Journal-format versions of the ORIGINAL (v2) figures that are not results of the revision.

Same artwork rules as make_journal_figures.py (8.6 or 17.6 cm, Arial 8 pt, 600 dpi RGB, no
titles inside figures). Content is unchanged except Fig. 2(a), which now shows the 250-cell
layered Voronoi grid used for Benchmark A in the revision (v2 showed the superseded 741-cell mesh).

  v2 number  final file  source of the original drawing
  Fig. 1     Fig01       PowerPoint slide 1 (figures/fig1_geometry_schematic_editable.pptx), exported
  Fig. 2     Fig02       make_paper_figures.fig_benchmark_setup (redrawn; grid_voronoi_250.json)
  Fig. 3     Fig03       make_fig_plane_geometry.py
  Fig. 4     Fig04       PowerPoint slide 3, exported
  Fig. 5     Fig05       disu paper/make_fig13.py + make_fig13_3d.py
  Fig. 6     FigS3       benchmark_d/benchmark_d_manuscript_figs.fig_setup     (moved to ESM)
  Fig. 12    Fig11       disu paper/make_fig12.py
  Fig. 13    Fig12       disu paper/make_fig14.py (dips 0-45 deg, as in v2)
  Fig. 14    Fig13       benchmark_d/benchmark_d_manuscript_figs.fig_results
  Fig. 17    Fig15       benchmark_d/make_benchmarkD_mesh3d.py (refined DISU)

Figs 1 and 4 need the slides exported first (PowerPoint, 7200 px wide) as slide1_hi.png and
slide3_hi.png in SLIDES. Benchmark D figures need the catchment elevation file read by
benchmark_d/catchment_geom.py (not distributed).
Usage: python make_journal_figures_originals.py [names...]  (default: all)
"""
import csv
import json
import os
import pathlib as pl
import sys

import numpy as np

HERE = pl.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_journal_figures as J  # noqa: E402  (style, sizes, save)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import FancyArrow, Patch, Polygon  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from PIL import Image  # noqa: E402

CM, COL, PAGE = J.CM, J.COL, J.PAGE
REV = HERE.parent
FEM = REV / "fem_reference"
BENCH_D = REV / "benchmark_d"
DISU_PAPER = HERE
SLIDES = HERE / "slides"


def cap(s):
    return s[:1].upper() + s[1:] if s else s


def restyle(ax, title=None):
    """Bring an axis drawn by an original script to the journal style."""
    if title is not None:
        ax.set_title("", loc="center")
        ax.set_title(title, fontsize=8, loc="left")
    axes = [ax.xaxis, ax.yaxis] + ([ax.zaxis] if hasattr(ax, "zaxis") else [])
    for a in axes:
        a.label.set_size(8)
        a.label.set_text(cap(a.label.get_text()))
    ax.tick_params(labelsize=8)
    leg = ax.get_legend()
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontsize(8)


# ---------------------------------------------------------------------------
# Figs 1 and 4: PowerPoint slides
# ---------------------------------------------------------------------------
def _slide(src, name, width_cm):
    im = Image.open(src).convert("RGB")
    box = Image.eval(im.convert("L"), lambda v: 255 if v < 250 else 0).getbbox()
    pad = 40
    im = im.crop((max(box[0] - pad, 0), max(box[1] - pad, 0),
                  min(box[2] + pad, im.width), min(box[3] + pad, im.height)))
    wpx = int(round(width_cm / 2.54 * J.DPI))
    im = im.resize((wpx, int(round(im.height * wpx / im.width))), Image.LANCZOS)
    im.save(J.OUT / f"{name}.tif", dpi=(J.DPI, J.DPI), compression="tiff_lzw")
    im.save(J.OUT / f"{name}.png", dpi=(J.DPI, J.DPI))
    print("wrote", name, im.size)


def fig01():
    _slide(SLIDES / "slide1_hi.png", "Fig01", 17.6)


def fig04():
    _slide(SLIDES / "slide3_hi.png", "Fig04", 17.6)


# ---------------------------------------------------------------------------
# Fig 2: Benchmark A set-up on the 250-cell grid
# ---------------------------------------------------------------------------
def fig02():
    g = json.load(open(REV / "benchmark_a" / "grid_voronoi_250.json"))
    verts = {int(v[0]): (v[1], v[2]) for v in g["vertices"]}
    polys = [[verts[int(k)] for k in c[4:4 + int(c[3])]] for c in g["cell2d"]]
    fig = plt.figure(figsize=(PAGE, 7.4 * CM), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35])
    ax = fig.add_subplot(gs[0, 0])
    ax.add_collection(PolyCollection(polys, facecolors="none", edgecolors="#7f8c8d", linewidths=0.35))
    L = 3000.0
    ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], color="black", lw=0.9)
    ax.plot([0, L], [L / 2, L / 2], color="#c0392b", lw=1.0, ls="--")
    ax.text(-90, L / 2, "A", color="#c0392b", fontweight="bold", ha="right", va="center")
    ax.text(L + 90, L / 2, "A′", color="#c0392b", fontweight="bold", ha="left", va="center")
    for y in (-250, L + 250):
        ax.text(L / 2, y, "no-flow", ha="center", va="center", color="#566573")
    ax.set_xlim(-450, L + 450)
    ax.set_ylim(-450, L + 450)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1000, 2000, 3000])
    ax.set_yticks([0, 1000, 2000, 3000])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("(a) Plan view, 250 cells per layer")

    ax = fig.add_subplot(gs[0, 1])
    slope, t, zmid = 0.18, 0.25, 1.0
    x = np.linspace(0, L, 200)
    z0 = zmid - slope * (x - L / 2) / L
    z1, z2, z3 = z0 - t, z0 - 2 * t, z0 - 3 * t
    for za, zb, c in ((z0, z1, "#f6ddcc"), (z1, z2, "#fad7a0"), (z2, z3, "#f5cba7")):
        ax.fill_between(x, za, zb, color=c, lw=0)
    for z in (z0, z1, z2, z3):
        ax.plot(x, z, color="#7f8c8d", lw=0.6)
    for xi in np.linspace(0.15 * L, 0.85 * L, 7):
        zi = zmid - slope * (xi - L / 2) / L
        ax.annotate("", xy=(xi, zi), xytext=(xi, zi + 0.2),
                    arrowprops=dict(arrowstyle="->", color="#2471a3", lw=0.8))
    ax.text(L / 2, z0[100] + 0.3, r"Recharge $R = 1\times10^{-3}$ m d$^{-1}$", ha="center", color="#2471a3")
    ax.fill_between(x, z3, z3 - 0.09, color="#aed6f1", hatch="//", edgecolor="#1f618d", linewidth=0.3)
    ax.text(L / 2, z3[100] - 0.2, "Constant head $h$ = 100 m", ha="center", va="top", color="#1f618d")
    ax.plot([0, 0], [z3[0], z0[0]], color="black", lw=1.4)
    ax.plot([L, L], [z3[-1], z0[-1]], color="black", lw=1.4)
    ax.text(-120, (z3[0] + z0[0]) / 2, "no-flow", rotation=90, va="center", ha="right")
    ax.text(L + 120, (z3[-1] + z0[-1]) / 2, "no-flow", rotation=90, va="center", ha="left")
    for i in range(3):
        ax.text(L / 2, z0[100] - t * (i + 0.5), f"Layer {i + 1} (50 m)", ha="center", va="center")
    ax.text(0.8 * L, z0[160] - 2.5 * t, r"$K$ = 1.0 m d$^{-1}$", ha="center", va="center")
    ax.text(0.8 * L, z0[160] + 0.52, "Dip α = 0–60°\n(shown 30°)", ha="center", color="#c0392b")
    ax.text(-60, z0[0] + 0.12, "A", color="#c0392b", fontweight="bold", ha="right")
    ax.text(L + 60, z0[-1] + 0.12, "A′", color="#c0392b", fontweight="bold", ha="left")
    ax.set_xlim(-420, L + 420)
    ax.set_ylim(z3[100] - 0.48, z0[100] + 0.9)
    ax.set_yticks([])
    ax.set_xlabel("Distance along A–A′ (m)")
    ax.set_ylabel("Elevation (schematic)")
    ax.set_title("(b) Section A–A′ (vertical exaggeration)")
    J.save(fig, "Fig02")


# ---------------------------------------------------------------------------
# Fig 3: local plane fit
# ---------------------------------------------------------------------------
def fig03():
    b, c, a = 0.42, 0.18, 0.6
    x0, x1, y0, y1 = 0.0, 4.0, 0.0, 4.0

    def z(x, y):
        return a + b * x + c * y

    fig = plt.figure(figsize=(COL, 8.2 * CM))
    ax = fig.add_axes([-0.06, -0.04, 1.12, 1.12], projection="3d")
    corners = np.array([[x0, y0, z(x0, y0)], [x1, y0, z(x1, y0)], [x1, y1, z(x1, y1)], [x0, y1, z(x0, y1)]])
    ax.add_collection3d(Poly3DCollection([corners], alpha=0.25, facecolor="royalblue",
                                         edgecolor="royalblue", lw=0.9))
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    ax.text(mx + 0.2, my + 0.9, z(mx, my) + 0.1, r"$z = a + bx + cy$", color="royalblue")
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [a] * 5, color="gray", lw=0.7, ls="--", alpha=0.7)
    ax.plot([x0, x0], [y0, y0], [0.0, a], color="black", lw=1.1)
    ax.plot([x0], [y0], [a], "ko", ms=2.5)
    ax.text(x0 - 0.45, y0 - 0.1, a / 2, r"$a$", ha="center")
    tx0, tx1 = 0.8, 1.8
    zb0, zt0 = z(tx0, 0), z(tx1, 0)
    ax.plot([tx0, tx1], [0, 0], [zb0, zb0], color="darkorange", lw=1.2)
    ax.plot([tx1, tx1], [0, 0], [zb0, zt0], color="darkorange", lw=1.2)
    ax.text((tx0 + tx1) / 2, -0.45, zb0 - 0.05, r"$\Delta x$", color="darkorange", ha="center")
    ax.text(tx1 + 0.12, 0, (zb0 + zt0) / 2, r"$b\,\Delta x$", color="darkorange", va="center")
    ty0, ty1 = 2.2, 3.2
    zb1, zt1 = z(0, ty0), z(0, ty1)
    ax.plot([0, 0], [ty0, ty1], [zb1, zb1], color="teal", lw=1.2)
    ax.plot([0, 0], [ty1, ty1], [zb1, zt1], color="teal", lw=1.2)
    ax.text(-0.35, (ty0 + ty1) / 2, zb1 - 0.12, r"$\Delta y$", color="teal", ha="center")
    ax.text(-0.45, ty1 + 0.1, (zb1 + zt1) / 2, r"$c\,\Delta y$", color="teal", ha="right", va="center")
    n = np.array([-b, -c, 1.0])
    n /= np.linalg.norm(n)
    pz = z(mx, my)
    ns = 1.5
    ax.quiver(mx, my, pz, *(n * ns), color="red", linewidth=1.3, arrow_length_ratio=0.12)
    ax.text(mx + n[0] * ns * 1.05, my + n[1] * ns * 1.05, pz + n[2] * ns * 1.1, "Normal", color="red", ha="center")
    ax.quiver(mx, my, pz, 0, 0, ns, color="forestgreen", linewidth=1.2, arrow_length_ratio=0.10, linestyle="dashed")
    ax.text(mx + 0.15, my - 0.1, pz + ns * 1.1, "Vertical", color="forestgreen")
    arc = np.array([(1 - s) * np.array([0, 0, 1.0]) + s * n for s in np.linspace(0, 1, 30)])
    arc /= np.linalg.norm(arc, axis=1, keepdims=True)
    r3 = 0.7
    ax.plot(mx + r3 * arc[:, 0], my + r3 * arc[:, 1], pz + r3 * arc[:, 2], color="purple", lw=1.2)
    ax.text(mx + 0.25, my, pz + 0.55, r"$\alpha$", color="purple")
    fig.text(0.03, 0.97, r"$b = \partial z/\partial x$", color="darkorange", va="top")
    fig.text(0.03, 0.91, r"$c = \partial z/\partial y$", color="teal", va="top")
    ax.set_xlabel("x", labelpad=-12)
    ax.set_ylabel("y", labelpad=-12)
    ax.set_zlabel("z", labelpad=-12)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_zlim(0, z(x1, y1) + 0.8)
    ax.view_init(elev=24, azim=-48)
    J.save(fig, "Fig03")


# ---------------------------------------------------------------------------
# Fig 5: Benchmark C concept (after disu paper/make_fig13.py)
# ---------------------------------------------------------------------------
def fig05():
    sys.path.insert(0, str(DISU_PAPER))
    import make_fig13_3d as m3
    AQ, AQT = "#9ecae1", "#d8b365"
    COLS = [AQ, AQT, AQ]
    L, S, ZI = 10.0, 0.26, [0.0, 0.4, 0.6, 1.0]
    rot = -np.degrees(np.arctan(S))

    def slab(ax):
        for li in range(3):
            zt0, zt1 = -ZI[li], -ZI[li + 1]
            ax.add_patch(Polygon([(0, zt0), (L, zt0 - S * L), (L, zt1 - S * L), (0, zt1)],
                                 facecolor=COLS[li], edgecolor="k", lw=0.5))
        ax.set_xlim(-0.6, L + 0.6)
        ax.set_ylim(-S * L - 1.9, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")

    fig = plt.figure(figsize=(PAGE, 14.0 * CM))
    gs = GridSpec(2, 2, height_ratios=[0.8, 1.3], hspace=0.02, wspace=0.08,
                  left=0.02, right=0.93, top=0.97, bottom=0.07)
    axa = fig.add_subplot(gs[0, 0])
    slab(axa)
    axa.set_title("(a) Confined case", loc="left")
    axa.annotate("Specified head, top aquifer (up-dip)", xy=(0.05, -0.2), xytext=(0.0, 0.85),
                 color="darkred", arrowprops=dict(arrowstyle="->", color="darkred", lw=0.7))
    axa.annotate("Specified head, basal aquifer (toe)", xy=(L, -0.85 - S * L), xytext=(L, -S * L - 1.75),
                 ha="right", color="darkred", arrowprops=dict(arrowstyle="->", color="darkred", lw=0.7))
    axa.annotate("", xy=(L - 1.5, -0.8 - S * (L - 1.5)), xytext=(1.5, -0.2 - S * 1.5),
                 arrowprops=dict(arrowstyle="->", color="navy", lw=1.3, connectionstyle="arc3,rad=0.25"))
    axa.text(3.3, -1.95 - S * 3.3, "Throughflow Q", color="navy", rotation=rot, rotation_mode="anchor")

    axb = fig.add_subplot(gs[0, 1])
    slab(axb)
    axb.set_title("(b) Unconfined case", loc="left")
    for xr in np.linspace(1, L - 1, 7):
        axb.add_patch(FancyArrow(xr, 0.7 - S * xr * 0.0, 0, -0.45, width=0.02, head_width=0.18,
                                 head_length=0.12, color="#2c7fb8"))
    axb.text(L / 2, 0.85, "Recharge R", ha="center", color="#2c7fb8")
    axb.plot([0, L], [-0.12, -0.12 - S * L], "b--", lw=0.9)
    axb.annotate("Water table", xy=(7.0, -0.12 - S * 7.0), xytext=(7.6, -0.2),
                 color="b", arrowprops=dict(arrowstyle="->", color="b", lw=0.7))
    axb.annotate("Specified head (toe)", xy=(L, -0.85 - S * L), xytext=(L, -S * L - 1.75),
                 ha="right", color="darkred", arrowprops=dict(arrowstyle="->", color="darkred", lw=0.7))
    xm = L / 2
    axb.annotate("", xy=(xm, -0.6 - S * xm), xytext=(xm, -0.4 - S * xm),
                 arrowprops=dict(arrowstyle="<->", color="k", lw=0.7))
    axb.annotate(r"$\Delta h$ across aquitard", xy=(xm, -0.62 - S * xm), xytext=(1.0, -S * L - 1.0),
                 arrowprops=dict(arrowstyle="-", color="k", lw=0.5))

    for spec, fn, title in ((gs[1, 0], m3.fem_mesh_3d, "(c) Finite-element mesh"),
                            (gs[1, 1], m3.disu_grid_3d, "(d) MODFLOW DISU grid")):
        ax = fig.add_subplot(spec, projection="3d")
        fn(ax, title="")
        ax.set_box_aspect((m3.L, m3.WD, abs(m3.surf(m3.L)) + 110), zoom=0.9)
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xlabel("Down-dip distance (m)", fontsize=8, labelpad=2)
        ax.set_ylabel("Strike (m)", fontsize=8, labelpad=4)
        ax.set_zlabel("Elevation (m)", fontsize=8, labelpad=4)
        ax.tick_params(labelsize=7, pad=0)
    fig.legend(handles=[Patch(fc=AQ, ec="k", lw=0.5, label="Aquifer"),
                        Patch(fc=AQT, ec="k", lw=0.5, label="Aquitard (tight)")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    J.save(fig, "Fig05")


# ---------------------------------------------------------------------------
# Figs 11 and 12: finite-element verification and Benchmark C results
# ---------------------------------------------------------------------------
def _csv(path, cols, maxdip=None):
    rows = list(csv.DictReader(open(path)))
    if maxdip is not None:
        rows = [r for r in rows if float(r["dip_deg"]) <= maxdip]
    return [np.array([float(r[c]) for r in rows]) for c in cols]


def fig11():
    dipA, fem, ana, plan = _csv(FEM / "benchmark_a_fem_vs_analytic.csv",
                                ["dip_deg", "fem_drop_mm", "analytic_drop_mm", "planview_drop_mm"])
    dipB, vp, va, vf = _csv(FEM / "benchmark_b_fem_vs_analytic.csv",
                            ["dip_deg", "fem_vs_plan_mm", "fem_vs_areas_mm", "fem_vs_full_mm"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(PAGE, 7.0 * CM), layout="constrained")
    a1.plot(dipA, plan, "o-", color=J.C_PLAN, label="Plan-view DISU")
    a1.plot(dipA, ana, "-", color="#2c3e50", lw=1.3, label=r"Analytical, $tR\cos^2\alpha/K$")
    a1.plot(dipA, fem, "D", color=J.C_FULL, ms=4, mfc="none", mew=0.9, label="Finite-element model")
    a1.set_xlabel("Dip angle (°)")
    a1.set_ylabel("Mean per-layer head drop (mm)")
    a1.set_title("(a) Benchmark A")
    a1.legend(frameon=False, loc="lower left")
    a1.grid(True, **J.GRID)
    a2.set_yscale("symlog", linthresh=1.0)
    a2.plot(dipB, vp, "o-", color=J.C_PLAN, label=r"vs plan-view leakance $K_v/b$")
    a2.plot(dipB, va, "s-", color=J.C_AREA, label=r"vs area-only $K_v/(b\cos\alpha)$")
    a2.plot(dipB, vf, "D-", color=J.C_FULL, label=r"vs full $K_v/(b\cos^2\alpha)$")
    a2.set_xlabel("Dip angle (°)")
    a2.set_ylabel("RMS head difference (mm)")
    a2.set_title("(b) Benchmark B")
    a2.legend(frameon=False, loc="upper left")
    a2.grid(True, which="both", **J.GRID)
    J.save(fig, "Fig11")


def fig12():
    dc, fc, pc, ac, uc = _csv(FEM / "benchmark_c_confined_disu_vs_fem.csv",
                              ["dip_deg", "fem_q", "plan_q", "area_q", "full_q"], maxdip=45)
    du, fu, pu, au, uu = _csv(FEM / "benchmark_c_unconfined_disu_vs_fem.csv",
                              ["dip_deg", "fem_dh", "plan_dh", "area_dh", "full_dh"], maxdip=45)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(PAGE, 7.0 * CM), layout="constrained")
    for ax, d, f, p, a_, u, ylab, title, loc in (
            (a1, dc, fc, pc, ac, uc, r"Cross-formational throughflow (m$^3$ d$^{-1}$)", "(a) Confined", "upper left"),
            (a2, du, fu, pu, au, uu, "Head drop across aquitard (m)", "(b) Unconfined", "lower left")):
        ax.plot(d, f, "-", color="#2c3e50", lw=1.6, label="Finite-element model")
        ax.plot(d, p, "o--", color=J.C_PLAN, label="Plan-view")
        ax.plot(d, a_, "s--", color=J.C_AREA, label="Area-only")
        ax.plot(d, u, "D", color=J.C_FULL, ms=5, mfc="none", mew=1.0, label="Full slope correction")
        ax.set_xlabel("Dip angle (°)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_xticks(range(0, 46, 15))
        ax.legend(frameon=False, loc=loc)
        ax.grid(True, **J.GRID)
    J.save(fig, "Fig12")


# ---------------------------------------------------------------------------
# Benchmark D: FigS3 (set-up, old Fig. 6), Fig13 (results, old Fig. 14), Fig15 (3-D meshes, old Fig. 17)
# ---------------------------------------------------------------------------
def _bd():
    sys.path.insert(0, str(BENCH_D))
    import benchmark_d_manuscript_figs as bd
    return bd


def figS3():
    bd = _bd()
    S = bd.setup()
    fig = plt.figure(figsize=(PAGE, 17.0 * CM))
    top, bot = fig.subfigures(2, 1, height_ratios=[1.1, 0.85], hspace=0.0)
    gst = top.add_gridspec(1, 2, width_ratios=[1.0, 1.1], wspace=0.25, left=0.09, right=0.99,
                           bottom=0.14, top=0.86)
    axa = top.add_subplot(gst[0, 0])
    bd.draw_plan(axa, S, top)
    restyle(axa, "(a) Plan footprint")
    for cb in top.axes:
        if cb is not axa:
            cb.tick_params(labelsize=8)
            cb.yaxis.label.set_size(8)
            cb.yaxis.label.set_text("Ground elevation (m)")
    axb = top.add_subplot(gst[0, 1], projection="3d")
    bd.draw_3d(axb, S, label="(b)")
    leg = axb.get_legend()
    handles, labels = leg.legend_handles, [cap(t.get_text()) for t in leg.get_texts()]
    leg.remove()
    restyle(axb, "(b) Layered block (vertical exaggeration)")
    axb.set_box_aspect((1, 1, 0.55), zoom=0.85)
    axb.xaxis.labelpad = 6
    axb.yaxis.labelpad = 8
    axb.zaxis.labelpad = 2
    axb.tick_params(labelsize=7, pad=0)
    top.legend(handles, labels, loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(0.99, 1.0))
    gsb = bot.add_gridspec(1, 2, wspace=0.06, left=0.09, right=0.99, bottom=0.17, top=0.9)
    axc = bot.add_subplot(gsb[0, 0])
    axd = bot.add_subplot(gsb[0, 1], sharey=axc)
    bd.draw_disu(axc, S, label="(c)")
    bd.draw_fem(axd, S, label="(d)")
    for coll in axc.collections:
        coll.set_sizes([2])
    restyle(axc, "(c) DISU cross-section A–A′")
    restyle(axd, "(d) Finite-element cross-section A–A′")
    leg = axc.get_legend()
    for t in leg.get_texts():
        t.set_text(cap(t.get_text()))
    plt.setp(axd.get_yticklabels(), visible=False)
    J.save(fig, "FigS3")


def fig13():
    bd = _bd()
    fig, (a, b) = plt.subplots(1, 2, figsize=(PAGE, 8.0 * CM), layout="constrained")
    bd.draw_baseflow(a)
    bd.draw_leakage(b)
    for t in a.texts:
        if t.get_text().endswith("%"):
            t.set_rotation(90)
            t.set_fontsize(7.5)
            t.set_y(t.get_position()[1] + 4)
        else:
            t.set_fontsize(8)
    femv = [p.get_height() for p in a.patches][0]
    a.set_ylim(0, femv * 1.5)
    a.legend(loc="upper right", ncol=1, frameon=False, fontsize=8)
    restyle(a, "(a) River baseflow")
    restyle(b, "(b) Aquitard leakage per column")
    a.set_ylabel(r"River baseflow (m$^3$ d$^{-1}$)")
    b.set_ylabel(r"Aquitard leakage per column (m$^3$ d$^{-1}$)")
    b.set_xlabel(r"$\sec^2\alpha$ (local aquitard dip)")
    J.save(fig, "Fig13")


def fig15():
    sys.path.insert(0, str(BENCH_D))
    import make_benchmarkD_mesh3d as m3d
    xs, ys, Z, dxy = m3d.build_grid()
    fig = plt.figure(figsize=(PAGE, 8.6 * CM))
    panels = ((lambda ax: m3d.fem_block(ax, xs, ys, Z), "(a) Finite-element mesh"),
              (lambda ax: m3d.disu_block(ax, xs, ys, Z, dxy, m3d.LEVELS),
               "(b) DISU grid refined to 11 cells per column"))
    for k, (fn, title) in enumerate(panels):
        ax = fig.add_axes([0.01 + 0.5 * k, 0.08, 0.47, 0.82], projection="3d")
        fn(ax)
        m3d.setup_ax(ax, xs, ys, Z, "")
        zlo, zhi = np.nanmin(Z) - m3d.BASE[3], np.nanmax(Z)
        ax.set_box_aspect((np.ptp(xs), np.ptp(ys), (zhi - zlo) * m3d.ZEXAG), zoom=0.85)
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xlabel("Local easting (m)", fontsize=8, labelpad=2)
        ax.set_ylabel("Local northing (m)", fontsize=8, labelpad=2)
        ax.set_zlabel("Elevation (m)", fontsize=8, labelpad=2)
        ax.tick_params(labelsize=7, pad=0)
    fig.legend(handles=[Patch(fc=m3d.AQ, ec="k", lw=0.5, label="Aquifer"),
                        Patch(fc=m3d.AQT, ec="k", lw=0.5, label="Aquitard (tight)")],
               loc="lower left", ncol=2, frameon=False, bbox_to_anchor=(0.01, 0.0))
    fig.text(0.99, 0.02, f"Vertical exaggeration ×{m3d.ZEXAG:g}", ha="right", va="bottom")
    J.save(fig, "Fig15")


ALL = {"Fig01": fig01, "Fig02": fig02, "Fig03": fig03, "Fig04": fig04, "Fig05": fig05, "Fig11": fig11,
       "Fig12": fig12, "Fig13": fig13, "Fig15": fig15, "FigS3": figS3}

if __name__ == "__main__":
    for name in (sys.argv[1:] or ALL):
        ALL[name]()
