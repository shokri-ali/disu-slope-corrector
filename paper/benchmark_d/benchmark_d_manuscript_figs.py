"""Benchmark D manuscript figures (3 figures, as agreed):

  fig_benchmarkD_domain.png    Fig 15 — (a) plan footprint, (b) 3-D layered block
  fig_benchmarkD_mesh.png      Fig 16 — (a) DISU cross-section, (b) FEM cross-
                               section, (c) plan stencil (centres vs corners)
  fig_benchmarkD_results.png   Fig 17 — (a) river baseflow FEM vs plan/area/full,
                               (b) per-column aquitard leakage vs sec^2(alpha)

Geometry/setup is reused from `catchment_mesh_figure.setup`; the results panel reads
the reproducible CSVs written by `catchment_solve_leaky.py` (`catchment_baseflow.csv`,
`catchment_leakage_vs_dip.csv`).  The site is referenced only by general geography and
all coordinates are LOCAL metres (window origin = 0).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm as mcm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import benchmark_d_mesh as bm
from catchment_mesh_figure import setup, GEOL_COL, GEOL_LAB, SITE, _FULL, _plainfmt


# ---------------------------------------------------------------------------
# panel drawers
# ---------------------------------------------------------------------------
def draw_plan(ax, S, fig):
    dxy, x0, y0 = S.dxy, S.x0, S.y0
    xe = np.concatenate([S.xs - dxy / 2, [S.xs[-1] + dxy / 2]]) - x0
    ye = np.concatenate([S.ys - dxy / 2, [S.ys[-1] + dxy / 2]]) - y0
    # smooth (bilinear) interpolation rather than blocky cells
    pc = ax.imshow(S.Z, extent=[xe.min(), xe.max(), ye.min(), ye.max()],
                   origin="lower", cmap="terrain", interpolation="bilinear",
                   aspect="equal")
    fig.colorbar(pc, ax=ax, label="ground elevation (m)", pad=0.02, fraction=0.046)
    ax.plot([S.xt.min() - x0, S.xt.max() - x0], [S.y_t - y0, S.y_t - y0],
            "r-", lw=2, zorder=6)
    ax.text(S.xt.min() - x0 - 25, S.y_t - y0, "A", color="r", fontweight="bold",
            ha="right", va="center")
    ax.text(S.xt.max() - x0 + 25, S.y_t - y0, "A'", color="r", fontweight="bold",
            ha="left", va="center")
    ax.set_aspect("equal")
    ax.set_title(f"(a) Plan footprint — {S.cm.gx.size} columns at {dxy:.0f} m "
                 f"(DISU cells {S.cm.gx.size*3})", fontsize=10)
    ax.set_xlabel("local easting (m)"); ax.set_ylabel("local northing (m)")
    _plainfmt(ax)


def draw_3d(ax, S, label="(b)"):
    X, Y = np.meshgrid(S.xs - S.x0, S.ys - S.y0); Z = S.Z
    ax.plot_surface(X, Y, Z - bm.T_TOP, color=GEOL_COL[1], rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=True)
    ax.plot_surface(X, Y, Z - bm.T_TOP - bm.T_AQT, color="#a07a2c", rstride=1,
                    cstride=1, linewidth=0, antialiased=False, shade=True)
    ax.plot_wireframe(X, Y, Z - bm.T_TOP - bm.T_AQT - bm.T_BOT, rstride=4,
                      cstride=4, color="0.6", linewidth=0.3)
    norm = Normalize(np.nanmin(Z), np.nanmax(Z))
    fc = mcm.terrain(norm(Z)); fc[..., 3] = 0.5
    ax.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=1, linewidth=0,
                    antialiased=False, shade=False)
    ax.plot(S.xt - S.x0, np.full(S.xt.size, S.y_t - S.y0), S.zt, "r-", lw=2.5,
            zorder=10)
    ax.set_title(f"{label} 3-D layered block: ground surface, dipping tight "
                 "aquitard\n(orange slab) and base of model  [vertical "
                 "exaggeration]", fontsize=10)
    ax.set_xlabel("local easting (m)", labelpad=22)
    ax.set_ylabel("local northing (m)", labelpad=28)
    ax.set_zlabel("elevation (m)", labelpad=8)
    ax.xaxis.set_major_formatter(_FULL); ax.yaxis.set_major_formatter(_FULL)
    ax.xaxis.set_major_locator(MaxNLocator(3)); ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(axis="x", labelrotation=18, pad=3, labelsize=8)
    ax.tick_params(axis="y", labelrotation=-20, pad=5, labelsize=8)
    ax.tick_params(axis="z", labelsize=8)
    ax.set_box_aspect((1, 1, 0.55)); ax.view_init(elev=26, azim=-122)
    ax.legend([Rectangle((0, 0), 1, 1, fc=GEOL_COL[1]),
               plt.Line2D([0], [0], color="0.6", lw=1),
               plt.Line2D([0], [0], color="r", lw=2)],
              ["tight aquitard slab", "base of model", "A–A' transect"],
              fontsize=8, loc="upper left")


def draw_disu(ax, S, label="(a)"):
    dxy, dd, ztop, nct = S.dxy, S.dd, S.ztop, S.dd.size
    for c in range(nct):
        for li in range(3):
            zb = ztop[c, li + 1]; zt = ztop[c, li]
            ax.add_patch(Rectangle((dd[c] - dxy / 2, zb), dxy, zt - zb,
                                   facecolor=GEOL_COL[li], edgecolor="0.25", lw=0.3))
        zc = 0.5 * (ztop[c, :-1] + ztop[c, 1:])
        ax.scatter(np.full(3, dd[c]), zc, s=5, c="k", zorder=4)
    ax.set_xlim(dd.min() - dxy, dd.max() + dxy)
    base = bm.T_TOP + bm.T_AQT + bm.T_BOT
    ax.set_ylim(S.zt.min() - base - 20, S.zt.max() + 20)
    ax.set_title(f"{label} MODFLOW DISU domain — 3 cells / column (stair-stepped)",
                 fontsize=10)
    ax.set_xlabel("distance along A–A' (m)"); ax.set_ylabel("elevation (m)")
    ax.legend([Rectangle((0, 0), 1, 1, fc=GEOL_COL[i]) for i in range(3)]
              + [plt.Line2D([0], [0], marker="o", ls="", c="k", ms=4)],
              GEOL_LAB + ["DISU node (cell centre)"], fontsize=8, loc="upper right")


def draw_fem(ax, S, label="(b)"):
    dd, nct, nzp = S.dd, S.dd.size, S.nzp
    XN = np.repeat(dd[:, None], nzp, axis=1)
    ZN = S.zt[:, None] - S.depths[None, :]
    for li in range(3):
        ax.fill_between(dd, ZN[:, S.geol_idx[li + 1]], ZN[:, S.geol_idx[li]],
                        color=GEOL_COL[li], zorder=0)
    idx = np.arange(nct * nzp).reshape(nct, nzp)
    tris = []
    for i in range(nct - 1):
        for k in range(nzp - 1):
            a, b = idx[i, k], idx[i + 1, k]; c2, d = idx[i, k + 1], idx[i + 1, k + 1]
            tris.append([a, b, d]); tris.append([a, d, c2])
    ax.triplot(Triangulation(XN.ravel(), ZN.ravel(), np.array(tris)),
               color="0.35", lw=0.25)
    ax.set_title(f"{label} FEM corner-grid domain — 11 sub-layers / column "
                 "(tetrahedra)", fontsize=10)
    ax.set_xlabel("distance along A–A' (m)")


def draw_stencil(ax, S, label="(c)"):
    dxy, x0, y0 = S.dxy, S.x0, S.y0
    cx0, cy0 = S.xs[S.nx // 2], S.ys[S.ny // 2]
    win = 2.5 * dxy
    m = (np.abs(S.cm.gx - cx0) <= win + 1) & (np.abs(S.cm.gy - cy0) <= win + 1)
    for gx, gy in zip(S.cm.gx[m] - x0, S.cm.gy[m] - y0):
        ax.add_patch(Rectangle((gx - dxy / 2, gy - dxy / 2), dxy, dxy,
                               fill=False, ec="0.45", lw=0.8))
    ax.scatter(S.cm.gx[m] - x0, S.cm.gy[m] - y0, s=55, c="k", marker="s",
               label="DISU centre (1 node/layer)")
    cxs = np.unique(np.concatenate([S.cm.gx[m] - dxy/2, S.cm.gx[m] + dxy/2])) - x0
    cys = np.unique(np.concatenate([S.cm.gy[m] - dxy/2, S.cm.gy[m] + dxy/2])) - y0
    CX, CY = np.meshgrid(cxs, cys)
    ax.scatter(CX, CY, s=55, c="crimson", marker="o",
               label="FEM corner (× sub-layers)")
    ax.set_xlim(cx0 - win - x0, cx0 + win - x0)
    ax.set_ylim(cy0 - win - y0, cy0 + win - y0)
    ax.set_aspect("equal")
    ax.set_title(f"{label} Plan stencil (representative interior block): "
                 "DISU cell centres vs FEM corner nodes", fontsize=10)
    ax.set_xlabel("local easting (m)"); ax.set_ylabel("local northing (m)")
    _plainfmt(ax)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, framealpha=.95, columnspacing=1.0)


def draw_baseflow(ax, label="(a)"):
    rows = list(csv.DictReader((HERE / "catchment_baseflow.csv").open()))
    data = {(r["method"], r["resolution"]):
            (float(r["baseflow_m3d"]), float(r["err_pct_vs_fem"])) for r in rows}
    femv = data[("fem", "-")][0]
    methods = ["plan", "area", "full"]
    names = {"plan": "plan", "area": "area", "full": "full\n(slope-corrected)"}
    mcolor = {"plan": "#d6604d", "area": "#f4a582", "full": "#4393c3"}
    w = 0.36
    ax.bar(0, femv, width=0.5, color="0.35", edgecolor="0.2")
    ax.text(0, femv + 6, "truth", ha="center", va="bottom", fontsize=8)
    ax.axhline(femv, color="0.35", ls="--", lw=1, zorder=0)
    for i, m in enumerate(methods):
        x = i + 1
        vc, ec = data[(m, "coarse")]; vr, er = data[(m, "refined")]
        ax.bar(x - w / 2, vc, width=w, color=mcolor[m], edgecolor="0.2",
               label="coarse (1 cell/layer)" if i == 0 else None)
        ax.bar(x + w / 2, vr, width=w, color=mcolor[m], edgecolor="0.2",
               hatch="///", label="refined (FEM sub-layers)" if i == 0 else None)
        ax.text(x - w / 2, vc + 6, f"{ec:+.1f}%", ha="center", va="bottom",
                fontsize=7)
        ax.text(x + w / 2, vr + 6, f"{er:+.1f}%", ha="center", va="bottom",
                fontsize=7)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["FEM truth"] + [names[m] for m in methods], fontsize=9)
    ax.set_ylabel("river baseflow [m$^3$ d$^{-1}$]")
    ax.set_ylim(0, femv * 1.2)
    ax.set_title(f"{label} River baseflow: FEM truth vs DISU "
                 "(two resolutions)", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=.95)


def draw_leakage(ax, label="(b)"):
    d = np.genfromtxt(HERE / "catchment_leakage_vs_dip.csv", delimiter=",", names=True)
    sec2 = d["sec2_alpha"]
    ok = np.isfinite(sec2) & np.isfinite(d["leak_plan"]) & (d["leak_plan"] > 0)
    ax.scatter(sec2[ok], d["leak_plan"][ok], s=9, alpha=.45, c="#d6604d",
               label="plan")
    ax.scatter(sec2[ok], d["leak_area"][ok], s=9, alpha=.45, c="#f4a582",
               label="area (× sec α)")
    ax.scatter(sec2[ok], d["leak_full"][ok], s=9, alpha=.45, c="#4393c3",
               label="full (× sec² α)")
    # reference: full = (median plan) × sec²α
    q0 = np.nanmedian(d["leak_plan"][ok])
    xx = np.linspace(1, np.nanmax(sec2[ok]), 50)
    ax.plot(xx, q0 * xx, "k:", lw=1, label=r"$q_0\,\sec^2\alpha$ (theory)")
    ax.set_xlabel(r"$\sec^2\alpha$  (local aquitard dip)")
    ax.set_ylabel("per-column aquitard leakage [m$^3$ d$^{-1}$]")
    ax.set_title(f"{label} Per-column leakage scales with the dip", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_setup(S):
    """Combined 4-panel figure: (a) plan, (b) 3-D block, (c) DISU and (d) FEM
    cross-sections.  Sub-figures give the top and bottom rows independent
    spacing (the 3-D panel needs room; the cross-sections sit close)."""
    fig = plt.figure(figsize=(13.5, 10.5))
    top, bot = fig.subfigures(2, 1, height_ratios=[1.1, 0.82], hspace=0.02)

    gst = top.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.34)
    draw_plan(top.add_subplot(gst[0, 0]), S, top)
    draw_3d(top.add_subplot(gst[0, 1], projection="3d"), S, label="(b)")

    gsb = bot.add_gridspec(1, 2, wspace=0.08)
    axc = bot.add_subplot(gsb[0, 0])
    axd = bot.add_subplot(gsb[0, 1], sharey=axc)
    draw_disu(axc, S, label="(c)")
    draw_fem(axd, S, label="(d)")
    plt.setp(axd.get_yticklabels(), visible=False)

    fig.suptitle(f"Benchmark D domain and meshes ({SITE})", fontsize=13, y=1.0)
    out = HERE / "fig_benchmarkD_setup.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out.name}")


def fig_results(S):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8),
                               gridspec_kw=dict(wspace=0.26))
    draw_baseflow(a); draw_leakage(b)
    fig.suptitle(f"Benchmark D results ({SITE})", fontsize=13, y=1.0)
    out = HERE / "fig_benchmarkD_results.png"
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out.name}")


def main():
    S = setup()
    fig_setup(S)
    fig_results(S)


if __name__ == "__main__":
    main()
