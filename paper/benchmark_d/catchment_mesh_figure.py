"""Mesh figures for the field-catchment benchmark.  To keep each panel readable
the content is split across THREE figures rather than one busy multi-panel:

  fig_catchment_plan.png      (a) shared plan footprint + D8 river + transect,
                              (b) plan stencil: DISU cell centres vs FEM corners
  fig_catchment_xsection.png  (a) MODFLOW DISU cross-section (3 cells / column),
                              (b) FEM corner-grid cross-section (11 sub-layers)
  fig_catchment_3d.png        3-D layered block with the dipping aquitard

  * MODFLOW DISU : one prismatic cell per (column, geologic layer) -> 3 cells per
                   column, piecewise-constant (stair-stepped) interfaces.
  * FEM truth    : the corner grid of `benchmark_d_mesh.build_fem_corner`,
                   CELLS=[7,2,2] -> 11 sub-layers per column, sloped continuous
                   interfaces, each prism split into tetrahedra.

The catchment window is taken from a real upland-catchment model DEM in
subtropical SE Queensland, Australia.  The site is not named and axes are shown
in LOCAL metres (window origin = 0) so the figures do not disclose the
absolute location.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm as mcm
from matplotlib.colors import Normalize
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

import catchment_geom
import benchmark_d_mesh as bm

GEOL_COL = ["#9ecae1", "#d8b365", "#6baed6"]      # top aq / aquitard / lower aq
GEOL_LAB = ["top aquifer", "aquitard (tight)", "lower aquifer"]

# Site referenced only by general geography (no name / no absolute coordinates).
SITE = "upland catchment, SE Queensland, Australia"

_FULL = FuncFormatter(lambda v, _p: f"{v:,.0f}")


def _plainfmt(ax, rot=30):
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.xaxis.set_major_formatter(_FULL)
    ax.yaxis.set_major_formatter(_FULL)
    for lab in ax.get_xticklabels():
        lab.set_rotation(rot); lab.set_ha("right")


def setup():
    cm = catchment_geom.build_catchment(50.0)
    dxy = cm.dxy
    depths = bm._node_depths()                    # 12 node levels (nzp)
    nzp = depths.size
    geol_idx = [0, bm.CELLS[0], bm.CELLS[0] + bm.CELLS[1], nzp - 1]

    xs = np.unique(cm.gx); ys = np.unique(cm.gy)
    x0, y0 = xs.min(), ys.min()                    # local-coordinate origin
    nx, ny = xs.size, ys.size
    Z = np.full((ny, nx), np.nan)
    ix = np.searchsorted(xs, cm.gx); iy = np.searchsorted(ys, cm.gy)
    Z[iy, ix] = cm.zsurf

    # transect = steepest grid row, but kept away from the very edges
    lo, hi = int(0.18 * ny), int(0.82 * ny)
    relief = np.array([np.ptp(Z[r][np.isfinite(Z[r])]) if np.isfinite(Z[r]).any()
                       else 0 for r in range(ny)])
    r_t = lo + int(np.argmax(relief[lo:hi]))
    sel = np.where(iy == r_t)[0]
    order = sel[np.argsort(cm.gx[sel])]
    xt = cm.gx[order]; zt = cm.zsurf[order]
    dd = xt - xt.min()                            # distance along A-A'
    return SimpleNamespace(
        cm=cm, dxy=dxy, depths=depths, nzp=nzp, geol_idx=geol_idx,
        xs=xs, ys=ys, x0=x0, y0=y0, nx=nx, ny=ny, Z=Z, iy=iy, r_t=r_t,
        order=order, xt=xt, zt=zt, dd=dd, ztop=cm.ztop[order], y_t=ys[r_t],
        relief=relief)


# ===========================================================================
# Figure 1 — plan footprint + stencil
# ===========================================================================
def fig_plan(S):
    fig, (axp, axb) = plt.subplots(1, 2, figsize=(13, 5.6),
                                   gridspec_kw=dict(width_ratios=[1.2, 1.0],
                                                    wspace=0.32))
    dxy = S.dxy; x0, y0 = S.x0, S.y0
    xe = np.concatenate([S.xs - dxy / 2, [S.xs[-1] + dxy / 2]]) - x0
    ye = np.concatenate([S.ys - dxy / 2, [S.ys[-1] + dxy / 2]]) - y0
    pc = axp.pcolormesh(xe, ye, S.Z, cmap="terrain", shading="flat")
    fig.colorbar(pc, ax=axp, label="ground elevation (m)", pad=0.02,
                 fraction=0.046)
    # transect A-A'
    axp.plot([S.xt.min() - x0, S.xt.max() - x0], [S.y_t - y0, S.y_t - y0],
             "r-", lw=2, zorder=6)
    axp.text(S.xt.min() - x0 - 25, S.y_t - y0, "A", color="r",
             fontweight="bold", ha="right", va="center")
    axp.text(S.xt.max() - x0 + 25, S.y_t - y0, "A'", color="r",
             fontweight="bold", ha="left", va="center")
    axp.set_aspect("equal")
    axp.set_title(f"(a) Shared plan footprint — {S.cm.gx.size} columns at "
                  f"{dxy:.0f} m\n(DISU cells {S.cm.gx.size*3})", fontsize=11)
    axp.set_xlabel("local easting (m)"); axp.set_ylabel("local northing (m)")
    _plainfmt(axp)

    cx0, cy0 = S.xs[S.nx // 2], S.ys[S.ny // 2]
    win = 2.5 * dxy
    axp.add_patch(Rectangle((cx0 - win - x0, cy0 - win - y0), 2 * win, 2 * win,
                            fill=False, ec="blue", lw=1.6, zorder=7))

    # stencil panel (local coords)
    m = (np.abs(S.cm.gx - cx0) <= win + 1) & (np.abs(S.cm.gy - cy0) <= win + 1)
    for gx, gy in zip(S.cm.gx[m] - x0, S.cm.gy[m] - y0):
        axb.add_patch(Rectangle((gx - dxy / 2, gy - dxy / 2), dxy, dxy,
                                fill=False, ec="0.45", lw=0.8))
    axb.scatter(S.cm.gx[m] - x0, S.cm.gy[m] - y0, s=60, c="k", marker="s",
                label="DISU centre (1 node/layer)")
    cxs = np.unique(np.concatenate([S.cm.gx[m] - dxy / 2,
                                    S.cm.gx[m] + dxy / 2])) - x0
    cys = np.unique(np.concatenate([S.cm.gy[m] - dxy / 2,
                                    S.cm.gy[m] + dxy / 2])) - y0
    CX, CY = np.meshgrid(cxs, cys)
    axb.scatter(CX, CY, s=60, c="crimson", marker="o",
                label="FEM corner (× sub-layers)")
    axb.set_xlim(cx0 - win - x0, cx0 + win - x0)
    axb.set_ylim(cy0 - win - y0, cy0 + win - y0)
    axb.set_aspect("equal")
    axb.set_title("(b) Plan stencil [blue box in (a)]\nDISU cell centres vs "
                  "FEM corner nodes", fontsize=11)
    axb.set_xlabel("local easting (m)"); axb.set_ylabel("local northing (m)")
    _plainfmt(axb)
    axb.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
               fontsize=8, framealpha=.95, columnspacing=1.0)

    for cyA, yB in ((cy0 + win, 1.0), (cy0 - win, 0.0)):
        fig.add_artist(ConnectionPatch(
            xyA=(cx0 + win - x0, cyA - y0), coordsA=axp.transData,
            xyB=(0.0, yB), coordsB=axb.transAxes,
            color="blue", lw=1.0, ls="--", zorder=10))

    fig.suptitle(f"Field-catchment benchmark ({SITE}) — plan view",
                 fontsize=13, y=1.02)
    out = HERE / "fig_catchment_plan.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


# ===========================================================================
# Figure 2 — cross-sections
# ===========================================================================
def fig_xsection(S):
    fig, (axd, axf) = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True,
                                   gridspec_kw=dict(wspace=0.08))
    dxy = S.dxy; dd = S.dd; ztop = S.ztop; nct = dd.size

    # (c) DISU
    for c in range(nct):
        for li in range(3):
            zb = ztop[c, li + 1]; ztp = ztop[c, li]
            axd.add_patch(Rectangle((dd[c] - dxy / 2, zb), dxy, ztp - zb,
                                    facecolor=GEOL_COL[li], edgecolor="0.25",
                                    lw=0.3))
        zc = 0.5 * (ztop[c, :-1] + ztop[c, 1:])
        axd.scatter(np.full(3, dd[c]), zc, s=5, c="k", zorder=4)
    axd.scatter([], [], s=12, c="k", label="DISU node (cell centre)")
    axd.set_xlim(dd.min() - dxy, dd.max() + dxy)
    base = bm.T_TOP + bm.T_AQT + bm.T_BOT
    axd.set_ylim(S.zt.min() - base - 20, S.zt.max() + 20)
    axd.set_title("(a) MODFLOW DISU domain — 3 cells / column (stair-stepped)",
                  fontsize=11)
    axd.set_xlabel("distance along A–A' (m)"); axd.set_ylabel("elevation (m)")
    handles = [Rectangle((0, 0), 1, 1, fc=GEOL_COL[i]) for i in range(3)]
    axd.legend(handles + [plt.Line2D([0], [0], marker="o", ls="", c="k", ms=4)],
               GEOL_LAB + ["DISU node (cell centre)"], fontsize=8,
               loc="upper right")

    # (d) FEM
    XN = np.repeat(dd[:, None], S.nzp, axis=1)
    ZN = S.zt[:, None] - S.depths[None, :]
    for li in range(3):
        axf.fill_between(dd, ZN[:, S.geol_idx[li + 1]], ZN[:, S.geol_idx[li]],
                         color=GEOL_COL[li], zorder=0)
    idx = np.arange(nct * S.nzp).reshape(nct, S.nzp)
    tris = []
    for i in range(nct - 1):
        for k in range(S.nzp - 1):
            a, b = idx[i, k], idx[i + 1, k]
            c2, d = idx[i, k + 1], idx[i + 1, k + 1]
            tris.append([a, b, d]); tris.append([a, d, c2])
    tri = Triangulation(XN.ravel(), ZN.ravel(), np.array(tris))
    axf.triplot(tri, color="0.35", lw=0.25)
    axf.set_title("(b) FEM corner-grid domain — 11 sub-layers / column "
                  "(tetrahedra)", fontsize=11)
    axf.set_xlabel("distance along A–A' (m)")

    fig.suptitle(f"Field-catchment benchmark ({SITE}) — cross-section A–A' "
                 f"(steepest transect, relief {S.relief[S.r_t]:.0f} m)",
                 fontsize=13, y=1.0)
    out = HERE / "fig_catchment_xsection.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}  (transect row {S.r_t}, {nct} columns)")


# ===========================================================================
# Figure 3 — 3-D layered block
# ===========================================================================
def fig_3d(S):
    fig = plt.figure(figsize=(10, 8.5))
    ax3 = fig.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(S.xs - S.x0, S.ys - S.y0); Z = S.Z

    z_aqt_top = Z - bm.T_TOP
    z_aqt_bot = Z - bm.T_TOP - bm.T_AQT
    ax3.plot_surface(X, Y, z_aqt_top, color=GEOL_COL[1], rstride=1, cstride=1,
                     linewidth=0, antialiased=False, shade=True)
    ax3.plot_surface(X, Y, z_aqt_bot, color="#a07a2c", rstride=1, cstride=1,
                     linewidth=0, antialiased=False, shade=True)
    z_base = Z - bm.T_TOP - bm.T_AQT - bm.T_BOT
    ax3.plot_wireframe(X, Y, z_base, rstride=4, cstride=4, color="0.6",
                       linewidth=0.3)
    norm = Normalize(np.nanmin(Z), np.nanmax(Z))
    facec = mcm.terrain(norm(Z)); facec[..., 3] = 0.5
    ax3.plot_surface(X, Y, Z, facecolors=facec, rstride=1, cstride=1,
                     linewidth=0, antialiased=False, shade=False)
    ax3.plot(S.xt - S.x0, np.full(S.xt.size, S.y_t - S.y0), S.zt, "r-",
             lw=2.5, zorder=10)

    ax3.set_title("3-D layered block — ground surface, dipping tight aquitard\n"
                  "(orange slab) and base of model  [vertical exaggeration]",
                  fontsize=11)
    ax3.set_xlabel("local easting (m)", labelpad=24)
    ax3.set_ylabel("local northing (m)", labelpad=30)
    ax3.set_zlabel("elevation (m)", labelpad=8)
    ax3.xaxis.set_major_formatter(_FULL); ax3.yaxis.set_major_formatter(_FULL)
    ax3.xaxis.set_major_locator(MaxNLocator(3))
    ax3.yaxis.set_major_locator(MaxNLocator(3))
    ax3.zaxis.set_major_locator(MaxNLocator(5))
    ax3.tick_params(axis="x", labelrotation=18, pad=4, labelsize=8)
    ax3.tick_params(axis="y", labelrotation=-20, pad=6, labelsize=8)
    ax3.tick_params(axis="z", labelsize=8)
    ax3.set_box_aspect((1, 1, 0.55))
    ax3.view_init(elev=26, azim=-122)

    aqt = Rectangle((0, 0), 1, 1, fc=GEOL_COL[1])
    basep = plt.Line2D([0], [0], color="0.6", lw=1)
    aa = plt.Line2D([0], [0], color="r", lw=2)
    ax3.legend([aqt, basep, aa],
               ["tight aquitard slab", "base of model", "A–A' transect"],
               fontsize=8, loc="upper left")
    out = HERE / "fig_catchment_3d.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


def main():
    S = setup()
    fig_plan(S)
    fig_xsection(S)
    fig_3d(S)


if __name__ == "__main__":
    main()
