"""3-D mesh figure for Benchmark D (real upland catchment), the field-data analog
of Figure 13(c)/(d):

  (a) FEM truth  : smooth, body-fitted layered block whose interfaces follow the
                   real topography (continuous, sloped); fine sub-layers.
  (b) MODFLOW DISU: vertical-sided prisms, one cell per layer, flat-topped per
                    column -> a blocky / stair-stepped version of the same
                    surface (MODFLOW cannot tilt a cell).

Coarsened plan resolution for a readable render.  Site referenced only
geographically; axes in LOCAL metres (window origin = 0).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import catchment_geom                       # sets bm layer thicknesses 350/50/100
import benchmark_d_mesh as bm

AQ, AQT = "#9ecae1", "#d8b365"
COL = [AQ, AQT, AQ]
SITE = "upland catchment, SE Queensland, Australia"
DXY = 160.0                           # coarsened plan resolution for the render
ZEXAG = 1.9                           # vertical exaggeration
BASE = [0.0, bm.T_TOP, bm.T_TOP + bm.T_AQT, bm.T_TOP + bm.T_AQT + bm.T_BOT]
# fine FEM sub-layer depths (CELLS = [7, 2, 2])
LEVELS = np.concatenate([np.linspace(0, BASE[1], 8),
                         np.linspace(BASE[1], BASE[2], 3)[1:],
                         np.linspace(BASE[2], BASE[3], 3)[1:]])


def layer_of_depth(d):
    return 0 if d < BASE[1] else (1 if d < BASE[2] else 2)


def build_grid():
    cm = catchment_geom.build_catchment(DXY)
    xs = np.unique(cm.gx); ys = np.unique(cm.gy)
    x0, y0 = xs.min(), ys.min()
    nx, ny = xs.size, ys.size
    Z = np.full((ny, nx), np.nan)
    ix = np.searchsorted(xs, cm.gx); iy = np.searchsorted(ys, cm.gy)
    Z[iy, ix] = cm.zsurf
    return xs - x0, ys - y0, Z, cm.dxy


def setup_ax(ax, xs, ys, Z, title):
    zlo = np.nanmin(Z) - BASE[3]; zhi = np.nanmax(Z)
    ax.set_xlim(xs.min(), xs.max()); ax.set_ylim(ys.min(), ys.max())
    ax.set_zlim(zlo, zhi + 10)
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), (zhi - zlo) * ZEXAG))
    ax.set_xlabel("local easting (m)", labelpad=14, fontsize=8)
    ax.set_ylabel("local northing (m)", labelpad=14, fontsize=8)
    ax.set_zlabel("elevation (m)", labelpad=8, fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.zaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(labelsize=7, pad=2)
    ax.tick_params(axis="y", pad=4)
    ax.view_init(elev=24, azim=-58)
    ax.set_title(title, fontsize=10)


def fem_block(ax, xs, ys, Z):
    """Smooth body-fitted block: sloped surfaces + perimeter walls with fine
    sub-layers, coloured by geology."""
    ny, nx = Z.shape
    quads, fc = [], []

    def q(p, c):
        quads.append(p); fc.append(c)

    # top (terrain) and base surfaces, sloped between node centres
    for d, col in ((0.0, COL[0]), (BASE[3], COL[2])):
        for j in range(ny - 1):
            for i in range(nx - 1):
                q([(xs[i], ys[j], Z[j, i] - d), (xs[i+1], ys[j], Z[j, i+1] - d),
                   (xs[i+1], ys[j+1], Z[j+1, i+1] - d),
                   (xs[i], ys[j+1], Z[j+1, i] - d)], col)
    # four perimeter walls, split into fine sub-layers, coloured by layer
    def wall(seq):
        for a in range(len(seq) - 1):
            (xa, ya, za), (xb, yb, zb) = seq[a], seq[a + 1]
            for k in range(len(LEVELS) - 1):
                d0, d1 = LEVELS[k], LEVELS[k + 1]
                col = COL[layer_of_depth(0.5 * (d0 + d1))]
                q([(xa, ya, za - d0), (xb, yb, zb - d0),
                   (xb, yb, zb - d1), (xa, ya, za - d1)], col)
    wall([(xs[i], ys[0], Z[0, i]) for i in range(nx)])
    wall([(xs[i], ys[-1], Z[-1, i]) for i in range(nx)])
    wall([(xs[0], ys[j], Z[j, 0]) for j in range(ny)])
    wall([(xs[-1], ys[j], Z[j, -1]) for j in range(ny)])

    ax.add_collection3d(Poly3DCollection(quads, facecolors=fc, edgecolors="0.3",
                                         linewidths=0.12))


def disu_block(ax, xs, ys, Z, dxy, levels):
    """Vertical prisms, flat-topped per column.  `levels` are the cell-boundary
    depths: BASE (one cell per layer) or the FEM sub-layers (refined)."""
    ny, nx = Z.shape
    h = dxy / 2.0
    faces, fc = [], []
    for j in range(ny):
        for i in range(nx):
            if not np.isfinite(Z[j, i]):
                continue
            x0, x1 = xs[i] - h, xs[i] + h
            y0, y1 = ys[j] - h, ys[j] + h
            s = Z[j, i]
            for k in range(len(levels) - 1):
                zt, zb = s - levels[k], s - levels[k + 1]
                col = COL[layer_of_depth(0.5 * (levels[k] + levels[k + 1]))]
                p = [(x0, y0, zb), (x1, y0, zb), (x1, y1, zb), (x0, y1, zb),
                     (x0, y0, zt), (x1, y0, zt), (x1, y1, zt), (x0, y1, zt)]
                faces += [[p[0], p[1], p[2], p[3]], [p[4], p[5], p[6], p[7]],
                          [p[0], p[1], p[5], p[4]], [p[2], p[3], p[7], p[6]],
                          [p[1], p[2], p[6], p[5]], [p[0], p[3], p[7], p[4]]]
                fc += [col] * 6
    ax.add_collection3d(Poly3DCollection(faces, facecolors=fc, edgecolors="k",
                                         linewidths=0.12))


def make_figure(xs, ys, Z, dxy, levels, disu_title, suptitle, out):
    fig = plt.figure(figsize=(13.5, 6.2))
    axc = fig.add_subplot(1, 2, 1, projection="3d")
    fem_block(axc, xs, ys, Z)
    setup_ax(axc, xs, ys, Z, "(a) Independent FEM truth (3-D): fine body-fitted\n"
                             "mesh, interfaces follow the real topography")
    axd = fig.add_subplot(1, 2, 2, projection="3d")
    disu_block(axd, xs, ys, Z, dxy, levels)
    setup_ax(axd, xs, ys, Z, disu_title)
    fig.legend(handles=[Patch(fc=AQ, ec="k", label="aquifer"),
                        Patch(fc=AQT, ec="k", label="aquitard (tight)")],
               loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(suptitle, fontsize=12, y=0.99)
    fig.savefig(HERE / out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def main():
    xs, ys, Z, dxy = build_grid()
    # (old) one cell per geological layer
    make_figure(xs, ys, Z, dxy, BASE,
                "(b) MODFLOW DISU grid (3-D): vertical prisms, one\n"
                "cell per layer, flat-topped (stair-stepped)",
                f"Benchmark D 3-D meshes ({SITE})  —  DISU one cell per layer  "
                f"[vertical exaggeration × {ZEXAG:g}]",
                "fig_benchmarkD_mesh3d.png")
    # (new) DISU refined to the same sub-layers as the FEM
    make_figure(xs, ys, Z, dxy, LEVELS,
                "(b) MODFLOW DISU grid (3-D): refined to the FEM sub-layers\n"
                "(11 cells/column), flat-topped (stair-stepped)",
                f"Benchmark D 3-D meshes ({SITE})  —  DISU refined to FEM "
                f"resolution  [vertical exaggeration × {ZEXAG:g}]",
                "fig_benchmarkD_mesh3d_refined.png")


if __name__ == "__main__":
    main()
