"""Reusable 3-D mesh drawers for Benchmark C (dipping three-layer system), used
as panels (c) and (d) of Figure 13:

  fem_mesh_3d(ax)   independent FEM truth: fine, body-fitted tetrahedral mesh
                    whose interfaces follow the dip (tilted)
  disu_grid_3d(ax)  MODFLOW DISU grid: vertical-sided prisms, one cell per layer,
                    flat-topped and stair-stepped down the dip (not tilted)

Geometry matches the manuscript's Benchmark C: 1000 m down-dip x 300 m strike,
layers 40/20/40 m, shown at a representative dip of 20 degrees.
"""
import math

import matplotlib.pyplot as plt  # noqa: F401  (registers 3d via mpl_toolkits)
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

AQ, AQT = "#9ecae1", "#d8b365"
COL = [AQ, AQT, AQ]
L, WD = 1000.0, 300.0
BASE = [0.0, 40.0, 60.0, 100.0]        # layer interface depths (40/20/40)
ALPHA = 20.0
T = math.tan(math.radians(ALPHA))


def surf(x):
    return -T * x


def layer_of_depth(d):
    return 0 if d < BASE[1] else (1 if d < BASE[2] else 2)


def cuboid(x0, x1, y0, y1, zb, zt):
    p = [(x0, y0, zb), (x1, y0, zb), (x1, y1, zb), (x0, y1, zb),
         (x0, y0, zt), (x1, y0, zt), (x1, y1, zt), (x0, y1, zt)]
    return [[p[0], p[1], p[2], p[3]], [p[4], p[5], p[6], p[7]],
            [p[0], p[1], p[5], p[4]], [p[2], p[3], p[7], p[6]],
            [p[1], p[2], p[6], p[5]], [p[0], p[3], p[7], p[4]]]


def setup_ax(ax, title, tsize=10):
    ax.set_xlim(0, L); ax.set_ylim(0, WD); ax.set_zlim(surf(L) - 100, 10)
    ax.set_box_aspect((L, WD, abs(surf(L)) + 110))
    ax.set_xlabel("down-dip distance (m)", labelpad=10, fontsize=8)
    ax.set_ylabel("strike (m)", labelpad=14, fontsize=8)
    ax.set_zlabel("elevation (m)", labelpad=8, fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(axis="x", labelsize=7, pad=1)
    ax.tick_params(axis="y", labelsize=7, pad=6)
    ax.tick_params(axis="z", labelsize=7, pad=1)
    ax.view_init(elev=20, azim=-54)
    ax.set_title(title, fontsize=tsize)


def fem_mesh_3d(ax, nx=18, ny=5, title="FEM mesh"):
    """Body-fitted tetrahedral mesh: sloped (tilted) interfaces, fine sub-layers."""
    levels = np.concatenate([np.linspace(0, 40, 4), np.linspace(40, 60, 3)[1:],
                             np.linspace(60, 100, 4)[1:]])
    xs = np.linspace(0, L, nx + 1); ys = np.linspace(0, WD, ny + 1)

    def Z(x, d):
        return surf(x) - d

    tris, fc = [], []

    def quad(p, col):
        tris.append([p[0], p[1], p[2]]); tris.append([p[0], p[2], p[3]])
        fc.extend([col, col])

    for d, col in ((0.0, COL[0]), (100.0, COL[2])):           # top / bottom
        for i in range(nx):
            for j in range(ny):
                quad([(xs[i], ys[j], Z(xs[i], d)), (xs[i+1], ys[j], Z(xs[i+1], d)),
                      (xs[i+1], ys[j+1], Z(xs[i+1], d)), (xs[i], ys[j+1], Z(xs[i], d))],
                     col)
    for y in (0.0, WD):                                       # strike walls
        for i in range(nx):
            for k in range(len(levels) - 1):
                d0, d1 = levels[k], levels[k+1]
                col = COL[layer_of_depth(0.5 * (d0 + d1))]
                quad([(xs[i], y, Z(xs[i], d0)), (xs[i+1], y, Z(xs[i+1], d0)),
                      (xs[i+1], y, Z(xs[i+1], d1)), (xs[i], y, Z(xs[i], d1))], col)
    for x in (0.0, L):                                        # up-dip / toe faces
        for j in range(ny):
            for k in range(len(levels) - 1):
                d0, d1 = levels[k], levels[k+1]
                col = COL[layer_of_depth(0.5 * (d0 + d1))]
                quad([(x, ys[j], Z(x, d0)), (x, ys[j+1], Z(x, d0)),
                      (x, ys[j+1], Z(x, d1)), (x, ys[j], Z(x, d1))], col)

    ax.add_collection3d(Poly3DCollection(tris, facecolors=fc, edgecolors="0.3",
                                         linewidths=0.15))
    setup_ax(ax, title)


def disu_grid_3d(ax, ncol=14, nrow=4, title="DISU grid"):
    """Vertical prisms, one cell per layer, flat-topped (stair-stepped)."""
    xs = np.linspace(0, L, ncol + 1); ys = np.linspace(0, WD, nrow + 1)
    faces, fc = [], []
    for i in range(ncol):
        xc = 0.5 * (xs[i] + xs[i+1]); s = surf(xc)            # flat per column
        for li in range(3):
            zt, zb = s - BASE[li], s - BASE[li+1]
            for j in range(nrow):
                faces += cuboid(xs[i], xs[i+1], ys[j], ys[j+1], zb, zt)
                fc += [COL[li]] * 6
    ax.add_collection3d(Poly3DCollection(faces, facecolors=fc, edgecolors="k",
                                         linewidths=0.2))
    setup_ax(ax, title)
