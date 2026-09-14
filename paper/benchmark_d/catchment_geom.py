"""Trimmed catchment demonstration geometry: a small structured grid over a steep,
window, with the TOPO borrowed from the real catchment model and three
topo-parallel layers over a 500 m depth (homogeneous K, steady state).

Produces a benchmark_d_mesh.CatchmentMesh so the existing corner-grid FEM truth
and DISU machinery can be reused directly.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator

import benchmark_d_mesh as bm

# 3 topo-parallel layers spanning 500 m.  Deep-river configuration: recharge on
# top must drain DOWN through the dipping aquitard to a river incised into the
# lower aquifer, so the aquitard's conductance controls the flow (Case-C
# mechanism on real topo).  The top aquifer is thick (350 m) so the regional
# water table stays above the aquitard and keeps it saturated everywhere.
bm.T_TOP, bm.T_AQT, bm.T_BOT = 350.0, 50.0, 100.0
bm.CELLS = [7, 2, 2]                       # FEM sub-layers per geological layer

# Land-surface elevations of the catchment are not distributed (real site). Supply a CSV with
# columns easting, northing, L1_top, and the window centre and half-width in the same coordinates.
TOP_CSV = os.environ.get("BENCHMARK_D_TOPO", "")
WIN_CX, WIN_CY, WIN_HALF = (float(v) for v in os.environ.get("BENCHMARK_D_WINDOW", "0,0,1000").split(","))
DEPTH = 500.0

_interp = None


def _topo_interp():
    global _interp
    if _interp is None:
        d = np.genfromtxt(TOP_CSV, delimiter=",", names=True)
        _interp = LinearNDInterpolator(np.c_[d["easting"], d["northing"]], d["L1_top"])
    return _interp


def build_catchment(dxy: float = 50.0) -> bm.CatchmentMesh:
    interp = _topo_interp()
    xs = np.arange(WIN_CX - WIN_HALF, WIN_CX + WIN_HALF + dxy, dxy)
    ys = np.arange(WIN_CY - WIN_HALF, WIN_CY + WIN_HALF + dxy, dxy)
    nx, ny = xs.size, ys.size
    X, Y = np.meshgrid(xs, ys)
    Z = interp(X, Y)
    valid = np.isfinite(Z)
    rows, cols = np.where(valid)
    gx = X[rows, cols]; gy = Y[rows, cols]; zsurf = Z[rows, cols]
    iface = np.cumsum([0.0, bm.T_TOP, bm.T_AQT, bm.T_BOT])
    ztop = zsurf[:, None] - iface[None, :]
    j = int(np.argmin(zsurf))
    outlet = np.array([gx[j], gy[j], zsurf[j]])
    return bm.CatchmentMesh(gx=gx, gy=gy, zsurf=zsurf, ztop=ztop,
                            col_ij=np.c_[rows, cols], nx=nx, ny=ny,
                            valid=valid, dxy=dxy, outlet=outlet)


if __name__ == "__main__":
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = build_catchment(50.0)
    g = cm.zsurf
    # local slope within the window
    Z = np.full((cm.ny, cm.nx), np.nan)
    Z[cm.col_ij[:, 0], cm.col_ij[:, 1]] = cm.zsurf
    gyv, gxv = np.gradient(Z, cm.dxy, cm.dxy)
    slope = np.degrees(np.arctan(np.hypot(gxv, gyv)))
    sl = slope[np.isfinite(slope)]
    print(f"window: {cm.gx.size} cells at {cm.dxy:.0f} m; topo {g.min():.0f}-{g.max():.0f} m")
    print(f"slope deg: median {np.median(sl):.1f}, mean {np.mean(sl):.1f}, "
          f"95th {np.percentile(sl,95):.1f}, max {sl.max():.1f}")
    print(f"layers (topo-parallel): {bm.T_TOP:.0f}/{bm.T_AQT:.0f}/{bm.T_BOT:.0f} m, "
          f"base = topo - {DEPTH:.0f} m")

    # cross-section along the steepest transect (max topo range row)
    OUT = Path(__file__).resolve().parent
    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_subplot(1, 2, 1)
    pc = ax.scatter(cm.gx, cm.gy, c=cm.zsurf, s=8, cmap="terrain", marker="s")
    fig.colorbar(pc, ax=ax, label="topo (m)"); ax.set_aspect("equal")
    ax.set_title("Trimmed catchment window (borrowed topo)")
    ax.set_xlabel("easting"); ax.set_ylabel("northing")
    # a cross-section: middle row
    midrow = cm.ny // 2
    sel = cm.col_ij[:, 0] == midrow
    order = np.argsort(cm.gx[sel])
    xx = cm.gx[sel][order]
    zt = cm.ztop[sel]
    ax2 = fig.add_subplot(1, 2, 2)
    cols3 = ["#9ecae1", "#d8b365", "#6baed6"]
    for li in range(3):
        ax2.fill_between(xx, zt[order, li + 1], zt[order, li], color=cols3[li],
                         label=f"layer {li+1}")
    ax2.set_title("Cross-section: 3 topo-parallel layers (dipping)")
    ax2.set_xlabel("easting"); ax2.set_ylabel("elevation (m)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_catchment_geom.png", dpi=160)
    print("wrote fig_catchment_geom.png")
