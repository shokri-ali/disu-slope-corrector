"""Benchmark D meshing on the real (anonymised) irregular catchment.

Builds, on the catchment's true irregular footprint (NOT a box), at a chosen
tilt about the river outlet:
  * the MODFLOW DISU grid  -- one prismatic cell per (column, layer),
  * the FEM tetrahedral mesh -- each prism split into tets,
3 layers following topography (aquifer / aquitard / aquifer), like Case C.

Only DEM columns inside the catchment are kept, so both meshes share the same
irregular plan footprint.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import LinearNDInterpolator

CSV = os.environ.get("BENCHMARK_D_DEM", "")   # land-surface elevations: not distributed

T_TOP, T_AQT, T_BOT = 40.0, 20.0, 40.0          # layer thicknesses [m]
CELLS = [3, 2, 3]                                # FEM sub-layers per geol layer
DXY = 100.0                                      # plan resolution [m]


@dataclass
class CatchmentMesh:
    # plan grid (regular, masked)
    gx: np.ndarray            # (ncol,) cell-centre x of valid columns
    gy: np.ndarray            # (ncol,) cell-centre y
    zsurf: np.ndarray         # (ncol,) tilted ground-surface elevation
    # per-column layer interface elevations, top->bottom (nlay+1 surfaces)
    ztop: np.ndarray          # (ncol, 4) interfaces: surf, -T_TOP, -T_AQT, -T_BOT
    col_ij: np.ndarray        # (ncol,2) row,col index in the full grid
    nx: int
    ny: int
    valid: np.ndarray         # (ny,nx) bool mask
    dxy: float
    outlet: np.ndarray        # (3,) outlet x,y,z (pivot)


def _load_grid():
    d = np.genfromtxt(CSV, delimiter=",", names=True)
    x, y, z = d["x"], d["y"], d["z"]
    x = x - x.min(); y = y - y.min()
    return x, y, z


def _downstream_frame(x, y, z):
    A = np.c_[np.ones_like(x), x, y]
    _, b, c = np.linalg.lstsq(A, z, rcond=None)[0]
    e_down = -np.array([b, c]) / np.hypot(b, c)
    j = np.argmin(z)
    return e_down, np.array([x[j], y[j], z[j]])


def build(tilt_deg: float = 0.0, dxy: float = DXY,
          elev_factor: float = 1.0) -> CatchmentMesh:
    """Build the catchment mesh.  Two ways to steepen the geology:
    * tilt_deg     : rigidly rotate the catchment about the river outlet, or
    * elev_factor  : multiply elevations by a factor (vertical exaggeration),
                     which scales every surface gradient (and the aquitard dip)
                     by that factor while keeping the catchment draining to the
                     same outlet.  Use one or the other (factor != 1 -> no tilt).
    """
    x, y, z = _load_grid()
    e_down, outlet = _downstream_frame(x, y, z)
    e_cross = np.array([-e_down[1], e_down[0]])
    interp = LinearNDInterpolator(np.c_[x, y], z)

    # regular plan grid at dxy, mask to catchment (where DEM interpolates finite)
    xs = np.arange(x.min(), x.max() + dxy, dxy)
    ys = np.arange(y.min(), y.max() + dxy, dxy)
    nx, ny = xs.size, ys.size
    X, Y = np.meshgrid(xs, ys)
    Z0 = interp(X, Y)
    valid = np.isfinite(Z0)

    if elev_factor != 1.0:
        # vertical exaggeration about the outlet level (outlet elevation fixed)
        Zt = outlet[2] + elev_factor * (Z0 - outlet[2])
    else:
        # rigid rotation about the outlet (untilted z -> tilted z), in (u, z) plane
        th = math.radians(tilt_deg)
        du = (X - outlet[0]) * e_down[0] + (Y - outlet[1]) * e_down[1]
        dz = Z0 - outlet[2]
        Zt = -du * math.sin(th) + dz * math.cos(th) + outlet[2]

    rows, cols = np.where(valid)
    gx = X[rows, cols]; gy = Y[rows, cols]
    zsurf = Zt[rows, cols]
    iface = np.cumsum([0.0, T_TOP, T_AQT, T_BOT])
    ztop = zsurf[:, None] - iface[None, :]       # (ncol, 4): surface .. base
    return CatchmentMesh(gx=gx, gy=gy, zsurf=zsurf, ztop=ztop,
                         col_ij=np.c_[rows, cols], nx=nx, ny=ny,
                         valid=valid, dxy=dxy, outlet=outlet)


def _node_depths():
    base = [0.0, T_TOP, T_TOP + T_AQT, T_TOP + T_AQT + T_BOT]
    segs = []
    for li in range(3):
        seg = np.linspace(base[li], base[li + 1], CELLS[li] + 1)
        segs.append(seg if li == 0 else seg[1:])
    return np.concatenate(segs)


def _col_lookup(cm: CatchmentMesh):
    return {(int(r), int(c)): k for k, (r, c) in enumerate(cm.col_ij)}


def build_fem_tetmesh(cm: CatchmentMesh):
    """Tetrahedral mesh of the irregular catchment (for the FEM truth)."""
    from skfem import MeshTet
    depths = _node_depths(); nzp = depths.size
    ncol = cm.gx.size
    pts = np.empty((3, ncol * nzp))
    for c in range(ncol):
        col = depths * 0 + cm.zsurf[c]
        pts[0, c * nzp:(c + 1) * nzp] = cm.gx[c]
        pts[1, c * nzp:(c + 1) * nzp] = cm.gy[c]
        pts[2, c * nzp:(c + 1) * nzp] = cm.zsurf[c] - depths
    look = _col_lookup(cm)
    split = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
             (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
    tets = []
    for (r, c0), c00 in look.items():
        c10 = look.get((r + 1, c0)); c11 = look.get((r + 1, c0 + 1))
        c01 = look.get((r, c0 + 1))
        if None in (c10, c11, c01):
            continue
        base = [c00, c10, c11, c01]
        for k in range(nzp - 1):
            v = [b * nzp + k for b in base] + [b * nzp + k + 1 for b in base]
            for a, bb, cc, e in split:
                tets.append([v[a], v[bb], v[cc], v[e]])
    return MeshTet(pts, np.array(tets, dtype=np.int64).T)


def build_fem_corner(cm: CatchmentMesh):
    """Corner-grid tet mesh: every DISU cell becomes ONE hexahedral column
    (4 corner nodes x sub-layers), so the FEM footprint and plan area match the
    DISU grid exactly (no half-cell boundary loss, no dropped columns).

    Returns (mesh, cell2corners, nzp): cell2corners maps a column index to its
    four corner-node base ids (multiply by nzp and add k for the node)."""
    from skfem import MeshTet
    depths = _node_depths(); nzp = depths.size
    dxy = cm.dxy
    r0, c0 = int(cm.col_ij[0][0]), int(cm.col_ij[0][1])
    x0 = cm.gx[0] - c0 * dxy
    y0 = cm.gy[0] - r0 * dxy

    corner_id: dict[tuple[int, int], int] = {}

    def gc(cr, cc):
        key = (cr, cc)
        if key not in corner_id:
            corner_id[key] = len(corner_id)
        return corner_id[key]

    cell2corners = {}
    for k in range(cm.gx.size):
        r, c = int(cm.col_ij[k][0]), int(cm.col_ij[k][1])
        cell2corners[k] = (gc(r, c), gc(r + 1, c), gc(r + 1, c + 1), gc(r, c + 1))
    ncorner = len(corner_id)

    # corner surface = mean surface of the (<=4) cells touching it
    acc = np.zeros(ncorner); cnt = np.zeros(ncorner)
    for k in range(cm.gx.size):
        for cidx in cell2corners[k]:
            acc[cidx] += cm.zsurf[k]; cnt[cidx] += 1
    csurf = acc / cnt
    cxy = np.zeros((ncorner, 2))
    for (cr, cc), cidx in corner_id.items():
        cxy[cidx] = (x0 + (cc - 0.5) * dxy, y0 + (cr - 0.5) * dxy)

    pts = np.empty((3, ncorner * nzp))
    for cidx in range(ncorner):
        sl = slice(cidx * nzp, (cidx + 1) * nzp)
        pts[0, sl] = cxy[cidx, 0]; pts[1, sl] = cxy[cidx, 1]
        pts[2, sl] = csurf[cidx] - depths

    split = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
             (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
    tets = []
    for k in range(cm.gx.size):
        ids = cell2corners[k]
        for s in range(nzp - 1):
            v = [i * nzp + s for i in ids] + [i * nzp + s + 1 for i in ids]
            for a, b, c, e in split:
                tets.append([v[a], v[b], v[c], v[e]])
    return MeshTet(pts, np.array(tets, dtype=np.int64).T), cell2corners, nzp


def active_columns(cm: CatchmentMesh):
    """Columns that take part in at least one complete plan unit-square, i.e.
    exactly those the FEM tet mesh uses.  Use the same set for the DISU grid so
    both share an identical footprint."""
    look = _col_lookup(cm)
    active = set()
    for (r, c), k in look.items():
        if (r + 1, c) in look and (r, c + 1) in look and (r + 1, c + 1) in look:
            active.update(look[key] for key in
                          ((r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)))
    return np.array(sorted(active))


def geol_layer_prisms(cm: CatchmentMesh):
    """Yield (col_index, geol_layer, ztop, zbot) for each DISU prismatic cell."""
    for c in range(cm.gx.size):
        for li in range(3):
            yield c, li, cm.ztop[c, li], cm.ztop[c, li + 1]


if __name__ == "__main__":
    cm = build(15.0)
    ncol = cm.gx.size
    print(f"catchment columns kept: {ncol} of {cm.nx*cm.ny} box cells "
          f"({100*ncol/(cm.nx*cm.ny):.0f}%) at {cm.dxy:.0f} m")
    print(f"DISU cells (3 geol layers): {ncol*3}")
    nz = sum(CELLS)
    print(f"FEM columns x {nz} sublayers -> ~{ncol*nz} hexes -> "
          f"~{ncol*nz*6} tets")
