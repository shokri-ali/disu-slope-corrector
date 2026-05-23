"""Apply slope-aware geometric corrections to a layered DISU connection set.

This module is format-agnostic: it consumes plain numpy arrays describing the
DISU layout and returns corrected arrays. The MF6 and MFUSG adapters in
``mf6.py`` and ``mfusg.py`` wrap this with file I/O.

The algorithm (matches Algorithm 1 in Shokri 2026, Section 2.3):

    1. Identify cell columns by clustering centroids in plan view.
    2. Within each column, sort cells by elevation. Adjacent pairs in the
       sorted list are vertically-connected stacked cells.
    3. For each vertical connection (i, j) where i is above j:
         a. Interface elevation at the column is bot_i = top_j.
         b. Collect interface elevations from the ``k`` nearest columns.
         c. Fit a plane to (x, y, z_interface) over those samples.
         d. Compute cos(alpha) and clip to ``max_dip_deg``.
         e. Multiply face area by sec(alpha).
         f. Multiply connection length by cos(alpha).
    4. For each horizontal connection (i, j) in the same layer:
         a. Compute the true elevation overlap (Eq. 9).
         b. If overlap <= 0, set connection conductance to zero.
         c. Otherwise replace the face height with the overlap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .geometry import (
    LocalPlane,
    clip_cos_alpha,
    cos_alpha_from_gradient,
    fit_local_plane,
    vertical_overlap,
)


@dataclass
class DisuGeometry:
    """Geometric inputs the corrector needs from a DISU model.

    The arrays follow MF6 DISU conventions:

    - ``xc``, ``yc``, ``zc``: per-cell centroid coordinates, shape (n_cells,).
    - ``top``, ``bot``: per-cell top and bottom elevations, shape (n_cells,).
    - ``area``: per-cell horizontal plan area, shape (n_cells,).
    - ``iac``: number of connections per cell row (CSR), shape (n_cells,).
    - ``ja``: flat connection list. The first entry of each row is the
      cell itself (self-reference); subsequent entries are neighbours.
    - ``ihc``: per-connection flag. 0 = vertical, 1 = horizontal,
      2 = horizontal staggered. Same length as ``ja``.
    - ``cl12``: per-connection length from cell centre to shared face.
    - ``hwva``: per-connection horizontal width (ihc != 0) or vertical
      area (ihc == 0).
    """

    xc: np.ndarray
    yc: np.ndarray
    zc: np.ndarray
    top: np.ndarray
    bot: np.ndarray
    area: np.ndarray
    iac: np.ndarray
    ja: np.ndarray
    ihc: np.ndarray
    cl12: np.ndarray
    hwva: np.ndarray

    def __post_init__(self) -> None:
        n = self.xc.size
        for name in ("yc", "zc", "top", "bot", "area", "iac"):
            arr = getattr(self, name)
            if arr.size != n:
                raise ValueError(f"{name} length {arr.size} != n_cells {n}")
        nja = self.ja.size
        for name in ("ihc", "cl12", "hwva"):
            arr = getattr(self, name)
            if arr.size != nja:
                raise ValueError(f"{name} length {arr.size} != nja {nja}")
        if int(self.iac.sum()) != nja:
            raise ValueError(
                f"sum(iac)={int(self.iac.sum())} disagrees with len(ja)={nja}"
            )


@dataclass
class CorrectionResult:
    """Outputs of the slope-aware correction pass."""

    cl12: np.ndarray
    hwva: np.ndarray
    # Per-connection diagnostics, indexed the same way as ja/ihc/cl12/hwva.
    cos_alpha: np.ndarray
    is_pinched: np.ndarray  # True where horizontal overlap <= 0
    columns: list[list[int]] = field(default_factory=list)


def identify_columns(xc: np.ndarray, yc: np.ndarray,
                     tol: float = 1.0e-6) -> list[list[int]]:
    """Group cells into vertical columns by clustering (xc, yc) centroids.

    Parameters
    ----------
    xc, yc : (n_cells,) arrays
        Horizontal cell centroids.
    tol : float
        Maximum horizontal distance for two cells to be considered in the
        same column. Default 1e-6 m (effectively "exact match"), which is
        appropriate for layered Voronoi grids extruded from a 2-D mesh.

    Returns
    -------
    list of lists of cell indices
        One inner list per column. Indices are not sorted by elevation
        within a column; callers should sort if needed.
    """
    n = xc.size
    visited = np.zeros(n, dtype=bool)
    columns: list[list[int]] = []
    tol2 = tol * tol
    for i in range(n):
        if visited[i]:
            continue
        dx = xc - xc[i]
        dy = yc - yc[i]
        same = (dx * dx + dy * dy) <= tol2
        idx = np.flatnonzero(same)
        for j in idx:
            visited[j] = True
        columns.append(idx.tolist())
    return columns


def _sorted_column(zc: np.ndarray, column: list[int]) -> list[int]:
    """Sort a column's cell indices by descending centroid elevation."""
    return sorted(column, key=lambda idx: zc[idx], reverse=True)


def _knn_columns(col_xy: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k nearest columns (including the target) to a query point."""
    diff = col_xy - target
    dist2 = diff[:, 0] ** 2 + diff[:, 1] ** 2
    k = min(k, col_xy.shape[0])
    return np.argpartition(dist2, k - 1)[:k]


def apply_correction(geom: DisuGeometry,
                     neighbours: int = 8,
                     max_dip_deg: float = 85.0,
                     pinchout_tol_m: float = 0.0,
                     column_tol_m: float = 1.0e-6) -> CorrectionResult:
    """Apply slope-aware geometric corrections to a DISU connection set.

    Parameters
    ----------
    geom : DisuGeometry
        Input geometry and connectivity.
    neighbours : int
        Number of nearest columns (including the target) used to fit the
        local interface plane. Default 8.
    max_dip_deg : float
        Maximum permitted local dip. cos(alpha) is clipped so that alpha
        cannot exceed this value. Default 85 degrees.
    pinchout_tol_m : float
        Horizontal-connection overlap below this threshold is treated as a
        pinch-out (zero conductance). Default 0.0 m.
    column_tol_m : float
        Horizontal tolerance for grouping cells into the same column.
        Default 1e-6 m (essentially exact centroid match).

    Returns
    -------
    CorrectionResult
        Corrected ``cl12`` and ``hwva`` arrays, plus per-connection
        diagnostics.
    """
    cl12 = geom.cl12.astype(float, copy=True)
    hwva = geom.hwva.astype(float, copy=True)
    cos_alpha_out = np.ones(geom.ja.size, dtype=float)
    is_pinched = np.zeros(geom.ja.size, dtype=bool)

    # --- Identify columns and per-column ordering. ---
    columns = identify_columns(geom.xc, geom.yc, tol=column_tol_m)
    if not columns:
        raise RuntimeError("no cells found")
    col_xy = np.array(
        [(geom.xc[col[0]], geom.yc[col[0]]) for col in columns],
        dtype=float,
    )

    # Map from cell index -> (column_id, position in sorted column).
    cell_to_colpos: dict[int, tuple[int, int]] = {}
    for col_id, col in enumerate(columns):
        order = _sorted_column(geom.zc, col)
        for pos, cell in enumerate(order):
            cell_to_colpos[int(cell)] = (col_id, pos)

    # Pre-compute interface elevations between consecutive cells in each
    # column. interface_z[col_id, upper_pos] = bot of upper cell.
    interface_z: dict[tuple[int, int], float] = {}
    for col_id, col in enumerate(columns):
        order = _sorted_column(geom.zc, col)
        for upper_pos in range(len(order) - 1):
            upper = order[upper_pos]
            interface_z[(col_id, upper_pos)] = float(geom.bot[upper])

    # --- Walk every CSR row and correct its connections. ---
    # MF6 and MFUSG both place the self-reference at position 0 of each
    # cell's row, so we skip k == 0 unconditionally.
    nja_offset = 0
    for i in range(geom.iac.size):
        n_conn = int(geom.iac[i])
        for k in range(n_conn):
            ja_idx = nja_offset + k
            if k == 0:
                continue  # self-reference
            j = int(geom.ja[ja_idx])
            j0 = j - 1  # 1-based to 0-based
            if not (0 <= j0 < geom.iac.size):
                continue
            ihc = int(geom.ihc[ja_idx])
            if ihc == 0:
                _correct_vertical(
                    geom, i, j0, ja_idx, cl12, hwva, cos_alpha_out,
                    columns, col_xy, cell_to_colpos, interface_z,
                    neighbours, max_dip_deg,
                )
            else:
                _correct_horizontal(
                    geom, i, j0, ja_idx, hwva, is_pinched, pinchout_tol_m,
                )
        nja_offset += n_conn

    return CorrectionResult(
        cl12=cl12,
        hwva=hwva,
        cos_alpha=cos_alpha_out,
        is_pinched=is_pinched,
        columns=columns,
    )


def _correct_vertical(geom: DisuGeometry, i: int, j: int, ja_idx: int,
                       cl12: np.ndarray, hwva: np.ndarray,
                       cos_alpha_out: np.ndarray,
                       columns: list[list[int]],
                       col_xy: np.ndarray,
                       cell_to_colpos: dict[int, tuple[int, int]],
                       interface_z: dict[tuple[int, int], float],
                       neighbours: int,
                       max_dip_deg: float) -> None:
    """Correct one vertical (ihc=0) connection in-place."""
    col_i, pos_i = cell_to_colpos.get(i, (None, None))
    col_j, pos_j = cell_to_colpos.get(j, (None, None))
    if col_i is None or col_j is None or col_i != col_j:
        return  # not a stacked pair in the same column

    upper_pos = min(pos_i, pos_j)
    key = (col_i, upper_pos)
    if key not in interface_z:
        return

    target = col_xy[col_i]
    nbr_cols = _knn_columns(col_xy, target, neighbours)
    samples_xy = []
    samples_z = []
    for c in nbr_cols:
        if (c, upper_pos) in interface_z:
            samples_xy.append(col_xy[c])
            samples_z.append(interface_z[(c, upper_pos)])
    if len(samples_xy) < 3:
        return  # not enough samples to fit a plane; leave connection alone

    plane = fit_local_plane(np.asarray(samples_xy), np.asarray(samples_z))
    cos_a = clip_cos_alpha(plane.cos_alpha, max_dip_deg=max_dip_deg)
    sec_a = 1.0 / cos_a

    hwva[ja_idx] = hwva[ja_idx] * sec_a
    cl12[ja_idx] = cl12[ja_idx] * cos_a
    cos_alpha_out[ja_idx] = cos_a


def _correct_horizontal(geom: DisuGeometry, i: int, j: int, ja_idx: int,
                         hwva: np.ndarray, is_pinched: np.ndarray,
                         pinchout_tol_m: float) -> None:
    """Correct one horizontal (ihc != 0) connection's face height (Eq. 9)."""
    overlap = vertical_overlap(
        float(geom.top[i]), float(geom.bot[i]),
        float(geom.top[j]), float(geom.bot[j]),
    )
    if overlap <= pinchout_tol_m:
        hwva[ja_idx] = 0.0
        is_pinched[ja_idx] = True
        return

    # Original HWVA for a horizontal connection encodes face width times
    # the *vertical* face height inherited from cell tops/bots in the
    # uncorrected model. To recover the face width, we divide by the
    # uncorrected face height. We use the average cell thickness as the
    # uncorrected height, which matches how plan-view preprocessors build
    # this term in practice. The corrected HWVA is then face_width * overlap.
    avg_height = 0.5 * (
        (float(geom.top[i]) - float(geom.bot[i]))
        + (float(geom.top[j]) - float(geom.bot[j]))
    )
    if avg_height <= 0.0:
        hwva[ja_idx] = 0.0
        is_pinched[ja_idx] = True
        return
    face_width = hwva[ja_idx] / avg_height
    hwva[ja_idx] = face_width * overlap
