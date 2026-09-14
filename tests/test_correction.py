"""Integration tests for apply_correction on synthetic DISU geometries."""

from __future__ import annotations

import math

import numpy as np
import pytest

from disu_slope_corrector.correction import (
    DisuGeometry,
    apply_correction,
    identify_columns,
)


# ---------------------------------------------------------------------------
# Helpers to build small synthetic layered DISU geometries.
# ---------------------------------------------------------------------------


def _build_layered_geometry(
    columns_xy: np.ndarray,
    layer_thickness: float,
    n_layers: int,
    dip_deg: float = 0.0,
    cell_area: float = 100.0,
) -> DisuGeometry:
    """Build a tiny layered DISU geometry with optional uniform dip in +x.

    No horizontal connections are populated (this fixture is for vertical-
    connection tests). Each cell has a self-reference plus its vertical
    neighbour(s).
    """
    n_cols = columns_xy.shape[0]
    n_cells = n_cols * n_layers
    xc = np.zeros(n_cells)
    yc = np.zeros(n_cells)
    zc = np.zeros(n_cells)
    top = np.zeros(n_cells)
    bot = np.zeros(n_cells)
    area = np.full(n_cells, cell_area)

    slope = math.tan(math.radians(dip_deg))

    # Cell numbering: layer-major. Cell index = col * n_layers + layer.
    # Layer 0 = topmost.
    for c in range(n_cols):
        x, y = columns_xy[c]
        # Reference elevation at this column: top of layer 0 dips with x.
        z_top0 = -slope * x
        for l in range(n_layers):
            idx = c * n_layers + l
            xc[idx] = x
            yc[idx] = y
            top[idx] = z_top0 - l * layer_thickness
            bot[idx] = z_top0 - (l + 1) * layer_thickness
            zc[idx] = 0.5 * (top[idx] + bot[idx])

    # Connectivity: for each cell, self + (up if exists) + (down if exists).
    iac_list = []
    ja_list = []
    ihc_list = []
    cl12_list = []
    hwva_list = []

    half_thick = 0.5 * layer_thickness  # uncorrected vertical half-thickness

    for c in range(n_cols):
        for l in range(n_layers):
            idx = c * n_layers + l
            row = [idx + 1]  # self, 1-based
            row_ihc = [0]
            row_cl = [0.0]
            row_hwva = [0.0]
            if l > 0:  # connection to layer above
                up = c * n_layers + (l - 1)
                row.append(up + 1)
                row_ihc.append(0)
                row_cl.append(half_thick)  # vertical half-thickness, plan view
                row_hwva.append(cell_area)  # plan-view face area
            if l < n_layers - 1:  # connection to layer below
                down = c * n_layers + (l + 1)
                row.append(down + 1)
                row_ihc.append(0)
                row_cl.append(half_thick)
                row_hwva.append(cell_area)
            iac_list.append(len(row))
            ja_list.extend(row)
            ihc_list.extend(row_ihc)
            cl12_list.extend(row_cl)
            hwva_list.extend(row_hwva)

    return DisuGeometry(
        xc=xc, yc=yc, zc=zc,
        top=top, bot=bot, area=area,
        iac=np.array(iac_list, dtype=int),
        ja=np.array(ja_list, dtype=int),
        ihc=np.array(ihc_list, dtype=int),
        cl12=np.array(cl12_list, dtype=float),
        hwva=np.array(hwva_list, dtype=float),
    )


def _grid_columns(nx: int, ny: int, spacing: float = 10.0) -> np.ndarray:
    pts = []
    for j in range(ny):
        for i in range(nx):
            pts.append((i * spacing, j * spacing))
    return np.array(pts, dtype=float)


# ---------------------------------------------------------------------------
# identify_columns
# ---------------------------------------------------------------------------


def test_identify_columns_groups_stacked_cells():
    """Cells with identical (x, y) but different z belong to the same column."""
    geom = _build_layered_geometry(
        columns_xy=_grid_columns(3, 3),
        layer_thickness=10.0,
        n_layers=4,
        dip_deg=0.0,
    )
    cols = identify_columns(geom.xc, geom.yc)
    assert len(cols) == 9
    for col in cols:
        assert len(col) == 4


# ---------------------------------------------------------------------------
# Flat-layer limit: alpha = 0 implies the correction is a no-op.
# ---------------------------------------------------------------------------


def test_flat_layer_correction_is_identity():
    geom = _build_layered_geometry(
        columns_xy=_grid_columns(4, 4),
        layer_thickness=10.0,
        n_layers=3,
        dip_deg=0.0,
    )
    result = apply_correction(geom, neighbours=6)
    np.testing.assert_allclose(result.cl12, geom.cl12, atol=1e-12)
    np.testing.assert_allclose(result.hwva, geom.hwva, atol=1e-12)
    # Vertical connections all have cos_alpha = 1.
    vert = geom.ihc == 0
    nonself = geom.ja != (np.repeat(np.arange(geom.iac.size), geom.iac) + 1)
    target = vert & nonself
    np.testing.assert_allclose(result.cos_alpha[target], 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Uniform-dip case: exact sec(alpha), cos(alpha) ratios.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dip_deg", [15.0, 30.0, 45.0, 60.0])
def test_uniform_dip_recovers_exact_ratios(dip_deg: float):
    geom = _build_layered_geometry(
        columns_xy=_grid_columns(5, 5, spacing=10.0),
        layer_thickness=10.0,
        n_layers=3,
        dip_deg=dip_deg,
    )
    result = apply_correction(geom, neighbours=8, max_dip_deg=89.0)
    cos_expected = math.cos(math.radians(dip_deg))
    sec_expected = 1.0 / cos_expected

    # Find the vertical (non-self) connections.
    cell_idx = np.repeat(np.arange(geom.iac.size), geom.iac)
    self_ref = geom.ja == (cell_idx + 1)
    vertical = (geom.ihc == 0) & (~self_ref)

    # cos_alpha matches the analytical value to high precision.
    np.testing.assert_allclose(
        result.cos_alpha[vertical], cos_expected, atol=1e-9,
    )
    # Default (code="mfusg") folds sec^2 into HWVA and leaves CL12 unchanged,
    # as for MF6: neither solver uses CL12 for vertical conductance.
    np.testing.assert_allclose(
        result.hwva[vertical] / geom.hwva[vertical],
        sec_expected ** 2,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        result.cl12[vertical], geom.cl12[vertical], atol=1e-12,
    )


@pytest.mark.parametrize("dip_deg", [15.0, 30.0, 45.0, 60.0])
def test_legacy_cl12_split_applies_sec_and_cos(dip_deg: float):
    """code='mfusg_cl12' keeps the earlier split (HWVA x sec, CL12 x cos)."""
    geom = _build_layered_geometry(
        columns_xy=_grid_columns(5, 5, spacing=10.0),
        layer_thickness=10.0,
        n_layers=3,
        dip_deg=dip_deg,
    )
    result = apply_correction(geom, neighbours=8, max_dip_deg=89.0, code="mfusg_cl12")
    cell_idx = np.repeat(np.arange(geom.iac.size), geom.iac)
    vertical = (geom.ihc == 0) & (geom.ja != cell_idx + 1)
    cos_expected = math.cos(math.radians(dip_deg))
    np.testing.assert_allclose(
        result.hwva[vertical] / geom.hwva[vertical], 1.0 / cos_expected, atol=1e-9)
    np.testing.assert_allclose(
        result.cl12[vertical] / geom.cl12[vertical], cos_expected, atol=1e-9)


@pytest.mark.parametrize("dip_deg", [15.0, 30.0, 45.0, 60.0])
def test_mf6_mode_folds_sec_squared_into_hwva(dip_deg: float):
    """In code='mf6' the length factor rides on HWVA (CL12 unchanged), because
    MF6 ignores CL12 for vertical conductance."""
    geom = _build_layered_geometry(
        columns_xy=_grid_columns(5, 5, spacing=10.0),
        layer_thickness=10.0,
        n_layers=3,
        dip_deg=dip_deg,
    )
    result = apply_correction(geom, neighbours=8, max_dip_deg=89.0, code="mf6")
    cell_idx = np.repeat(np.arange(geom.iac.size), geom.iac)
    self_ref = geom.ja == (cell_idx + 1)
    vertical = (geom.ihc == 0) & (~self_ref)
    sec2 = 1.0 / math.cos(math.radians(dip_deg)) ** 2
    # HWVA carries sec^2; CL12 is left unchanged.
    np.testing.assert_allclose(
        result.hwva[vertical] / geom.hwva[vertical], sec2, atol=1e-9)
    np.testing.assert_allclose(
        result.cl12[vertical], geom.cl12[vertical], atol=1e-12)


def test_invalid_code_raises():
    geom = _build_layered_geometry(
        columns_xy=_grid_columns(4, 4), layer_thickness=10.0, n_layers=2)
    with pytest.raises(ValueError):
        apply_correction(geom, code="modflow7")


def test_combined_correction_ratio_is_sec_squared():
    """The vertical conductance ratio plan/corrected is exactly cos^2(alpha) (Eq. 6)."""
    for dip_deg in (15.0, 30.0, 45.0, 60.0):
        geom = _build_layered_geometry(
            columns_xy=_grid_columns(5, 5, spacing=10.0),
            layer_thickness=10.0,
            n_layers=3,
            dip_deg=dip_deg,
        )
        result = apply_correction(geom, neighbours=8, max_dip_deg=89.0)
        cell_idx = np.repeat(np.arange(geom.iac.size), geom.iac)
        self_ref = geom.ja == (cell_idx + 1)
        vertical = (geom.ihc == 0) & (~self_ref)

        plan_C = geom.hwva[vertical] / geom.cl12[vertical]
        slope_C = result.hwva[vertical] / result.cl12[vertical]
        cos_a = math.cos(math.radians(dip_deg))
        np.testing.assert_allclose(
            slope_C / plan_C, 1.0 / (cos_a * cos_a), atol=1e-8,
        )


# ---------------------------------------------------------------------------
# Cross-section models: one row of columns, so no variation in y.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dip_deg", [15.0, 30.0, 45.0, 60.0])
def test_single_row_cross_section_is_corrected(dip_deg: float):
    """A one-row cross-section is corrected like any other model.

    Every column shares the same y, so the local plane fit is degenerate. The
    dip along the section is still fully determined and the correction must go
    through rather than fail.
    """
    geom = _build_layered_geometry(
        columns_xy=_grid_columns(6, 1, spacing=10.0),
        layer_thickness=10.0,
        n_layers=3,
        dip_deg=dip_deg,
    )
    result = apply_correction(geom, neighbours=8, max_dip_deg=89.0, code="mf6")

    cell_idx = np.repeat(np.arange(geom.iac.size), geom.iac)
    self_ref = geom.ja == (cell_idx + 1)
    vertical = (geom.ihc == 0) & (~self_ref)

    cos_expected = math.cos(math.radians(dip_deg))
    np.testing.assert_allclose(
        result.cos_alpha[vertical], cos_expected, atol=1e-9)
    np.testing.assert_allclose(
        result.hwva[vertical] / geom.hwva[vertical],
        1.0 / (cos_expected * cos_expected),
        atol=1e-9,
    )


def test_unstacked_vertical_connections_warn():
    """Vertical connections between cells that are not in the same column are left
    uncorrected and reported with a warning."""
    geom = _build_layered_geometry(
        columns_xy=_grid_columns(4, 4), layer_thickness=10.0, n_layers=2, dip_deg=20.0)
    lower = np.arange(geom.xc.size) % 2 == 1          # layer 1 cells (layer-major numbering)
    geom.xc[lower] += 3.0                              # shift the lower layer in plan view
    with pytest.warns(UserWarning, match="not stacked"):
        result = apply_correction(geom, neighbours=8, max_dip_deg=89.0, code="mf6")
    np.testing.assert_allclose(result.hwva, geom.hwva)


def test_invalid_pinchout_mode_raises():
    geom = _build_layered_geometry(
        columns_xy=_grid_columns(4, 4), layer_thickness=10.0, n_layers=2)
    with pytest.raises(ValueError, match="pinchout_mode"):
        apply_correction(geom, pinchout_mode="zero")
