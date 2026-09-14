"""Horizontal connections with no vertical overlap.

Zero overlap has two quite different causes and they need different answers:
a unit that has wedged out carries no flow, while a unit that dips steeply
enough to carry neighbouring cells past each other is still continuous and
must keep conducting along the dip.
"""

from __future__ import annotations

import numpy as np
import pytest

from disu_slope_corrector.correction import DisuGeometry, apply_correction


def _two_cell_geometry(top: np.ndarray, bot: np.ndarray,
                       hwva_horizontal: float) -> DisuGeometry:
    """Two cells side by side, joined by one horizontal connection each way."""
    return DisuGeometry(
        xc=np.array([0.0, 10.0]),
        yc=np.array([0.0, 0.0]),
        zc=0.5 * (top + bot),
        top=top, bot=bot,
        area=np.array([100.0, 100.0]),
        iac=np.array([2, 2]),
        ja=np.array([1, 2, 2, 1]),            # 1-based
        ihc=np.array([0, 1, 0, 1]),           # self, horizontal, self, horizontal
        cl12=np.array([0.0, 5.0, 0.0, 5.0]),
        hwva=np.array([0.0, hwva_horizontal, 0.0, hwva_horizontal]),
    )


HORIZONTAL = (1, 3)  # positions of the two horizontal entries in ja


def test_true_pinchout_zeros_hwva():
    """A unit wedged out to zero thickness carries no flow."""
    # Cell 1 has collapsed: top == bot.
    geom = _two_cell_geometry(
        top=np.array([10.0, 4.0]), bot=np.array([5.0, 4.0]), hwva_horizontal=45.0)
    result = apply_correction(geom)

    for idx in HORIZONTAL:
        assert result.hwva[idx] == 0.0
        assert bool(result.is_pinched[idx]) is True
        assert bool(result.is_offset[idx]) is False


def test_steeply_offset_cells_keep_their_connection():
    """Cells carried past each other by a steep dip stay connected.

    Both cells have real thickness, so the unit is continuous and zeroing the
    face would sever flow down the dip. The face is left as supplied and the
    connection is flagged instead.
    """
    geom = _two_cell_geometry(
        top=np.array([10.0, 4.0]), bot=np.array([5.0, 0.0]), hwva_horizontal=45.0)
    result = apply_correction(geom)

    for idx in HORIZONTAL:
        assert result.hwva[idx] == pytest.approx(45.0)
        assert bool(result.is_offset[idx]) is True
        assert bool(result.is_pinched[idx]) is False


def test_overlap_mode_restores_the_published_rule():
    """pinchout_mode='overlap' reproduces Algorithm 1 as published."""
    geom = _two_cell_geometry(
        top=np.array([10.0, 4.0]), bot=np.array([5.0, 0.0]), hwva_horizontal=45.0)
    result = apply_correction(geom, pinchout_mode="overlap")

    for idx in HORIZONTAL:
        assert result.hwva[idx] == 0.0
        assert bool(result.is_pinched[idx]) is True
        assert bool(result.is_offset[idx]) is False


def test_partial_overlap_replaces_face_height():
    """Partial vertical overlap rescales HWVA by overlap / avg_height."""
    # face_width = 10 m, avg_height = 5 m, so HWVA = 50 m^2.
    geom = _two_cell_geometry(
        top=np.array([10.0, 8.0]), bot=np.array([5.0, 3.0]), hwva_horizontal=50.0)
    result = apply_correction(geom)

    # overlap = min(10, 8) - max(5, 3) = 3 m, face_width = 10 -> HWVA = 30.
    for idx in HORIZONTAL:
        assert result.hwva[idx] == pytest.approx(30.0, abs=1e-10)
        assert bool(result.is_pinched[idx]) is False
        assert bool(result.is_offset[idx]) is False
