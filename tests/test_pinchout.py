"""Pinch-out handling: horizontal connections with no vertical overlap."""

from __future__ import annotations

import numpy as np

from disu_slope_corrector.correction import DisuGeometry, apply_correction


def test_pinchout_horizontal_connection_zeros_hwva():
    """When two horizontally-connected cells do not overlap vertically, the
    corrected HWVA should be zero and the pinch-out flag should be set."""
    # Two cells side by side; cell 0 sits entirely above cell 1.
    xc = np.array([0.0, 10.0])
    yc = np.array([0.0, 0.0])
    top = np.array([10.0, 4.0])
    bot = np.array([5.0, 0.0])
    zc = 0.5 * (top + bot)
    area = np.array([100.0, 100.0])

    iac = np.array([2, 2])
    ja = np.array([1, 2, 2, 1])           # 1-based
    ihc = np.array([0, 1, 0, 1])          # self, horizontal, self, horizontal
    cl12 = np.array([0.0, 5.0, 0.0, 5.0])
    # HWVA for horizontal connections is face_width * avg_height in the
    # uncorrected model. Pretend face_width = 10 m, avg_height = 4.5 m.
    hwva = np.array([0.0, 45.0, 0.0, 45.0])

    geom = DisuGeometry(
        xc=xc, yc=yc, zc=zc,
        top=top, bot=bot, area=area,
        iac=iac, ja=ja, ihc=ihc, cl12=cl12, hwva=hwva,
    )
    result = apply_correction(geom)

    # The two horizontal entries are at ja indices 1 and 3.
    horiz_indices = [1, 3]
    for idx in horiz_indices:
        assert result.hwva[idx] == 0.0
        assert bool(result.is_pinched[idx]) is True


def test_partial_overlap_replaces_face_height():
    """Partial vertical overlap should rescale HWVA by overlap / avg_height."""
    xc = np.array([0.0, 10.0])
    yc = np.array([0.0, 0.0])
    top = np.array([10.0, 8.0])
    bot = np.array([5.0, 3.0])
    zc = 0.5 * (top + bot)
    area = np.array([100.0, 100.0])

    iac = np.array([2, 2])
    ja = np.array([1, 2, 2, 1])
    ihc = np.array([0, 1, 0, 1])
    cl12 = np.array([0.0, 5.0, 0.0, 5.0])
    # face_width = 10 m, avg_height = 5 m, so HWVA = 50 m^2.
    hwva = np.array([0.0, 50.0, 0.0, 50.0])

    geom = DisuGeometry(
        xc=xc, yc=yc, zc=zc,
        top=top, bot=bot, area=area,
        iac=iac, ja=ja, ihc=ihc, cl12=cl12, hwva=hwva,
    )
    result = apply_correction(geom)

    # overlap = min(10, 8) - max(5, 3) = 3 m. face_width = 10. New HWVA = 30.
    for idx in (1, 3):
        assert abs(result.hwva[idx] - 30.0) < 1e-10
        assert bool(result.is_pinched[idx]) is False
