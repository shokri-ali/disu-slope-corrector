"""Unit tests for the geometry primitives."""

from __future__ import annotations

import math

import numpy as np
import pytest

from disu_slope_corrector.geometry import (
    LocalPlane,
    bedding_normal_thickness,
    clip_cos_alpha,
    cos_alpha_from_gradient,
    fit_local_plane,
    slope_aware_face_area,
    vertical_overlap,
)


# ---------------------------------------------------------------------------
# fit_local_plane
# ---------------------------------------------------------------------------


def test_fit_flat_plane():
    """A flat patch (alpha = 0) returns b = c = 0 and cos_alpha = 1."""
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    z = np.array([5.0, 5.0, 5.0, 5.0])
    plane = fit_local_plane(xy, z)
    assert plane.a == pytest.approx(5.0, abs=1e-12)
    assert plane.b == pytest.approx(0.0, abs=1e-12)
    assert plane.c == pytest.approx(0.0, abs=1e-12)
    assert plane.cos_alpha == pytest.approx(1.0, abs=1e-12)
    assert plane.alpha_deg == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("dip_deg", [5.0, 15.0, 30.0, 45.0, 60.0])
def test_fit_uniform_dip_recovers_alpha(dip_deg: float):
    """A uniformly dipping plane in the +x direction returns the correct alpha."""
    slope = math.tan(math.radians(dip_deg))
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0], [5.0, 5.0]])
    z = -slope * xy[:, 0]  # dip toward +x
    plane = fit_local_plane(xy, z)
    assert plane.b == pytest.approx(-slope, abs=1e-10)
    assert plane.c == pytest.approx(0.0, abs=1e-10)
    assert plane.alpha_deg == pytest.approx(dip_deg, abs=1e-8)


def test_fit_too_few_samples_raises():
    xy = np.array([[0.0, 0.0], [1.0, 0.0]])
    z = np.array([0.0, 0.0])
    with pytest.raises(ValueError, match="at least 3"):
        fit_local_plane(xy, z)


def test_fit_collinear_samples_uses_the_sampled_direction():
    """Samples on one line in plan view still determine the dip along it.

    This is the one-row cross-section case: every column shares the same y, so
    the across-strike gradient cannot be known and is returned as zero.
    """
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    z = np.array([0.0, -1.0, -2.0, -3.0])  # dips 45 degrees toward +x
    plane = fit_local_plane(xy, z)
    assert plane.b == pytest.approx(-1.0, abs=1e-12)
    assert plane.c == pytest.approx(0.0, abs=1e-12)
    assert plane.a == pytest.approx(0.0, abs=1e-12)
    assert plane.alpha_deg == pytest.approx(45.0, abs=1e-9)


def test_fit_collinear_along_y():
    """The degenerate fit works along any direction, not only x."""
    xy = np.array([[5.0, 0.0], [5.0, 10.0], [5.0, 20.0]])
    z = np.array([0.0, -10.0, -20.0])
    plane = fit_local_plane(xy, z)
    assert plane.b == pytest.approx(0.0, abs=1e-12)
    assert plane.c == pytest.approx(-1.0, abs=1e-12)
    assert plane.alpha_deg == pytest.approx(45.0, abs=1e-9)


def test_fit_collinear_oblique_splits_gradient_between_axes():
    """Along y = x, the unit gradient splits evenly between b and c."""
    xy = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    z = np.array([0.0, -math.sqrt(2.0), -2.0 * math.sqrt(2.0)])
    plane = fit_local_plane(xy, z)
    half = -1.0 / math.sqrt(2.0)
    assert plane.b == pytest.approx(half, abs=1e-12)
    assert plane.c == pytest.approx(half, abs=1e-12)
    assert plane.alpha_deg == pytest.approx(45.0, abs=1e-9)


def test_fit_identical_points_returns_flat_plane():
    """Samples at a single location carry no dip information at all."""
    xy = np.array([[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]])
    z = np.array([7.0, 7.0, 7.0])
    plane = fit_local_plane(xy, z)
    assert plane.b == pytest.approx(0.0, abs=1e-12)
    assert plane.c == pytest.approx(0.0, abs=1e-12)
    assert plane.cos_alpha == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# cos_alpha_from_gradient
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dip_deg", [0.0, 15.0, 30.0, 45.0, 60.0, 85.0])
def test_cos_alpha_from_gradient_matches_definition(dip_deg: float):
    slope = math.tan(math.radians(dip_deg))
    cos_a = cos_alpha_from_gradient(slope, 0.0)
    assert cos_a == pytest.approx(math.cos(math.radians(dip_deg)), abs=1e-12)


# ---------------------------------------------------------------------------
# slope_aware_face_area and bedding_normal_thickness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dip_deg", [0.0, 30.0, 45.0, 60.0])
def test_face_area_factor_is_sec_alpha(dip_deg: float):
    cos_a = math.cos(math.radians(dip_deg))
    A_p = 100.0
    A_s = slope_aware_face_area(A_p, cos_a)
    assert A_s == pytest.approx(A_p / cos_a, abs=1e-12)


@pytest.mark.parametrize("dip_deg", [0.0, 30.0, 45.0, 60.0])
def test_thickness_factor_is_cos_alpha(dip_deg: float):
    cos_a = math.cos(math.radians(dip_deg))
    t = 50.0
    t_n = bedding_normal_thickness(t, cos_a)
    assert t_n == pytest.approx(t * cos_a, abs=1e-12)


def test_face_area_rejects_nonpositive_cos_alpha():
    with pytest.raises(ValueError):
        slope_aware_face_area(100.0, 0.0)
    with pytest.raises(ValueError):
        slope_aware_face_area(100.0, -0.1)


def test_combined_conductance_ratio_is_sec_squared():
    """The combined area/thickness correction is sec**2(alpha) (Eq. 6)."""
    for dip_deg in (0.0, 15.0, 30.0, 45.0, 60.0):
        cos_a = math.cos(math.radians(dip_deg))
        A_p, t = 100.0, 50.0
        plan_cond = A_p / t
        slope_cond = slope_aware_face_area(A_p, cos_a) / bedding_normal_thickness(t, cos_a)
        ratio = slope_cond / plan_cond
        assert ratio == pytest.approx(1.0 / (cos_a * cos_a), abs=1e-12)


# ---------------------------------------------------------------------------
# vertical_overlap (Eq. 9)
# ---------------------------------------------------------------------------


def test_overlap_full_overlap():
    assert vertical_overlap(10.0, 0.0, 10.0, 0.0) == pytest.approx(10.0)


def test_overlap_partial():
    # i: top=10, bot=5; j: top=8, bot=3 -> overlap = min(10,8) - max(5,3) = 3
    assert vertical_overlap(10.0, 5.0, 8.0, 3.0) == pytest.approx(3.0)


def test_overlap_pinchout_returns_zero():
    # i sits entirely above j -> no overlap
    assert vertical_overlap(10.0, 5.0, 4.0, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# clip_cos_alpha
# ---------------------------------------------------------------------------


def test_clip_cos_alpha_below_max_unchanged():
    cos_a = math.cos(math.radians(30.0))
    assert clip_cos_alpha(cos_a, max_dip_deg=85.0) == pytest.approx(cos_a, abs=1e-12)


def test_clip_cos_alpha_above_max_clipped():
    # Try to feed alpha = 89 deg with max 85 deg -> should clip to cos(85).
    cos_in = math.cos(math.radians(89.0))
    cos_out = clip_cos_alpha(cos_in, max_dip_deg=85.0)
    assert cos_out == pytest.approx(math.cos(math.radians(85.0)), abs=1e-12)
