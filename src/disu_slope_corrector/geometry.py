"""Slope-aware geometric primitives for layered DISU groundwater models.

This module implements the geometric quantities derived in Section 2.1 of
Shokri (2026):

    z(x, y) = a + b x + c y                                       (Eq. 2)
    cos(alpha) = 1 / sqrt(1 + b**2 + c**2)                        (Eq. 3)
    A_s = A_p / cos(alpha) = A_p sec(alpha)                       (Eq. 4)
    t_n = t cos(alpha)                                            (Eq. 5)
    h_face = min(top_i, top_j) - max(bot_i, bot_j)                (Eq. 9)

Everything here is pure numpy: no I/O, no FloPy dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocalPlane:
    """Best-fit plane z = a + b x + c y for an interface elevation patch."""

    a: float
    b: float
    c: float

    @property
    def cos_alpha(self) -> float:
        return 1.0 / float(np.sqrt(1.0 + self.b * self.b + self.c * self.c))

    @property
    def sec_alpha(self) -> float:
        return 1.0 / self.cos_alpha

    @property
    def alpha_rad(self) -> float:
        return float(np.arccos(self.cos_alpha))

    @property
    def alpha_deg(self) -> float:
        return float(np.degrees(self.alpha_rad))


def fit_local_plane(xy: np.ndarray, z: np.ndarray) -> LocalPlane:
    """Least-squares fit of z = a + b x + c y to (xy, z) samples.

    Parameters
    ----------
    xy : (n, 2) array_like
        Horizontal coordinates of the interface samples.
    z : (n,) array_like
        Interface elevations at those points.

    Returns
    -------
    LocalPlane
        Best-fit plane coefficients.

    Raises
    ------
    ValueError
        If fewer than 3 samples are supplied, or if the design matrix is
        rank-deficient (collinear samples).
    """
    xy = np.asarray(xy, dtype=float)
    z = np.asarray(z, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must have shape (n, 2); got {xy.shape}")
    if z.shape != (xy.shape[0],):
        raise ValueError(f"z must have shape ({xy.shape[0]},); got {z.shape}")
    if xy.shape[0] < 3:
        raise ValueError(f"need at least 3 samples to fit a plane; got {xy.shape[0]}")

    design = np.column_stack([np.ones(xy.shape[0]), xy[:, 0], xy[:, 1]])
    rank = np.linalg.matrix_rank(design)
    if rank < 3:
        raise ValueError("interface samples are collinear; cannot fit a plane")

    coeffs, *_ = np.linalg.lstsq(design, z, rcond=None)
    return LocalPlane(a=float(coeffs[0]), b=float(coeffs[1]), c=float(coeffs[2]))


def cos_alpha_from_gradient(b: float, c: float) -> float:
    """cos(alpha) given the in-plane gradient components b, c. (Eq. 3)."""
    return 1.0 / float(np.sqrt(1.0 + b * b + c * c))


def slope_aware_face_area(plan_area: float, cos_alpha: float) -> float:
    """Inclined face area A_s = A_p / cos(alpha) (Eq. 4).

    cos_alpha is required to be strictly positive; the caller is responsible
    for clipping to a maximum dip if needed (see ``clip_cos_alpha``).
    """
    if cos_alpha <= 0.0:
        raise ValueError(f"cos_alpha must be > 0, got {cos_alpha}")
    return plan_area / cos_alpha


def bedding_normal_thickness(vertical_thickness: float, cos_alpha: float) -> float:
    """Bedding-normal thickness t_n = t cos(alpha) (Eq. 5)."""
    return vertical_thickness * cos_alpha


def vertical_overlap(top_i: float, bot_i: float,
                     top_j: float, bot_j: float) -> float:
    """True elevation overlap between two cells (Eq. 9).

    Returns 0.0 when the cells do not overlap vertically (pinch-out or
    layer truncation). The caller should treat zero overlap as a
    disconnected connection (zero conductance).
    """
    overlap = min(top_i, top_j) - max(bot_i, bot_j)
    return float(max(overlap, 0.0))


def clip_cos_alpha(cos_alpha: float, max_dip_deg: float = 85.0) -> float:
    """Clip cos(alpha) to avoid numerical issues at near-vertical dips.

    Equivalent to capping the local dip at ``max_dip_deg``. For the
    practical range tested in Shokri (2026) (dips up to 60 degrees) this
    clip is a safeguard, not an active constraint.
    """
    cos_min = float(np.cos(np.radians(max_dip_deg)))
    return max(cos_alpha, cos_min)
