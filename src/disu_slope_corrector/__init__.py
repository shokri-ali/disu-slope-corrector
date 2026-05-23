"""disu_slope_corrector: slope-aware geometric correction for layered DISU models.

Reference: Shokri, A. (2026). Slope-Aware Connection Geometry for Layered
Unstructured Groundwater Models.
"""

from .correction import (
    CorrectionResult,
    DisuGeometry,
    apply_correction,
    identify_columns,
)
from .geometry import (
    LocalPlane,
    bedding_normal_thickness,
    clip_cos_alpha,
    cos_alpha_from_gradient,
    fit_local_plane,
    slope_aware_face_area,
    vertical_overlap,
)

__all__ = [
    "CorrectionResult",
    "DisuGeometry",
    "LocalPlane",
    "apply_correction",
    "bedding_normal_thickness",
    "clip_cos_alpha",
    "cos_alpha_from_gradient",
    "fit_local_plane",
    "identify_columns",
    "slope_aware_face_area",
    "vertical_overlap",
]

__version__ = "0.1.0"
