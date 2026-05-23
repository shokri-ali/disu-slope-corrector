# disu-slope-corrector

Slope-aware geometric correction for layered DISU groundwater models.

This is the reference implementation accompanying:

> Shokri, A. (2026). *Slope-Aware Connection Geometry for Layered
> Unstructured Groundwater Models.*

The tool reads an existing DISU package (MODFLOW 6 or MODFLOW-USG),
identifies vertical connections crossing dipping interfaces, and replaces
the plan-view face areas and vertical thicknesses with their slope-aware
equivalents: inclined face area `A_p sec(alpha)` and bedding-normal
thickness `t cos(alpha)`. Horizontal connections receive an Eq. 9 face-height
correction. Connectivity, layer count, and physical cell elevations are
preserved; only the geometric quantities supplied to the conductance
calculation are modified.

XT3D in MODFLOW 6 is complementary to this correction, not a substitute:
XT3D corrects how flux is computed from the supplied geometry, while this
tool corrects the supplied geometry itself. For MODFLOW-USG models, XT3D
is unavailable, so this geometric correction is the only practical remedy
for dipping-interface conductance bias.

## Installation

```bash
pip install -e .
```

Requires Python >= 3.9, numpy, and flopy.

## Quickstart

### MODFLOW 6

```bash
disu-slope-correct \
    --target mf6 \
    --input  path/to/sim_ws \
    --output path/to/sim_ws_corrected
```

### MODFLOW-USG

```bash
disu-slope-correct \
    --target mfusg \
    --input     path/to/model_ws \
    --output    path/to/model_ws_corrected \
    --nam-file  model.nam \
    --centroids centroids.csv
```

The MFUSG path requires a CSV with per-cell `(xc, yc)` centroids in node
order, because MFUSG DISU does not expose horizontal centroids through
FloPy in a uniform way.

### Python API

```python
from disu_slope_corrector import apply_correction, DisuGeometry
from disu_slope_corrector.mf6 import correct_mf6

result = correct_mf6(
    sim_ws="path/to/sim_ws",
    output_ws="path/to/sim_ws_corrected",
    neighbours=8,
    max_dip_deg=85.0,
)
print(f"{(result.cos_alpha < 1.0).sum()} vertical connections corrected")
```

## Algorithm

For each vertical inter-layer connection between upper cell `i` and lower
cell `j`:

1. Group all cells into vertical columns by clustering centroids
   `(xc, yc)` within a small tolerance.
2. Within each column, sort cells by elevation; adjacent pairs in the
   sorted list are stacked.
3. For each stacked pair, take the interface elevation
   `z_int = bot_i = top_j` at the column's `(xc, yc)`.
4. Find the `k` nearest columns and collect their interface elevations at
   the same layer position.
5. Fit a least-squares plane `z = a + b x + c y` to the `(x, y, z_int)`
   samples; compute `cos(alpha) = 1 / sqrt(1 + b^2 + c^2)`.
6. Clip to `max_dip_deg` as a numerical safeguard.
7. Multiply face area (HWVA) by `sec(alpha)`.
8. Multiply connection length (CL12) by `cos(alpha)`.

For each horizontal connection between cells `i` and `j` in the same
layer, replace the face height with the true elevation overlap
`min(top_i, top_j) - max(bot_i, bot_j)`. If the overlap is non-positive,
the connection is marked as a pinch-out (zero conductance).

## Tests

```bash
pip install -e ".[test]"
pytest
```

The test suite covers:

- Plane fitting recovers gradient components to numerical tolerance.
- Flat-layer case (`alpha = 0`) returns inputs unchanged.
- Uniform-dip case produces exactly `sec(alpha)` and `cos(alpha)` ratios
  for face area and connection length.
- Pinch-out case produces zero conductance for non-overlapping cells.
- MF6 / MFUSG parity: the same synthetic geometry routed through both
  adapters produces identical corrected arrays.

## License

MIT. See `LICENSE`.
