# disu-slope-corrector

Slope-aware geometric correction for layered DISU groundwater models.

This is the reference implementation accompanying:

> Shokri, A. (2026). *Slope correction for dipping layers in unstructured
> groundwater models.* Hydrogeology Journal (in review).

The scripts and results that reproduce the paper's benchmarks and figures are
in [`paper/`](paper/README.md).

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
7. Scale the vertical conductance by `sec^2(alpha)` by multiplying the
   vertical face area (HWVA in MODFLOW 6, FAHL in MODFLOW-USG) by
   `sec^2(alpha)`. CL12 and the cell TOP/BOT elevations are left unchanged.
   Both **MODFLOW 6** and **MODFLOW-USG** (verified with the LPF package) derive
   vertical conductance between layered cells from the cell TOP/BOT thicknesses and
   the face area and
   **do not use CL12 for vertical connections**, so a length correction
   written to CL12 has no effect in either code.

   The `correct_mf6` / `correct_mfusg` adapters set `code` automatically; both
   codes produce the same arrays. `code="mfusg_cl12"` reproduces the earlier
   split (area on HWVA, length on CL12) for comparison only.

The correction assumes a layered grid in which all layers share one plan-view
tessellation, so that the cells of a column are stacked. `apply_correction` warns
if a vertical connection joins cells that are not in the same column; such
connections are left uncorrected.

For each horizontal connection between cells `i` and `j` in the same
layer, replace the face height with the true elevation overlap
`min(top_i, top_j) - max(bot_i, bot_j)`. Where the cells do not overlap at
all, the right answer depends on why:

- **A pinch-out**, where one of the cells has wedged out to zero thickness.
  The connection carries no flow: HWVA is set to zero and the connection is
  flagged in `is_pinched`.
- **A steep dip**, where both cells have real thickness but the dip has
  carried them past each other. The unit is continuous and still has to
  conduct along it, so the connection keeps the face height the preprocessor
  supplied and is flagged in `is_offset` instead. Zeroing it here would sever
  flow down the dip.

Pass `pinchout_mode="overlap"` (or `--pinchout-mode overlap`) to restore the
original rule, which treats every non-overlapping pair as a pinch-out.

The local plane fit does not require samples in both horizontal directions.
In a one-row cross-section model every column shares the same `y`; the dip
along the section is still fully determined, and the across-strike gradient
is returned as zero.

## Tests

```bash
pip install -e ".[test]"
pytest
```

The test suite covers:

- Plane fitting recovers gradient components to numerical tolerance.
- Flat-layer case (`alpha = 0`) returns inputs unchanged.
- Uniform-dip case: the plane fit recovers `cos(alpha)` exactly and HWVA is
  scaled by `sec^2(alpha)` with CL12 unchanged, for both `code="mfusg"` and
  `code="mf6"`; `code="mfusg_cl12"` gives the earlier `sec`/`cos` split.
- Pinch-out case produces zero conductance where a cell has wedged out, while
  cells merely carried past each other by a steep dip keep their connection.
- Collinear plane fits: a one-row cross-section, where every column shares the
  same `y`, is corrected along the section instead of being rejected.
- **MF6 end-to-end flux** (`test_mf6_flux.py`, skipped if no mf6 executable):
  running the corrected geometry through MODFLOW 6 scales the *simulated*
  vertical flux by `sec^2(alpha)`, and scaling CL12 alone leaves the flux
  unchanged — confirming the correction must ride on HWVA for MF6.

## License

MIT. See `LICENSE`.
