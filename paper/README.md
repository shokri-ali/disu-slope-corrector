# Reproducing the paper

Scripts and results for

> Shokri, A. (2026). *Slope correction for dipping layers in unstructured
> groundwater models.* Hydrogeology Journal (in review).

Every script writes its results next to itself; the CSV files here are the
results reported in the paper, so the figures can be redrawn without re-running
the models.

## Requirements

- Python 3.9 or later with the package installed from the repository root
  (`pip install -e .`), plus `pandas`, `scipy`, `shapely`, `matplotlib` and
  `pillow`; the finite-element references also need `scikit-fem`.
- MODFLOW 6 (version 6.5.0 was used) and, for the MODFLOW-USG runs of
  Benchmark A, MODFLOW-USG (`USGs_1`).
- For the enhanced-connectivity comparison and the Benchmark A grids: the data
  release of Provost et al. (2025), doi:10.5066/P13BNARA, unzipped so that its
  `ancillary` and `bin` folders are inside the folder named by
  `PROVOST_ARCHIVE`.

## Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `MF6_EXE` | all MODFLOW 6 runs | MODFLOW 6 executable (default `mf6` on the path, or `bin/mf6.exe` of the Provost archive) |
| `MFUSG_EXE` | `benchmark_a/benchmarkA_25d.py` | MODFLOW-USG executable (default `USGs_1`) |
| `PROVOST_ARCHIVE` | `benchmark_a/`, `enhanced_connectivity/` | unzipped Provost et al. (2025) data release (default `enhanced_connectivity/archive`) |
| `A5_RUNS` | `benchmark_a/`, `benchmark_b/` | folder for model run files (default: system temporary folder) |
| `BENCHMARK_D_TOPO`, `BENCHMARK_D_WINDOW` | `benchmark_d/` | land-surface elevation CSV (`easting,northing,L1_top`) and window `centre_x,centre_y,half_width` |

## Contents

| Folder | Paper | Scripts → results |
|---|---|---|
| `benchmark_a/` | Benchmark A (Sections 2.2, 3.1–3.3, Table 1), Fig. 14 | `benchmarkA_25d.py` → `results_benchmarkA_25d.csv` (250-cell layered Voronoi grid, `grid_voronoi_250.json`); `resolution_study.py` → `results_resolution_{1000,4000,16000}.csv`; `provost_sublayers.py` → `results_provost_sublayers_*.csv`; `provost_aspect_check.py` → Table S1 |
| `benchmark_b/` | Benchmark B (Sections 2.6, 3.4) | `benchmark_b_disu.py` → `results_benchmarkB.csv`, `profiles_benchmarkB.csv` |
| `fem_reference/` | Finite-element references for Benchmarks A–C; Benchmark C DISU runs | `benchmark_a_fem.py`, `benchmark_b_fem.py`, `benchmark_c_confined_fem.py`, `benchmark_c_unconfined_fem3d.py` (and the earlier `_fem.py`, `_fem2d.py` variants); `benchmark_c_confined_disu.py`, `benchmark_c_unconfined_disu.py` → `*_disu_vs_fem.csv` |
| `benchmark_d/` | Benchmark D (real catchment) | `catchment_solve_leaky.py` → `catchment_baseflow.csv`, `catchment_leakage_vs_dip.csv`; figure helpers |
| `enhanced_connectivity/` | ESM section S1 | `step2_benchmarkA_xsection.py` → `results_step2*.csv` (Fig. S2); `provost_benchmark.py` → `results_provost_benchmark.csv` (Fig. S1) |
| `figures/` | All figures | `make_journal_figures.py` (Figs 6–10, 14, S1, S2) and `make_journal_figures_originals.py` (Figs 1–5, 11–13, 15, S3); output in `figures/journal/` |

## Benchmark D data

The land-surface elevations of the Benchmark D catchment come from a model of a
real site and are not distributed; the scripts are provided so that the method
can be followed and applied to other elevation data. The results they produced
(`catchment_baseflow.csv`, `catchment_leakage_vs_dip.csv`) are included, and the
elevation data are available from the author on reasonable request.

## Notes

- Benchmark A uses layered grids in which all layers share one plan-view
  tessellation, the grids the package is designed for.
- Figs 1 and 4 are drawings exported from PowerPoint (`figures/slides/`).
