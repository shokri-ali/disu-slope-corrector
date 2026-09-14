"""Does the accuracy of enhanced connectivity (Provost et al. 2025) + XT3D depend on the
ratio of horizontal cell size to layer thickness?

(A) Structured cross-section of Benchmark A (step2 set-up, analytical lateral heads,
    one grid layer per unit), with dx = 1, 2 and 4 times the layer thickness.
(B) The 2.5-D Voronoi Benchmark A grid refined from 250 to 1000 cells per layer.
Uses the published utilities and MODFLOW 6.5.0 unchanged, as the other comparison runs.
"""
import importlib.util
import math
import os
import pathlib as pl
import sys

import numpy as np
import pandas as pd

HERE = pl.Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = pl.Path(os.environ.get("A5_RUNS", pl.Path(__import__("tempfile").gettempdir()) / "a25_runs")) / "aspect"


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


def cross_section():
    st = load("step2", PAPER / "enhanced_connectivity" / "step2_benchmarkA_xsection.py")
    rows = []
    for ratio in (1, 2, 4):
        def build_grid(nsub, dip, _r=ratio):
            dz = st.T_GEO / nsub
            dx = dz * _r
            ncol = int(round(st.LX * _r / dx))          # keep ~21 columns
            nlay = st.N_GEO * nsub
            tan_a = math.tan(math.radians(dip))
            xc = (np.arange(ncol) + 0.5) * dx
            zbase = xc * tan_a
            top = (zbase + st.N_GEO * st.T_GEO).reshape(1, ncol)
            botm = np.array([zbase + st.N_GEO * st.T_GEO - (k + 1) * dz for k in range(nlay)]).reshape(nlay, 1, ncol)
            return dx, dz, ncol, nlay, xc, top, botm
        st.build_grid = build_grid
        for dip in (15, 30, 45, 60):
            for var in ("provost_xt3d", "corrected"):
                r = st.build_and_run(("analytical", 1, float(dip), var, [1.0, 1.0, 1.0],
                                      str(ROOT / f"xs_r{ratio}"), 1e-11))
                r["dx_over_dz"] = ratio
                rows.append(r)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                       if k in ("dx_over_dz", "dip", "variant", "status", "rms_all_mm", "rms_mid_mm", "drop_ratio")}, flush=True)
    return rows


def voronoi_fine():
    ps = load("provost_sub", HERE / "provost_sublayers.py")
    b = ps.b
    b.NCELL = 1000
    b.GRID_JSON = HERE / "grid_voronoi_1000.json"
    rows = []
    for dip in (15, 30, 45, 60):
        r = ps.run((dip, 1, str(ROOT / "vor1000")))
        r["ncell"] = 1000
        rows.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
               if k in ("ncell", "dip", "status", "rms_mm", "drop_ratio")}, flush=True)
    return rows


if __name__ == "__main__":
    xs = cross_section()
    pd.DataFrame(xs).to_csv(HERE / "provost_aspect_cross_section.csv", index=False, float_format="%.6g")
    vf = voronoi_fine()
    pd.DataFrame(vf).to_csv(HERE / "provost_voronoi_1000.csv", index=False, float_format="%.6g")
