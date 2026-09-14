"""Enhanced connectivity (Provost et al. 2025) + XT3D on the 2.5-D Voronoi Benchmark A grid
with one, two and three grid layers per hydrostratigraphic unit (uniform K).

Same grid, boundary conditions and executable as benchmarkA_25d.py. Each 50 m unit is
split into nsub grid layers of equal thickness; constant head on the bottom grid layer at
its analytical value (100 m); analytical head in grid layer k (from the top) is
100 + (nlay - 1 - k) * d / nsub, with d = t R cos^2(alpha) / K.
"""
import math
import multiprocessing as mp
import os
import pathlib as pl
import shutil

import numpy as np

HERE = pl.Path(__file__).resolve().parent
import importlib.util

_s = importlib.util.spec_from_file_location("b25", HERE / "benchmarkA_25d.py")
b = importlib.util.module_from_spec(_s)
_s.loader.exec_module(b)


def run(task):
    import flopy
    from disv2disu import Disv2Disu
    dip, nsub, root = task
    verts, cell2d = b.load_grid()
    ncpl = len(cell2d)
    nlay = b.NLAY * nsub
    dz = b.T / nsub
    xc = np.array([c[1] for c in cell2d])
    top = b.ZTOP0 - xc * math.tan(math.radians(dip))
    botm = np.array([top - (k + 1) * dz for k in range(nlay)])
    gp = Disv2Disu(verts, cell2d, top, botm, True).get_gridprops_disu6()
    ws = pl.Path(root) / "provost_nsub" / b.RUNTAG / f"n{nsub}" / f"dip_{dip:05.2f}"
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    sim = flopy.mf6.MFSimulation(sim_name="a", sim_ws=str(ws), exe_name=str(b.MF6))
    flopy.mf6.ModflowTdis(sim)
    flopy.mf6.ModflowIms(sim, linear_acceleration="bicgstab", outer_maximum=200, inner_maximum=2000,
                         outer_dvclose=1e-10, inner_dvclose=1e-10, rcloserecord=[1e-10, "strict"])
    gwf = flopy.mf6.ModflowGwf(sim, modelname="a")
    flopy.mf6.ModflowGwfdisu(gwf, **gp)
    flopy.mf6.ModflowGwfic(gwf, strt=b.HB)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=1.0, xt3doptions=True, angle1=0.0, angle2=0.0, angle3=0.0)
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[[((nlay - 1) * ncpl + i,), b.HB] for i in range(ncpl)])
    flopy.mf6.ModflowGwfrch(gwf, stress_period_data=[[(i,), b.R] for i in range(ncpl)])
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="a.hds", saverecord=[("HEAD", "ALL")])
    sim.write_simulation(silent=True)
    ok, _ = sim.run_simulation(silent=True)
    row = dict(code="mf6", case="homog", dip=dip, variant=f"provost_n{nsub}", xt3d=True, nsub=nsub,
               status="ok" if ok else "failed")
    if not ok:
        return row
    h = np.asarray(gwf.output.head().get_data()).ravel()
    d = b.T * b.R * math.cos(math.radians(dip)) ** 2
    lay = np.arange(h.size) // ncpl
    ana = b.HB + (nlay - 1 - lay) * d / nsub
    e = h - ana
    hh = h.reshape(nlay, ncpl)
    unit_drop = (hh[:-nsub] - hh[nsub:]).mean()
    row.update(rms_mm=1e3 * math.sqrt(np.mean(e ** 2)), max_mm=1e3 * np.abs(e).max(),
               drop_mm=1e3 * unit_drop, drop_ana_mm=1e3 * d, drop_ratio=unit_drop / d)
    return row


def main():
    root = os.environ.get("A5_RUNS", str(pl.Path(__import__("tempfile").gettempdir()) / "a25_runs"))
    nsubs = [int(v) for v in os.environ.get("A25_NSUB", "1,2,3").split(",")]
    tasks = [(dp, n, root) for n in nsubs for dp in range(5, 61, 5)]
    import pandas as pd
    with mp.Pool(6) as pool:
        rows = list(pool.imap_unordered(run, tasks))
    df = pd.DataFrame(rows).sort_values(["nsub", "dip"])
    df["ncell"] = b.NCELL
    name = "results_provost_sublayers_25d.csv" if b.NCELL == 250 else f"results_provost_sublayers_{b.NCELL}.csv"
    df.to_csv(HERE / name, index=False, float_format="%.6g")
    print(df.pivot_table(index="dip", columns="nsub", values="rms_mm").round(3).to_string())
    print(df.pivot_table(index="dip", columns="nsub", values="drop_ratio").round(4).to_string())


if __name__ == "__main__":
    main()
