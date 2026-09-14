"""Benchmark A on a 2.5-D layered Voronoi grid (one plan tessellation, stacked columns).

Replaces the offset-layer grid of the lost mesh3d pipeline, so that the manuscript uses
only layered grids whose layers share one plan-view tessellation (the scope of the paper
and of the released package).

Grid: 3000 m x 3000 m square, centroidal Voronoi tessellation of 250 cells (40 Lloyd
iterations, fixed seed), extruded through three 50 m layers dipping in +x at alpha.
Layered DISU arrays are built with disv2disu (staggered=False) from the Provost et al.
(2025) data release; enhanced connectivity with the same utility (staggered=True).

Formulations
  plan        layered connectivity, plan-view geometry
  area        vertical HWVA x sec(alpha_fit)
  full        vertical HWVA x sec^2(alpha_fit)          alpha_fit: released package, k = 8
  full_k16    as full, k = 16 (heterogeneous, 30/45/60 deg)
  full_k32    as full, k = 32 (heterogeneous, 30/45/60 deg)
  full_exact  vertical HWVA x sec^2(dip)
  provost     enhanced (staggered) connectivity, Provost et al. (2025)
Each MF6 run optionally with XT3D (ANGLE1-3 written). MODFLOW-USG: plan, area, full,
full_exact (FAHL: horizontal = width x thickness, vertical x sec^2; CL12 unchanged).

Boundary conditions as in the manuscript: uniform recharge R on the top layer, constant
head 100 m on every cell of the bottom layer, lateral sides no-flow. Analytical head per
layer from Eq. (10) of v2 (bottom layer = 100 m).
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import pathlib as pl
import shutil
import sys

import numpy as np

HERE = pl.Path(__file__).resolve().parent
PAPER = HERE.parent
ARCH = pl.Path(os.environ.get("PROVOST_ARCHIVE", PAPER / "enhanced_connectivity" / "archive"))
sys.path.insert(0, str(ARCH / "ancillary"))
sys.path.insert(0, str(PAPER.parent / "src"))
MF6 = pl.Path(os.environ.get("MF6_EXE", ARCH / "bin" / "mf6.exe"))   # MODFLOW 6.5.0 from the Provost et al. (2025) release
USG = pl.Path(os.environ.get("MFUSG_EXE", "USGs_1"))   # MODFLOW-USG (USGs_1)
L, SEED, LLOYD = 3000.0, 20260913, 40
NCELL = int(os.environ.get("A25_NCELL", "250"))       # cells per layer (resolution study sets this)
GRID_JSON = HERE / f"grid_voronoi_{NCELL}.json"
RUNTAG = "" if NCELL == 250 else f"n{NCELL}"
T, R, NLAY, HB, ZTOP0 = 50.0, 1.0e-3, 3, 100.0, 150.0
KSETS = {"homog": (1.0, 1.0, 1.0), "hetK": (1.0, 0.01, 1.0)}


# ---------------------------------------------------------------- grid
def make_voronoi():
    from scipy.spatial import Voronoi
    from shapely.geometry import MultiPoint, box
    from shapely.geometry.polygon import orient

    rng = np.random.default_rng(SEED)
    pts = rng.uniform(0.0, L, (NCELL, 2))
    dom = box(0.0, 0.0, L, L)
    for it in range(LLOYD + 1):
        mir = np.vstack([pts,
                         np.c_[-pts[:, 0], pts[:, 1]], np.c_[2 * L - pts[:, 0], pts[:, 1]],
                         np.c_[pts[:, 0], -pts[:, 1]], np.c_[pts[:, 0], 2 * L - pts[:, 1]]])
        vor = Voronoi(mir)
        polys = []
        for i in range(NCELL):
            reg = vor.regions[vor.point_region[i]]
            assert -1 not in reg and len(reg) >= 3
            polys.append(MultiPoint(vor.vertices[reg]).convex_hull.intersection(dom))
        if it < LLOYD:
            pts = np.array([[p.centroid.x, p.centroid.y] for p in polys])
    key, verts, cell2d = {}, [], []
    for i, p in enumerate(polys):
        p = orient(p, sign=-1.0)                       # clockwise, as MODFLOW 6 requires
        iv = []
        for x, y in list(p.exterior.coords)[:-1]:
            k = (round(x, 3), round(y, 3))
            if k not in key:
                key[k] = len(verts)
                verts.append([len(verts), k[0], k[1]])
            j = key[k]
            if not iv or iv[-1] != j:
                iv.append(j)
        if len(iv) > 1 and iv[0] == iv[-1]:
            iv.pop()
        c = p.centroid
        cell2d.append([i, float(c.x), float(c.y), len(iv)] + iv)
    return verts, cell2d


def load_grid():
    if not GRID_JSON.exists():
        verts, cell2d = make_voronoi()
        GRID_JSON.write_text(json.dumps({"vertices": verts, "cell2d": cell2d}))
    g = json.loads(GRID_JSON.read_text())
    return g["vertices"], g["cell2d"]


def layered(cell2d, dip):
    xc = np.array([c[1] for c in cell2d])
    top = ZTOP0 - xc * math.tan(math.radians(dip))
    botm = np.array([top - (k + 1) * T for k in range(NLAY)])
    return top, botm


def analytic(dip, K):
    K = np.asarray(K, float)
    d = T * R * math.cos(math.radians(dip)) ** 2 / K
    inc = 0.5 * (d[:-1] + d[1:])
    return HB + np.array([inc.sum(), inc[1], 0.0]), d, inc


def gridprops(verts, cell2d, dip, staggered):
    from disv2disu import Disv2Disu
    top, botm = layered(cell2d, dip)
    return Disv2Disu(verts, cell2d, top, botm, staggered).get_gridprops_disu6()


def fitted_cos(gp, cell2d, k):
    from disu_slope_corrector.correction import DisuGeometry, apply_correction
    xc = np.tile([c[1] for c in cell2d], NLAY)
    yc = np.tile([c[2] for c in cell2d], NLAY)
    geom = DisuGeometry(xc=xc, yc=yc, zc=0.5 * (gp["top"] + gp["bot"]), top=gp["top"], bot=gp["bot"],
                        area=gp["area"], iac=gp["iac"], ja=gp["ja"] + 1, ihc=gp["ihc"],
                        cl12=gp["cl12"], hwva=gp["hwva"])
    return apply_correction(geom, neighbours=k, code="mf6").cos_alpha


def vertical_mask(gp):
    ia = np.concatenate([[0], np.cumsum(gp["iac"])])
    diag = np.zeros(gp["nja"], bool)
    diag[ia[:-1]] = True
    return (gp["ihc"] == 0) & ~diag


def geometry(var, dip, verts, cell2d):
    """Return (gridprops, vertical HWVA factor array or None, alpha diagnostics)."""
    gp = gridprops(verts, cell2d, dip, staggered=(var == "provost"))
    diag = {}
    if var in ("plan", "provost"):
        return gp, diag
    vert = vertical_mask(gp)
    hw = gp["hwva"].astype(float).copy()
    if var == "full_exact":
        fac = np.full(vert.sum(), 1.0 / math.cos(math.radians(dip)) ** 2)
    else:
        k = {"area": 8, "full": 8, "full_k16": 16, "full_k32": 32}[var]
        cosa = fitted_cos(gp, cell2d, k)[vert]
        alpha = np.degrees(np.arccos(np.clip(cosa, -1, 1)))
        diag = dict(alpha_err_max=float(np.abs(alpha - dip).max()),
                    alpha_err_rms=float(np.sqrt(np.mean((alpha - dip) ** 2))))
        fac = 1.0 / cosa if var == "area" else 1.0 / cosa ** 2
    hw[vert] *= fac
    gp = dict(gp)
    gp["hwva"] = hw
    return gp, diag


def metrics(h, dip, K, ncpl):
    hl, d, inc = analytic(dip, K)
    lay = np.arange(h.size) // ncpl
    e = h - hl[lay]
    hh = h.reshape(NLAY, ncpl)
    pair12, pair23 = hh[0] - hh[1], hh[1] - hh[2]
    return dict(
        rms_mm=1e3 * math.sqrt(np.mean(e ** 2)), max_mm=1e3 * np.abs(e).max(),
        mean_abs_mm=1e3 * np.abs(e).mean(),
        rms_L1_mm=1e3 * math.sqrt(np.mean(e[lay == 0] ** 2)), rms_L2_mm=1e3 * math.sqrt(np.mean(e[lay == 1] ** 2)),
        inc12_mm=1e3 * pair12.mean(), inc23_mm=1e3 * pair23.mean(),
        inc12_sd_mm=1e3 * pair12.std(), inc23_sd_mm=1e3 * pair23.std(), inc_ana_mm=1e3 * inc[0],
        drop_mm=1e3 * 0.5 * (pair12.mean() + pair23.mean()), drop_ana_mm=1e3 * d[0],
        head_sum_ana_mm=1e3 * d.sum(),
        nrms_pct=100 * math.sqrt(np.mean(e ** 2)) / d.sum())


# ---------------------------------------------------------------- MODFLOW 6
def run_mf6(task):
    import flopy
    case, dip, var, xt3d, root = task
    verts, cell2d = load_grid()
    ncpl, K = len(cell2d), KSETS[case]
    ws = pl.Path(root) / "mf6" / RUNTAG / case / f"dip_{dip:02d}" / var / ("xt3d" if xt3d else "off")
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    row = dict(code="mf6", case=case, dip=dip, variant=var, xt3d=xt3d)
    gp, diag = geometry(var, dip, verts, cell2d)
    row.update(diag)
    sim = flopy.mf6.MFSimulation(sim_name="a", sim_ws=str(ws), exe_name=str(MF6))
    flopy.mf6.ModflowTdis(sim)
    flopy.mf6.ModflowIms(sim, linear_acceleration="bicgstab", outer_maximum=200, inner_maximum=2000,
                         outer_dvclose=1e-10, inner_dvclose=1e-10, rcloserecord=[1e-10, "strict"])
    gwf = flopy.mf6.ModflowGwf(sim, modelname="a")
    flopy.mf6.ModflowGwfdisu(gwf, **gp)
    flopy.mf6.ModflowGwfic(gwf, strt=HB)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=np.repeat(np.array(K), ncpl), xt3doptions=xt3d,
                            angle1=0.0, angle2=0.0, angle3=0.0)
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[[((NLAY - 1) * ncpl + i,), HB] for i in range(ncpl)])
    flopy.mf6.ModflowGwfrch(gwf, stress_period_data=[[(i,), R] for i in range(ncpl)])
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="a.hds", saverecord=[("HEAD", "ALL")])
    sim.write_simulation(silent=True)
    ok, buff = sim.run_simulation(silent=True, report=True)
    if not ok:
        row["status"] = "failed"
        row["message"] = " | ".join(ln.strip() for ln in buff if "error" in ln.lower())[:200]
        return row
    h = np.asarray(gwf.output.head().get_data()).ravel()
    row.update(status="ok", cells=h.size, nja=int(gp["nja"]), **metrics(h, dip, K, ncpl))
    return row


# ---------------------------------------------------------------- MODFLOW-USG
def run_usg(task):
    import flopy
    case, dip, var, root = task
    verts, cell2d = load_grid()
    ncpl, K = len(cell2d), KSETS[case]
    ws = pl.Path(root) / "usg" / case / f"dip_{dip:02d}" / var
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    row = dict(code="mfusg", case=case, dip=dip, variant=var, xt3d=False)
    gp, diag = geometry(var, dip, verts, cell2d)
    row.update(diag)
    n = gp["nodes"]
    ihc = np.asarray(gp["ihc"])
    ia = np.concatenate([[0], np.cumsum(gp["iac"])])
    isdiag = np.zeros(gp["nja"], bool)
    isdiag[ia[:-1]] = True
    ivc = np.where((ihc == 0) & ~isdiag, 1, 0)
    thick = (gp["top"] - gp["bot"])
    owner = np.repeat(np.arange(n), gp["iac"])
    fahl = np.asarray(gp["hwva"], float).copy()
    horiz = (ivc == 0) & ~isdiag
    fahl[horiz] = fahl[horiz] * thick[owner[horiz]]    # MODFLOW-USG: horizontal FAHL is an area
    cl12 = np.asarray(gp["cl12"], float).copy()
    fahl[isdiag] = 0.0
    cl12[isdiag] = 0.0
    m = flopy.mfusg.MfUsg(modelname="a", model_ws=str(ws), exe_name=str(USG), structured=False)
    flopy.mfusg.MfUsgDisU(m, nodes=n, nlay=NLAY, njag=int(gp["nja"]), ivsd=0, nodelay=[ncpl] * NLAY,
                          top=np.asarray(gp["top"]), bot=np.asarray(gp["bot"]), area=np.asarray(gp["area"]),
                          iac=np.asarray(gp["iac"]), ja=np.asarray(gp["ja"]), ivc=ivc, cl12=cl12, fahl=fahl,
                          idsymrd=0, lenuni=2)
    kc = [np.full(ncpl, K[i]) for i in range(NLAY)]
    flopy.mfusg.MfUsgBas(m, ibound=[np.ones(ncpl, int), np.ones(ncpl, int), -np.ones(ncpl, int)],
                         strt=HB, structured=False)
    flopy.mfusg.MfUsgLpf(m, laytyp=0, hk=kc, vka=kc, constantcv=True, novfc=True)
    flopy.mfusg.MfUsgRch(m, nrchop=1, rech=R)
    flopy.mfusg.MfUsgSms(m, hclose=1e-9, hiclose=1e-10, mxiter=2000, iter1=1000, nonlinmeth=1, linmeth=2,
                         options="SIMPLE")
    flopy.mfusg.MfUsgOc(m)
    m.write_input()
    ok, _ = m.run_model(silent=True)
    hds = ws / "a.hds"
    if not (ok and hds.exists()):
        row["status"] = "failed"
        return row
    data = flopy.utils.HeadUFile(str(hds)).get_data()
    h = np.concatenate([np.asarray(a).ravel() for a in data])
    row.update(status="ok", cells=h.size, **metrics(h, dip, K, ncpl))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runroot", default=os.environ.get("A5_RUNS", str(pl.Path(__import__("tempfile").gettempdir()) / "a25_runs")))
    ap.add_argument("--procs", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--only", default="mf6,usg")
    a = ap.parse_args()
    verts, cell2d = load_grid()
    print("grid:", len(cell2d), "cells per layer,", len(verts), "vertices", flush=True)
    dips = list(range(0, 61, 5))
    mf6_tasks = []
    for dp in dips:
        mf6_tasks += [("homog", dp, v, x, a.runroot) for v, x in
                      (("plan", False), ("plan", True), ("area", False), ("full", False), ("full", True),
                       ("full_exact", False), ("provost", True))]
        mf6_tasks += [("hetK", dp, v, x, a.runroot) for v, x in
                      (("plan", False), ("plan", True), ("area", False), ("full", False), ("full", True),
                       ("full_exact", False), ("provost", True))]
    usg_tasks = [(c, dp, v, a.runroot) for c in KSETS for dp in dips for v in ("plan", "area", "full", "full_exact")]
    import pandas as pd
    rows = []
    with mp.Pool(a.procs) as pool:
        if "mf6" in a.only:
            for i, r in enumerate(pool.imap_unordered(run_mf6, mf6_tasks), 1):
                rows.append(r)
                if i % 25 == 0 or i == len(mf6_tasks):
                    print(f"  mf6 {i}/{len(mf6_tasks)}", flush=True)
        if "usg" in a.only:
            for i, r in enumerate(pool.imap_unordered(run_usg, usg_tasks), 1):
                rows.append(r)
                if i % 25 == 0 or i == len(usg_tasks):
                    print(f"  usg {i}/{len(usg_tasks)}", flush=True)
    df = pd.DataFrame(rows).sort_values(["code", "case", "variant", "xt3d", "dip"])
    out = HERE / ("results_benchmarkA_25d.csv" if a.only == "mf6,usg" else f"results_benchmarkA_25d_{a.only}.csv")
    df.to_csv(out, index=False, float_format="%.6g")
    print("wrote", out)
    show = df[df.dip.isin([30, 45, 60])][["code", "case", "variant", "xt3d", "dip", "status", "rms_mm", "drop_mm", "inc12_mm", "nrms_pct"]]
    print(show.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
