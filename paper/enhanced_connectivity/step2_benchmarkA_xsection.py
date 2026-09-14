"""A3 step 2: Benchmark A on a structured cross-section.

Compares, for flow across dipping layers driven by recharge:
  planview        layered connectivity, standard conductance (plan-view DISU)
  planview_xt3d   layered connectivity, XT3D
  provost_std     full (vertically staggered) connectivity, standard conductance
  provost_xt3d    full connectivity + XT3D  (Provost et al. 2025)
  corrected       layered connectivity, standard, HWVA x sec^2(a) on vertical
                  connections (slope correction, MODFLOW 6 form)
  corrected_xt3d  slope correction + XT3D

Grids are built with dis2disu.py and run with MODFLOW 6.5.0, both taken
unchanged from the Provost et al. (2025) data release (doi:10.5066/P13BNARA).

Boundary sets:
  noflow      lateral sides no-flow (as in the manuscript's Benchmark A)
  analytical  analytical heads on the left and right columns (as in Provost
              et al. 2025), consistent with the 1-D bedding-normal solution
Both use recharge R on the top cell of each column and analytical heads on
the bottom grid layer.

Analytical solution (bedding-normal flux R cos a): within geological layer g,
dh/ds = R cos^2(a) / K_g, where s is the vertical height above the base plane.
"""
import argparse
import math
import os
import multiprocessing as mp
import pathlib as pl
import shutil
import sys

import numpy as np

HERE = pl.Path(__file__).resolve().parent
ARCHIVE = pl.Path(os.environ.get("PROVOST_ARCHIVE", HERE / "archive"))
sys.path.insert(0, str(ARCHIVE / "ancillary"))
MF6 = pl.Path(os.environ.get("MF6_EXE", ARCHIVE / "bin" / "mf6.exe"))

T_GEO = 50.0          # vertical thickness of each geological layer (m)
N_GEO = 3             # number of geological layers
RECH = 1.0e-3         # recharge per unit plan area (m/d)
LX = 1050.0           # cross-section length (m)
DZTOL = 1.0e-5        # overlap tolerance for staggered connections (Provost)

VARIANTS = {
    "planview": dict(staggered=False, xt3d=False, correct=False),
    "planview_xt3d": dict(staggered=False, xt3d=True, correct=False),
    "provost_std": dict(staggered=True, xt3d=False, correct=False),
    "provost_xt3d": dict(staggered=True, xt3d=True, correct=False),
    "corrected": dict(staggered=False, xt3d=False, correct=True),
    "corrected_xt3d": dict(staggered=False, xt3d=True, correct=True),
}


def build_grid(nsub, dip):
    """Square cells (dx = dz), interfaces evaluated at column centres."""
    dz = T_GEO / nsub
    dx = dz
    ncol = int(round(LX / dx))
    nlay = N_GEO * nsub
    tan_a = math.tan(math.radians(dip))
    xc = (np.arange(ncol) + 0.5) * dx
    zbase = xc * tan_a
    top = (zbase + N_GEO * T_GEO).reshape(1, ncol)
    botm = np.array(
        [zbase + N_GEO * T_GEO - (k + 1) * dz for k in range(nlay)]
    ).reshape(nlay, 1, ncol)
    return dx, dz, ncol, nlay, xc, top, botm


def analytical_head(xc, top, botm, dip, kgeo, nsub):
    """Head at cell centres; h = 0 on the base plane."""
    nlay, _, ncol = botm.shape
    c2 = math.cos(math.radians(dip)) ** 2
    tan_a = math.tan(math.radians(dip))
    tops = np.vstack([top, botm[:-1, 0, :]])
    kbot_up = list(reversed(kgeo))  # geological K from the base upward
    h = np.zeros((nlay, ncol))
    for k in range(nlay):
        s = 0.5 * (tops[k] + botm[k, 0]) - xc * tan_a
        g = int(np.floor(s[0] / T_GEO + 1e-9))
        hk = sum(RECH * c2 * T_GEO / kbot_up[i] for i in range(g))
        h[k] = hk + RECH * c2 * (s - g * T_GEO) / kbot_up[g]
    return h


def build_and_run(task):
    import flopy
    from dis2disu import Dis2Disu

    bc, nsub, dip, vname, kgeo, runroot, dvclose = task
    opt = VARIANTS[vname]
    ws = pl.Path(runroot) / bc / f"n{nsub}" / f"dip{dip:04.1f}" / vname
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)

    dx, dz, ncol, nlay, xc, top, botm = build_grid(nsub, dip)
    h_ana = analytical_head(xc, top, botm, dip, kgeo, nsub)
    d2d = Dis2Disu(np.full(ncol, dx), np.array([1.0]), top, botm,
                   staggered=opt["staggered"], dztol=DZTOL)
    gp = d2d.get_gridprops_disu6()

    if opt["correct"]:
        sec2 = 1.0 / math.cos(math.radians(dip)) ** 2
        ia = np.concatenate([[0], np.cumsum(gp["iac"])])
        isdiag = np.zeros(gp["nja"], dtype=bool)
        isdiag[ia[:-1]] = True
        vert = (gp["ihc"] == 0) & ~isdiag
        gp["hwva"] = gp["hwva"].copy()
        gp["hwva"][vert] *= sec2

    kcell = np.repeat(np.array(kgeo, dtype=float), nsub)
    kcell = np.repeat(kcell, ncol)

    sim = flopy.mf6.MFSimulation(sim_name="m", sim_ws=str(ws), exe_name=str(MF6))
    flopy.mf6.ModflowTdis(sim)
    flopy.mf6.ModflowIms(sim, linear_acceleration="bicgstab",
                         outer_maximum=200, inner_maximum=2000,
                         outer_dvclose=dvclose, inner_dvclose=dvclose,
                         rcloserecord=[dvclose, "strict"])
    gwf = flopy.mf6.ModflowGwf(sim, modelname="m")
    flopy.mf6.ModflowGwfdisu(gwf, **gp)
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)
    # ANGLE2 must be given explicitly: if it is absent MODFLOW 6 sets NOZEE and
    # XT3D ignores the vertical offsets of horizontal connections
    # (Xt3dInterface.f90, "if (this%iangle2 == 0) this%nozee = .true.").
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=kcell, xt3doptions=opt["xt3d"],
                            angle1=0.0, angle2=0.0, angle3=0.0)

    ischd = np.zeros((nlay, ncol), dtype=bool)
    ischd[-1, :] = True
    if bc == "analytical":
        ischd[:, 0] = True
        ischd[:, -1] = True
    chd = [[(k * ncol + j,), h_ana[k, j]] for k, j in zip(*np.nonzero(ischd))]
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd)
    rch = [[(j,), RECH] for j in range(ncol) if not ischd[0, j]]
    flopy.mf6.ModflowGwfrch(gwf, stress_period_data=rch)
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="m.hds",
                           saverecord=[("HEAD", "ALL")])
    sim.write_simulation(silent=True)
    success, _ = sim.run_simulation(silent=True)

    row = dict(bc=bc, nsub=nsub, dip=dip, variant=vname, ncol=ncol, nlay=nlay,
               status="ok" if success else "failed")
    if not success:
        lst = ws / "mfsim.lst"
        msg = ""
        if lst.exists():
            lines = lst.read_text(errors="ignore").splitlines()
            err = [ln.strip() for ln in lines if "ERROR" in ln.upper() or "FAIL" in ln.upper()]
            msg = " | ".join(err[:2])
        row.update(message=msg[:200])
        return row

    h = flopy.utils.HeadFile(str(ws / "m.hds")).get_data().ravel().reshape(nlay, ncol)
    err = h - h_ana
    free = ~ischd
    mid = np.zeros(ncol, dtype=bool)
    mid[ncol // 3: 2 * ncol // 3] = True
    # Head drop between cell centres one geological layer apart, measured
    # against the analytical field for the same cell pairs, so the metric is
    # correct when K varies between layers (paper Eq. 11).
    drops = (h[:-nsub] - h[nsub:])[:, mid]
    drops_ana = (h_ana[:-nsub] - h_ana[nsub:])[:, mid]
    drop_ana = float(drops_ana.mean())
    row.update(
        rms_all_mm=1e3 * float(np.sqrt(np.mean(err[free] ** 2))),
        rms_mid_mm=1e3 * float(np.sqrt(np.mean(err[:, mid][free[:, mid]] ** 2))),
        max_abs_mm=1e3 * float(np.abs(err[free]).max()),
        drop_mid_mm=1e3 * float(drops.mean()),
        drop_ana_mm=1e3 * drop_ana,
        drop_ratio=float(drops.mean() / drop_ana),
        kgeo=",".join(str(v) for v in kgeo),
    )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runroot", required=True, help="scratch folder for model runs")
    ap.add_argument("--dips", default="0,5,10,15,20,25,30,35,40,45,50,55,60")
    ap.add_argument("--nsub", default="1,2,3")
    ap.add_argument("--bcs", default="noflow,analytical")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--kgeo", default="1,1,1", help="K of geological layers, top to bottom")
    ap.add_argument("--out", default=str(HERE / "results_step2.csv"))
    ap.add_argument("--procs", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--dvclose", type=float, default=1e-11, help="solver closure (m)")
    a = ap.parse_args()

    kgeo = [float(v) for v in a.kgeo.split(",")]
    tasks = [(bc, int(n), float(d), v, kgeo, a.runroot, a.dvclose)
             for bc in a.bcs.split(",")
             for n in a.nsub.split(",")
             for d in a.dips.split(",")
             for v in a.variants.split(",")]
    print(f"{len(tasks)} runs on {a.procs} processes", flush=True)
    rows = []
    with mp.Pool(a.procs) as pool:
        for i, row in enumerate(pool.imap_unordered(build_and_run, tasks), 1):
            rows.append(row)
            if i % 25 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} done", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows).sort_values(["bc", "nsub", "variant", "dip"])
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
