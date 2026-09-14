"""A3 step 3a: the Provost et al. (2025) dipping-aquifer benchmark.

Their test problem (after Bardot et al. 2023): a homogeneous isotropic aquifer
of uniform thickness dipping at angle theta, embedded between two low-K domains.
Heads are specified on the model perimeter from the uniform-flow potential

    h(x, z) = -(x cos theta + z sin theta)

which drives unit specific discharge ALONG the aquifer.  A correct model returns
|q| = 1 at angle theta in the middle of the aquifer.  Note this is flow along the
dip, not the cross-bedding flow of the manuscript's Benchmark A.

Grid construction (dis2disu.py), the DISU conventions and MODFLOW 6.5.0 are
taken unchanged from the Provost et al. (2025) data release (doi:10.5066/P13BNARA).
The slope correction is applied with the published disu_slope_corrector package.

Variants (their four, plus the correction):
    vo-s        layered connectivity, standard conductance
    vo-x        layered connectivity, XT3D
    vs-s        full (staggered) connectivity, standard conductance
    vs-x        full connectivity + XT3D         <- their remedy
    vo-s-corr   layered connectivity, standard, slope-corrected geometry
    vo-x-corr   layered connectivity, XT3D, slope-corrected geometry
    vs-x-corr   full connectivity + XT3D, slope-corrected geometry
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
sys.path.insert(0, str(HERE.parents[1] / "src"))
MF6 = pl.Path(os.environ.get("MF6_EXE", ARCHIVE / "bin" / "mf6.exe"))

NCOL = 11          # as in Provost et al. (2025)
LX = 11.0
K_CHAN = 1.0
DZTOL = 1.0e-5

VARIANTS = {
    "vo-s": dict(staggered=False, xt3d=False, correct=False),
    "vo-x": dict(staggered=False, xt3d=True, correct=False),
    "vs-s": dict(staggered=True, xt3d=False, correct=False),
    "vs-x": dict(staggered=True, xt3d=True, correct=False),
    "vo-s-corr": dict(staggered=False, xt3d=False, correct=True),
    "vo-x-corr": dict(staggered=False, xt3d=True, correct=True),
    "vs-x-corr": dict(staggered=True, xt3d=True, correct=True),
    # vertical part of the correction only (HWVA x sec^2 on ihc==0), leaving
    # horizontal face heights untouched, to separate the two halves.
    "vo-s-corrV": dict(staggered=False, xt3d=False, correct=True, vertical_only=True),
    "vs-x-corrV": dict(staggered=True, xt3d=True, correct=True, vertical_only=True),
}


def set_up_grid(nlay_chan, theta, nrow=1):
    """Port of set_up_dis_grid() from the data release (domain=True).

    nrow > 1 repeats the cross-section into the page (no flow in y, so the
    solution is unchanged). It is needed because the slope corrector fits a
    plane z = a + bx + cy to interface samples, which is underdetermined when
    every cell shares one y value, as in the original one-row model.
    """
    delr = LX / NCOL
    delz = delr
    zoffset = delr * math.tan(math.radians(theta))
    zthick = nlay_chan * delz              # vertical thickness of the aquifer
    zspan = (NCOL - 1) * zoffset + zthick
    nlay_dom = nlay_chan                   # layers above and below the aquifer
    nlay = nlay_chan + 2 * nlay_dom

    top_chan = zthick + 0.5 * zoffset + np.arange(NCOL) * zoffset + zthick
    lz = zthick + zspan + zthick
    top = np.full((nrow, NCOL), lz)
    botm = np.empty((nlay, nrow, NCOL))
    tc = np.tile(top_chan, (nrow, 1))
    dz_up = (top - tc) / nlay_dom
    botm[0] = top - dz_up
    for k in range(1, nlay_dom):
        botm[k] = botm[k - 1] - dz_up
    for k in range(nlay_dom, nlay_dom + nlay_chan):
        botm[k] = botm[k - 1] - delz
    dz_lo = (tc - zthick) / nlay_dom
    for k in range(nlay_dom + nlay_chan, nlay):
        botm[k] = botm[k - 1] - dz_lo

    ischan = np.zeros((nlay, nrow, NCOL), dtype=bool)
    ischan[nlay_dom: nlay_dom + nlay_chan, :, :] = True
    xc = (np.arange(NCOL) + 0.5) * delr
    return delr, nlay, nlay_dom, zthick, xc, top, botm, ischan


def cell_centres(top, botm):
    tops = np.concatenate([top[None, :, :], botm[:-1]], axis=0)
    return 0.5 * (tops + botm)


def recalculated_q(gp, flowja, ischan, delz_chan, staggered, cell):
    """Port of recalculate_spdis() from the data release: cell-centred qx, qz
    from FLOW-JA-FACE, excluding connections that cross the aquifer boundary."""
    iac, ja, ihc, hwva = gp["iac"], gp["ja"], gp["ihc"], gp["hwva"]
    top, bot, angx = gp["top"], gp["bot"], gp["angldegx"]
    flat = ischan.ravel()
    start = int(np.sum(iac[:cell]))
    qxn = qxd = qzn = qzd = 0.0
    for k in range(int(iac[cell])):
        pos = start + k
        nbr = int(ja[pos])
        if nbr == cell or flat[nbr] != flat[cell]:
            continue
        if ihc[pos] > 0:
            # x-direction faces only (Dis2Disu writes ANGLDEGX 0/180 for those);
            # y faces carry no flow here but would dilute the average.
            if min(abs(angx[pos] % 180.0), 180.0 - abs(angx[pos] % 180.0)) > 1e-6:
                continue
            dz = (min(top[cell], top[nbr]) - max(bot[cell], bot[nbr])) if staggered else delz_chan
            incr = flowja[pos] / (hwva[pos] * dz)
            qxn += -incr if nbr > cell else incr
            qxd += 1.0
        else:
            incr = flowja[pos] / hwva[pos]
            qzn += incr if nbr > cell else -incr
            qzd += 1.0
    return (qxn / qxd if qxd else 0.0), (qzn / qzd if qzd else 0.0)


def run_one(task):
    import flopy
    from dis2disu import Dis2Disu
    from disu_slope_corrector.correction import DisuGeometry, apply_correction

    nlay_chan, theta, vname, k_dom, runroot, dvclose, nrow = task
    opt = VARIANTS[vname]
    ws = pl.Path(runroot) / f"n{nlay_chan}" / f"dip{theta:04.1f}" / vname
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)

    delr, nlay, nlay_dom, zthick, xc, top, botm, ischan = set_up_grid(nlay_chan, theta, nrow)
    zc = cell_centres(top, botm)
    th = math.radians(theta)
    h_ana = -(xc[None, None, :] * math.cos(th) + zc * math.sin(th))

    d2d = Dis2Disu(np.full(NCOL, delr), np.full(nrow, 1.0), top, botm,
                   staggered=opt["staggered"], dztol=DZTOL)
    gp = d2d.get_gridprops_disu6()

    if opt["correct"]:
        yc2d = np.broadcast_to(((np.arange(nrow) + 0.5))[:, None], (nrow, NCOL))
        geom = DisuGeometry(
            xc=np.tile(np.broadcast_to(xc, (nrow, NCOL)).ravel(), nlay),
            yc=np.tile(yc2d.ravel(), nlay),
            zc=zc.ravel(), top=gp["top"].astype(float), bot=gp["bot"].astype(float),
            area=gp["area"].astype(float), iac=gp["iac"], ja=gp["ja"] + 1,
            ihc=gp["ihc"], cl12=gp["cl12"].astype(float),
            hwva=gp["hwva"].astype(float))
        res = apply_correction(geom, neighbours=8, max_dip_deg=89.0, code="mf6")
        cl12, hwva = res.cl12.copy(), res.hwva.copy()
        if opt.get("vertical_only"):
            horiz = gp["ihc"] != 0
            hwva[horiz] = gp["hwva"][horiz]
            cl12[horiz] = gp["cl12"][horiz]
        gp["cl12"], gp["hwva"] = cl12, hwva
        n_pinched = 0 if opt.get("vertical_only") else int(res.is_pinched.sum())
    else:
        n_pinched = 0

    kcell = np.where(ischan.ravel(), K_CHAN, k_dom)

    sim = flopy.mf6.MFSimulation(sim_name="m", sim_ws=str(ws), exe_name=str(MF6))
    flopy.mf6.ModflowTdis(sim)
    flopy.mf6.ModflowIms(sim, linear_acceleration="bicgstab",
                         outer_maximum=500, inner_maximum=2000,
                         outer_dvclose=dvclose, inner_dvclose=dvclose,
                         rcloserecord=[dvclose, "strict"])
    gwf = flopy.mf6.ModflowGwf(sim, modelname="m", save_flows=True)
    flopy.mf6.ModflowGwfdisu(gwf, **gp)
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)
    # ANGLE2 must be written or MODFLOW sets NOZEE and XT3D ignores the
    # vertical offsets of horizontal connections (Xt3dInterface.f90).
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=kcell, save_specific_discharge=True,
                            xt3doptions=opt["xt3d"], angle1=0.0, angle2=0.0, angle3=0.0)

    ischd = np.zeros((nlay, nrow, NCOL), dtype=bool)
    ischd[:, :, 0] = True
    ischd[:, :, -1] = True
    ischd[0, :, 1:-1] = True
    ischd[-1, :, 1:-1] = True
    node = lambda k, i, j: (k * nrow + i) * NCOL + j
    chd = [[(node(k, i, j),), h_ana[k, i, j]] for k, i, j in zip(*np.nonzero(ischd))]
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd)
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="m.hds", budget_filerecord="m.cbc",
                           saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")])
    sim.write_simulation(silent=True)
    success, _ = sim.run_simulation(silent=True)

    row = dict(nlay_chan=nlay_chan, dip=theta, variant=vname, k_dom=k_dom,
               n_pinched=n_pinched, status="ok" if success else "failed")
    if not success:
        lst = ws / "mfsim.lst"
        msg = ""
        if lst.exists():
            bad = [ln.strip() for ln in lst.read_text(errors="ignore").splitlines()
                   if "ERROR" in ln.upper() or "FAIL" in ln.upper()]
            msg = " | ".join(bad[:2])
        row["message"] = msg[:200]
        return row

    h = flopy.utils.HeadFile(str(ws / "m.hds")).get_data().ravel().reshape(nlay, nrow, NCOL)
    free = ischan & ~ischd
    err = (h - h_ana)[free]

    row.update(rms_head=float(np.sqrt(np.mean(err ** 2))),
               max_head=float(np.abs(err).max()))

    # Flux diagnostics need the budget file; a model whose aquifer has been
    # disconnected (every same-layer face pinched out) can leave it unusable.
    try:
        bud = flopy.utils.CellBudgetFile(str(ws / "m.cbc"))
        ctr = node(nlay_dom + nlay_chan // 2, nrow // 2, NCOL // 2)
        spdis = bud.get_data(text="DATA-SPDIS")[0]
        idx = int(np.nonzero(np.asarray(spdis["node"]) - 1 == ctr)[0][0])
        qmag_mf6 = math.hypot(float(spdis["qx"][idx]), float(spdis["qz"][idx]))
        flowja = bud.get_data(text="FLOW-JA-FACE")[0].ravel()
        qx, qz = recalculated_q(gp, flowja, ischan, delr, opt["staggered"], ctr)
        qmag = math.hypot(qx, qz)
        qang = math.degrees(math.atan2(qz, qx))
        row.update(qmag=qmag, qang=qang,
                   qmag_err_pct=100.0 * (qmag - 1.0),
                   qang_err_deg=qang - theta,
                   qmag_err_pct_mf6spdis=100.0 * (qmag_mf6 - 1.0))
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        row["status"] = "ok-no-budget"
        row["message"] = f"{type(exc).__name__}: {exc}"[:200]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runroot", required=True)
    ap.add_argument("--dips", default="0,10,20,30,40,50,60,70")
    ap.add_argument("--nlay-chan", default="1,3")
    ap.add_argument("--kdom", type=float, default=1.0e-6)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--dvclose", type=float, default=1e-9)
    ap.add_argument("--nrow", type=int, default=3,
                    help="rows into the page; >1 is required by the slope corrector")
    ap.add_argument("--out", default=str(HERE / "results_provost_benchmark.csv"))
    ap.add_argument("--procs", type=int, default=max(1, mp.cpu_count() - 2))
    a = ap.parse_args()

    tasks = [(int(n), float(d), v, a.kdom, a.runroot, a.dvclose, a.nrow)
             for n in a.nlay_chan.split(",")
             for d in a.dips.split(",")
             for v in a.variants.split(",")]
    print(f"{len(tasks)} runs", flush=True)
    rows = []
    with mp.Pool(a.procs) as pool:
        for i, row in enumerate(pool.imap_unordered(run_one, tasks), 1):
            rows.append(row)
            if i % 20 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}", flush=True)

    import pandas as pd
    pd.DataFrame(rows).sort_values(["nlay_chan", "variant", "dip"]).to_csv(a.out, index=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
