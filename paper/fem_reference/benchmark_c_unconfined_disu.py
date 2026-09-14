"""Benchmark C (unconfined): MODFLOW 6 DISU vs the saturated free-surface FEM truth.

Layered DISU grid (one cell per geological layer), NPF icelltype=1 (convertible /
unconfined) with Newton.  Recharge on the top aquifer; basal toe discharge.  The
three formulations (plan / area / full) differ only in the vertical-connection
HWVA (the sec^2 fix rides on HWVA, since MF6 ignores CL12 for vertical flow).

Diagnostic: head drop across the dipping aquitard, dh = h(top aquifer) -
h(basal aquifer), compared with the FEM truth (benchmark_c_unconfined_fem2d).
"""
from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import flopy  # noqa: E402
from disu_slope_corrector.correction import DisuGeometry, apply_correction  # noqa: E402
import benchmark_c_unconfined_fem3d as femu  # noqa: E402  (full 3-D truth)

MF6 = os.environ.get("MF6_EXE", "mf6")

L = femu.L                       # 500 m, matches the FEM truth
W = 300.0
LAYERS = [(femu.T_TOP, femu.K_AQ), (femu.T_AQT, femu.K_AQT), (femu.T_BOT, femu.K_AQ)]
NLAY = 3
NCOL = 40
NROW = 8
R = femu.R
TOE_DRAWDOWN = femu.TOE_DRAWDOWN
_ZIFACE = np.cumsum([0.0] + [t for t, _ in LAYERS])


def build_disu_arrays(alpha_rad):
    ta = math.tan(alpha_rad)
    dx, dy = L / NCOL, W / NROW
    ncell = NLAY * NROW * NCOL

    def nid(lay, row, col):
        return lay * NROW * NCOL + row * NCOL + col

    xc = np.zeros(ncell); yc = np.zeros(ncell); zc = np.zeros(ncell)
    top = np.zeros(ncell); bot = np.zeros(ncell); area = np.full(ncell, dx * dy)
    kcell = np.zeros(ncell)
    for lay, (t_l, k_l) in enumerate(LAYERS):
        for row in range(NROW):
            for col in range(NCOL):
                xcen = (col + 0.5) * dx
                tp = -xcen * ta - _ZIFACE[lay]
                n = nid(lay, row, col)
                xc[n] = xcen; yc[n] = (row + 0.5) * dy
                top[n] = tp; bot[n] = tp - t_l; zc[n] = tp - 0.5 * t_l
                kcell[n] = k_l

    iac = np.zeros(ncell, dtype=int)
    ja = []; ihc = []; cl12 = []; hwva = []
    for lay, (t_l, _) in enumerate(LAYERS):
        for row in range(NROW):
            for col in range(NCOL):
                n = nid(lay, row, col)
                conns = [(n, 0, 0.0, 0.0)]
                if lay > 0:
                    conns.append((nid(lay - 1, row, col), 0, t_l / 2, dx * dy))
                if lay < NLAY - 1:
                    conns.append((nid(lay + 1, row, col), 0, t_l / 2, dx * dy))
                if col > 0:
                    conns.append((nid(lay, row, col - 1), 1, dx / 2, dy))
                if col < NCOL - 1:
                    conns.append((nid(lay, row, col + 1), 1, dx / 2, dy))
                if row > 0:
                    conns.append((nid(lay, row - 1, col), 1, dy / 2, dx))
                if row < NROW - 1:
                    conns.append((nid(lay, row + 1, col), 1, dy / 2, dx))
                conns = [conns[0]] + sorted(conns[1:], key=lambda c: c[0])
                iac[n] = len(conns)
                for (m, h, cl, hw) in conns:
                    ja.append(m); ihc.append(h); cl12.append(cl); hwva.append(hw)

    return dict(ncell=ncell, xc=xc, yc=yc, zc=zc, top=top, bot=bot, area=area,
                kcell=kcell, iac=iac, ja=np.array(ja, dtype=int),
                ihc=np.array(ihc, dtype=int), cl12=np.array(cl12, dtype=float),
                hwva=np.array(hwva, dtype=float))


def corrected_hwva(d):
    geom = DisuGeometry(xc=d["xc"], yc=d["yc"], zc=d["zc"], top=d["top"],
                        bot=d["bot"], area=d["area"], iac=d["iac"],
                        ja=d["ja"] + 1, ihc=d["ihc"], cl12=d["cl12"], hwva=d["hwva"])
    res = apply_correction(geom, neighbours=8)
    vert = d["ihc"] == 0
    sec = 1.0 / np.where(res.cos_alpha > 0, res.cos_alpha, 1.0)
    area = d["hwva"].copy(); area[vert] = d["hwva"][vert] * sec[vert]
    full = d["hwva"].copy(); full[vert] = d["hwva"][vert] * sec[vert] ** 2
    return {"plan": d["hwva"], "area": area, "full": full}


def run_mf6(d, hwva, alpha_rad, ws):
    ta = math.tan(alpha_rad)
    sim = flopy.mf6.MFSimulation(sim_name="c", sim_ws=str(ws), exe_name=MF6)
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1.0, 1, 1.0)])
    flopy.mf6.ModflowIms(sim, complexity="COMPLEX", outer_dvclose=1e-5,
                         inner_dvclose=1e-6, linear_acceleration="BICGSTAB",
                         outer_maximum=1000, inner_maximum=200,
                         under_relaxation="DBD", under_relaxation_theta=0.8,
                         under_relaxation_kappa=0.1, backtracking_number=20)
    gwf = flopy.mf6.ModflowGwf(sim, modelname="c", save_flows=True,
                               newtonoptions="NEWTON UNDER_RELAXATION")
    flopy.mf6.ModflowGwfdisu(gwf, nodes=d["ncell"], nja=d["ja"].size, top=d["top"],
                             bot=d["bot"], area=d["area"], iac=d["iac"], ja=d["ja"],
                             ihc=d["ihc"], cl12=d["cl12"], hwva=hwva)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=d["kcell"])
    strt = -d["xc"] * ta + 1.0          # start fully saturated; Newton draws down
    flopy.mf6.ModflowGwfic(gwf, strt=strt)

    def nid(lay, row, col):
        return lay * NROW * NCOL + row * NCOL + col
    # recharge on the top aquifer
    rch = [[(nid(0, row, col),), R] for row in range(NROW) for col in range(NCOL)]
    flopy.mf6.ModflowGwfrch(gwf, stress_period_data=rch)
    # basal toe discharge
    h_toe = -L * ta - TOE_DRAWDOWN
    chd = [[(nid(NLAY - 1, row, NCOL - 1),), h_toe] for row in range(NROW)]
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd, save_flows=True)
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="c.hds",
                           saverecord=[("HEAD", "ALL")])
    sim.write_simulation(silent=True)
    ok, _ = sim.run_simulation(silent=True)
    if not ok:
        raise RuntimeError(f"MF6 failed in {ws}")

    head = gwf.output.head().get_data().ravel()
    # head drop across the aquitard: mean h(top aquifer) - h(basal aquifer)
    dh = []
    for row in range(NROW):
        for col in range(NCOL):
            dh.append(head[nid(0, row, col)] - head[nid(NLAY - 1, row, col)])
    return float(np.mean(dh))


def main():
    work = HERE / "_cu_runs"
    work.mkdir(exist_ok=True)
    rows = []
    print(f"{'dip':>5} {'FEM dh':>9} {'plan':>9} {'area':>9} {'full':>9}   "
          f"{'plan err%':>9} {'full err%':>9}")
    print("-" * 74)
    for dip in (0, 15, 30, 45):
        a = math.radians(dip)
        dh_fem, _, _ = femu.solve_free_surface(a)
        d = build_disu_arrays(a)
        modes = corrected_hwva(d)
        dh = {m: run_mf6(d, hw, a, work / f"dip{dip}_{m}") for m, hw in modes.items()}
        ep = 100 * (dh["plan"] - dh_fem) / dh_fem
        ef = 100 * (dh["full"] - dh_fem) / dh_fem
        print(f"{dip:5d} {dh_fem:9.3f} {dh['plan']:9.3f} {dh['area']:9.3f} "
              f"{dh['full']:9.3f}   {ep:8.1f}% {ef:8.1f}%")
        rows.append(dict(dip=dip, fem=dh_fem, plan=dh["plan"], area=dh["area"],
                         full=dh["full"], plan_err=ep, full_err=ef))

    with (HERE / "benchmark_c_unconfined_disu_vs_fem.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dip_deg", "fem_dh", "plan_dh", "area_dh", "full_dh",
                    "plan_err_pct", "full_err_pct"])
        for r in rows:
            w.writerow([r["dip"], f"{r['fem']:.4f}", f"{r['plan']:.4f}",
                        f"{r['area']:.4f}", f"{r['full']:.4f}",
                        f"{r['plan_err']:.3f}", f"{r['full_err']:.3f}"])
    print("\nwrote benchmark_c_unconfined_disu_vs_fem.csv")


if __name__ == "__main__":
    main()
