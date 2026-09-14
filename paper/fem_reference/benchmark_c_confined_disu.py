"""Benchmark C (confined): MODFLOW 6 DISU vs the independent 3-D FEM truth.

Builds a layered (one-cell-per-geological-layer) DISU grid of the dipping
3-layer system with PLAN-VIEW connection geometry, then runs three formulations
through MODFLOW 6:

  * plan-view    : as-built (uncorrected) cl12 / hwva
  * area-only    : vertical hwva x sec(alpha)
  * full         : vertical hwva x sec(alpha)^2 (MF6 does not use cl12 for vertical connections)

The area-only and full vertical corrections come from the project's own
``disu_slope_corrector`` package (apply_correction).  Only the VERTICAL
connection arrays are touched: MF6 computes horizontal saturated thickness
internally from the (dipping) tops/bots, so the operative correction for an
MF6 DISU model is the vertical-conductance one, which is exactly the sec^2
claim Benchmark C tests.

Each formulation's cross-formational outlet discharge is compared against the
FEM truth from ``benchmark_c_confined_fem.py``.
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
PKG = HERE.parents[1] / "src"
sys.path.insert(0, str(PKG))

import flopy  # noqa: E402
from disu_slope_corrector.correction import DisuGeometry, apply_correction  # noqa: E402
import benchmark_c_confined_fem as femc  # noqa: E402

MF6 = os.environ.get("MF6_EXE", "mf6")

# geometry / properties -- identical to the FEM benchmark
LAYERS = femc.LAYERS          # [(40,5),(20,0.05),(40,5)]
L, W = femc.L, femc.W
H_IN, H_OUT = femc.H_IN, femc.H_OUT
NCOL = 40
NROW = 8   # 2-D plan spread so the local interface-plane fit is well posed
NLAY = len(LAYERS)
_ZIFACE = np.cumsum([0.0] + [t for t, _ in LAYERS])


def build_disu_arrays(alpha_rad):
    """Plan-view DISU arrays for the dipping layered grid."""
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
                zsurf = -xcen * ta
                tp = zsurf - _ZIFACE[lay]
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
                conns = [(n, 0, 0.0, 0.0)]  # self
                # vertical up / down (ihc=0): hwva = plan area, cl12 = t/2
                if lay > 0:
                    conns.append((nid(lay - 1, row, col), 0, t_l / 2, dx * dy))
                if lay < NLAY - 1:
                    conns.append((nid(lay + 1, row, col), 0, t_l / 2, dx * dy))
                # horizontal col-direction (ihc=1): hwva = width dy, cl12 = dx/2
                if col > 0:
                    conns.append((nid(lay, row, col - 1), 1, dx / 2, dy))
                if col < NCOL - 1:
                    conns.append((nid(lay, row, col + 1), 1, dx / 2, dy))
                # horizontal row-direction: hwva = width dx, cl12 = dy/2
                if row > 0:
                    conns.append((nid(lay, row - 1, col), 1, dy / 2, dx))
                if row < NROW - 1:
                    conns.append((nid(lay, row + 1, col), 1, dy / 2, dx))
                # MF6 requires each row's neighbours sorted ascending (self first)
                conns = [conns[0]] + sorted(conns[1:], key=lambda c: c[0])
                iac[n] = len(conns)
                for (m, h, cl, hw) in conns:
                    ja.append(m)   # 0-based; FloPy writes 1-based to the file
                    ihc.append(h); cl12.append(cl); hwva.append(hw)

    return dict(
        ncell=ncell, xc=xc, yc=yc, zc=zc, top=top, bot=bot, area=area,
        kcell=kcell, iac=iac,
        ja=np.array(ja, dtype=int), ihc=np.array(ihc, dtype=int),
        cl12=np.array(cl12, dtype=float), hwva=np.array(hwva, dtype=float),
    )


def corrected_arrays(d):
    """Return (cl12, hwva) for plan / area / full, isolating the vertical fix.

    IMPORTANT MF6 detail (verified by a 2-cell test): MODFLOW 6 derives vertical
    conductance from the cell TOP/BOT half-thicknesses and HWVA, and IGNORES
    CL12 for vertical connections.  The corrector's cl12 x cos(alpha) length fix
    is therefore inert in MF6.  Vertical conductance is proportional to HWVA, so
    we inject the correction through HWVA:
        area-only : HWVA x sec(alpha)       -> conductance x sec
        full      : HWVA x sec(alpha)^2     -> conductance x sec^2
    cos(alpha) per connection comes from the project's own corrector.
    """
    geom = DisuGeometry(
        xc=d["xc"], yc=d["yc"], zc=d["zc"], top=d["top"], bot=d["bot"],
        area=d["area"], iac=d["iac"], ja=d["ja"] + 1, ihc=d["ihc"],  # corrector wants 1-based ja
        cl12=d["cl12"], hwva=d["hwva"],
    )
    res = apply_correction(geom, neighbours=8)
    vert = d["ihc"] == 0
    cos = np.where(res.cos_alpha > 0, res.cos_alpha, 1.0)
    sec = 1.0 / cos
    hwva_area = d["hwva"].copy(); hwva_area[vert] = d["hwva"][vert] * sec[vert]
    hwva_full = d["hwva"].copy(); hwva_full[vert] = d["hwva"][vert] * sec[vert] ** 2
    return {
        "plan": (d["cl12"], d["hwva"]),
        "area": (d["cl12"], hwva_area),
        "full": (d["cl12"], hwva_full),
    }


def run_mf6(d, cl12, hwva, ws):
    sim = flopy.mf6.MFSimulation(sim_name="c", sim_ws=str(ws), exe_name=MF6)
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1.0, 1, 1.0)])
    flopy.mf6.ModflowIms(sim, complexity="SIMPLE", outer_dvclose=1e-9,
                         inner_dvclose=1e-10, linear_acceleration="BICGSTAB")
    gwf = flopy.mf6.ModflowGwf(sim, modelname="c", save_flows=True)
    flopy.mf6.ModflowGwfdisu(
        gwf, nodes=d["ncell"], nja=d["ja"].size, top=d["top"], bot=d["bot"],
        area=d["area"], iac=d["iac"], ja=d["ja"], ihc=d["ihc"],
        cl12=cl12, hwva=hwva,
    )
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=d["kcell"])
    flopy.mf6.ModflowGwfic(gwf, strt=H_IN)

    # CHD: inflow on whole up-dip face (col 0, all layers); outflow basal toe.
    def nid(lay, row, col):
        return lay * NROW * NCOL + row * NCOL + col
    chd = []
    in_nodes, out_nodes = [], []
    for row in range(NROW):                       # inflow: TOP aquifer, up-dip
        n = nid(0, row, 0)
        chd.append([(n,), H_IN]); in_nodes.append(n)
    for row in range(NROW):                       # outflow: BASAL aquifer, toe
        n = nid(NLAY - 1, row, NCOL - 1)
        chd.append([(n,), H_OUT]); out_nodes.append(n)
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd, save_flows=True)
    flopy.mf6.ModflowGwfoc(
        gwf, budget_filerecord="c.cbc", head_filerecord="c.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    sim.write_simulation(silent=True)
    ok, _ = sim.run_simulation(silent=True)
    if not ok:
        raise RuntimeError(f"MF6 failed in {ws}")

    bud = gwf.output.budget()
    chd_rec = bud.get_data(text="CHD")[-1]
    nodes = np.asarray(chd_rec["node"]) - 1
    q = np.asarray(chd_rec["q"])
    out_set = set(out_nodes)
    q_out = float(q[np.isin(nodes, list(out_set))].sum())
    return q_out


def main():
    dips = (0, 15, 30, 45, 60)
    work = HERE / "_c_runs"
    work.mkdir(exist_ok=True)
    rows = []
    print(f"{'dip':>5} {'FEM truth':>11} {'plan-view':>11} {'area-only':>11} "
          f"{'full':>11}   {'plan err%':>9} {'full err%':>9}")
    print("-" * 84)
    for dip in dips:
        a = math.radians(dip)
        _, _, q_fem = femc.solve_head(a)
        d = build_disu_arrays(a)
        modes = corrected_arrays(d)
        q = {}
        for mode, (cl, hw) in modes.items():
            q[mode] = run_mf6(d, cl, hw, work / f"dip{dip}_{mode}")
        # use magnitudes; FEM q_out is the outlet reaction (negative)
        f = abs(q_fem)
        ep = 100 * (abs(q["plan"]) - f) / f
        ef = 100 * (abs(q["full"]) - f) / f
        print(f"{dip:5d} {f:11.2f} {abs(q['plan']):11.2f} {abs(q['area']):11.2f} "
              f"{abs(q['full']):11.2f}   {ep:8.1f}% {ef:8.1f}%")
        rows.append(dict(dip=dip, fem=f, plan=abs(q["plan"]),
                         area=abs(q["area"]), full=abs(q["full"]),
                         plan_err_pct=ep, full_err_pct=ef))

    csv_path = HERE / "benchmark_c_confined_disu_vs_fem.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dip_deg", "fem_q", "plan_q", "area_q", "full_q",
                    "plan_err_pct", "full_err_pct"])
        for r in rows:
            w.writerow([r["dip"], f"{r['fem']:.4f}", f"{r['plan']:.4f}",
                        f"{r['area']:.4f}", f"{r['full']:.4f}",
                        f"{r['plan_err_pct']:.3f}", f"{r['full_err_pct']:.3f}"])
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
