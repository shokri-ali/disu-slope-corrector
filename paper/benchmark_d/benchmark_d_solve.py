"""Benchmark D (confined): FEM truth vs MODFLOW DISU on the real tilted catchment.

Conceptual model (isolates the vertical aquitard-conductance correction, as in
Case C): both aquifers are well-mixed (one head per column per layer, matching a
one-cell-per-layer DISU cell); the tight aquitard is isotropic so its leakance
follows the tilted geometry.  A regional head is held in the deep aquifer; the
river drains the top aquifer.  Water leaks up through the dipping aquitard to the
river, so the river baseflow is set by the aquitard vertical conductance -- the
sec^2 quantity the correction targets.

Run over the tilt sweep (0/15/30/45 deg): plan-view DISU should under-predict
baseflow more and more as the catchment steepens; the full slope-aware DISU
should track the FEM truth.
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

import benchmark_d_mesh as bm

# physical parameters
KH = 100.0          # aquifer horizontal K [m/d] (transmissive shallow system,
                    #   so river head controls the top aquifer and the
                    #   diagnostic is the aquitard leakage, not lateral routing)
KZZ_AQ = 1000.0     # aquifer vertical K (well-mixed)
K_AQT = 5.0e-4      # aquitard isotropic K (tight)
H_BOT = 100.0       # regional head in the deep aquifer
H_RIV = 90.0        # river stage (drains the top aquifer)

NZP = None          # set at runtime


def _geol_of_klevel(k):
    return 0 if k < bm.CELLS[0] else (1 if k < bm.CELLS[0] + bm.CELLS[1] else 2)


def river_columns(cm):
    """Thalweg columns: lowest cell per down-river band (from untilted relief)."""
    x, y, z = bm._load_grid()
    e_down, outlet = bm._downstream_frame(x, y, z)
    u = (cm.gx - outlet[0]) * e_down[0] + (cm.gy - outlet[1]) * e_down[1]
    # untilted surface elevation per column (tilt 0 build shares column order)
    z0 = bm.build(0.0).zsurf
    bins = np.floor((u - u.min()) / cm.dxy).astype(int)
    riv = []
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        riv.append(idx[np.argmin(z0[idx])])
    return np.array(sorted(riv))


# ---------------------------------------------------------------------------
# FEM truth
# ---------------------------------------------------------------------------
def solve_fem(cm, riv_cols):
    from skfem import (Basis, ElementTetP1, ElementTetP0, asm, condense, solve,
                       BilinearForm)
    from skfem.helpers import grad
    mesh, cell2corners, nzp = bm.build_fem_corner(cm)
    elem = ElementTetP1()
    basis = Basis(mesh, elem)
    basis0 = basis.with_element(ElementTetP0())

    klevel = (mesh.t % nzp).min(axis=0)
    geol = np.array([_geol_of_klevel(k) for k in klevel])
    kxx = np.where(geol == 1, K_AQT, KH)
    kzz = np.where(geol == 1, K_AQT, KZZ_AQ)

    @BilinearForm
    def form(u, v, w):
        du, dv = grad(u), grad(v)
        return (w["kx"] * du[0] * dv[0] + w["kx"] * du[1] * dv[1]
                + w["kz"] * du[2] * dv[2])

    A = asm(form, basis, kx=basis0.interpolate(kxx), kz=basis0.interpolate(kzz))

    node_k = np.arange(basis.N) % nzp
    node_corner = np.arange(basis.N) // nzp
    base = node_k == (nzp - 1)                       # deep reservoir surface
    river_corners = set()
    for c in riv_cols:
        river_corners.update(cell2corners[int(c)])
    riv_mask = np.isin(node_corner, list(river_corners)) & (node_k <= bm.CELLS[0])
    x = np.full(basis.N, np.nan)
    x[base] = H_BOT
    x[riv_mask] = H_RIV
    riv_dofs = np.where(riv_mask)[0]
    D = np.where(~np.isnan(x))[0]
    x[np.isnan(x)] = 0.0

    h = solve(*condense(A, np.zeros(basis.N), x=x, D=D))
    baseflow = float((A @ h)[riv_dofs].sum())        # flux into the river [m^3/d]
    return baseflow


# ---------------------------------------------------------------------------
# MODFLOW DISU (plan / area / full)
# ---------------------------------------------------------------------------
def build_disu_arrays(cm, active):
    # contiguous re-index of the active columns only (same footprint as the FEM)
    active = np.asarray(active)
    remap = {int(old): new for new, old in enumerate(active)}
    ncol = active.size
    ncell = ncol * 3
    dxy = cm.dxy
    area = dxy * dxy
    look = {(int(cm.col_ij[old][0]), int(cm.col_ij[old][1])): remap[int(old)]
            for old in active}
    gx = cm.gx[active]; gy = cm.gy[active]; ztop = cm.ztop[active]
    col_rc = cm.col_ij[active]

    def cid(c, li):
        return c * 3 + li

    xc = np.repeat(gx, 3); yc = np.repeat(gy, 3)
    top = np.empty(ncell); bot = np.empty(ncell); zc = np.empty(ncell)
    kh = np.empty(ncell); k33 = np.empty(ncell)
    for c in range(ncol):
        for li in range(3):
            n = cid(c, li)
            top[n] = ztop[c, li]; bot[n] = ztop[c, li + 1]
            zc[n] = 0.5 * (top[n] + bot[n])
            kh[n] = K_AQT if li == 1 else KH
            k33[n] = K_AQT if li == 1 else KZZ_AQ

    iac = np.zeros(ncell, dtype=int)
    ja = []; ihc = []; cl12 = []; hwva = []
    for c in range(ncol):
        r, col = int(col_rc[c][0]), int(col_rc[c][1])
        for li in range(3):
            n = cid(c, li)
            t = top[n] - bot[n]
            conns = [(n, 0, 0.0, 0.0)]
            if li > 0:
                conns.append((cid(c, li - 1), 0, t / 2, area))
            if li < 2:
                conns.append((cid(c, li + 1), 0, t / 2, area))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cc = look.get((r + dr, col + dc))
                if cc is None:
                    continue
                m = cid(cc, li)
                ov = min(top[n], top[m]) - max(bot[n], bot[m])
                if ov <= 0:
                    continue
                conns.append((m, 1, dxy / 2, dxy * ov))
            conns = [conns[0]] + sorted(conns[1:], key=lambda z: z[0])
            iac[n] = len(conns)
            for (m, h, cl, hw) in conns:
                ja.append(m); ihc.append(h); cl12.append(cl); hwva.append(hw)

    return dict(ncell=ncell, xc=xc, yc=yc, zc=zc, top=top, bot=bot,
                area=np.full(ncell, area), kh=kh, k33=k33, iac=iac,
                ja=np.array(ja, int), ihc=np.array(ihc, int),
                cl12=np.array(cl12, float), hwva=np.array(hwva, float),
                remap=remap)


def corrected_hwva(d):
    from disu_slope_corrector.correction import DisuGeometry, apply_correction
    geom = DisuGeometry(xc=d["xc"], yc=d["yc"], zc=d["zc"], top=d["top"],
                        bot=d["bot"], area=d["area"], iac=d["iac"],
                        ja=d["ja"] + 1, ihc=d["ihc"], cl12=d["cl12"], hwva=d["hwva"])
    res = apply_correction(geom, neighbours=8, max_dip_deg=89.0, code="mf6")
    vert = d["ihc"] == 0
    sec = 1.0 / np.where(res.cos_alpha > 0, res.cos_alpha, 1.0)
    area = d["hwva"].copy(); area[vert] = d["hwva"][vert] * sec[vert]
    full = d["hwva"].copy(); full[vert] = d["hwva"][vert] * sec[vert] ** 2
    return {"plan": d["hwva"], "area": area, "full": full}


def run_disu(d, hwva, riv_cols, ws):
    import flopy
    mf6 = os.environ.get("MF6_EXE", "mf6")

    def cid(c, li):
        return c * 3 + li

    sim = flopy.mf6.MFSimulation(sim_name="d", sim_ws=str(ws), exe_name=mf6)
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1.0, 1, 1.0)])
    flopy.mf6.ModflowIms(sim, complexity="SIMPLE", outer_dvclose=1e-8,
                         inner_dvclose=1e-9, linear_acceleration="BICGSTAB")
    gwf = flopy.mf6.ModflowGwf(sim, modelname="d", save_flows=True)
    flopy.mf6.ModflowGwfdisu(gwf, nodes=d["ncell"], nja=d["ja"].size, top=d["top"],
                             bot=d["bot"], area=d["area"], iac=d["iac"], ja=d["ja"],
                             ihc=d["ihc"], cl12=d["cl12"], hwva=hwva)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=d["kh"], k33=d["k33"])
    flopy.mf6.ModflowGwfic(gwf, strt=H_BOT)
    chd = []
    riv_nodes = []
    nset = {d["remap"][int(c)] for c in riv_cols if int(c) in d["remap"]}
    for c in range(d["ncell"] // 3):
        chd.append([(cid(c, 2),), H_BOT])              # deep reservoir
        if c in nset:
            chd.append([(cid(c, 0),), H_RIV]); riv_nodes.append(cid(c, 0))
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd, save_flows=True)
    flopy.mf6.ModflowGwfoc(gwf, budget_filerecord="d.cbc",
                           saverecord=[("BUDGET", "ALL")])
    sim.write_simulation(silent=True)
    ok, _ = sim.run_simulation(silent=True)
    if not ok:
        raise RuntimeError(f"MF6 failed in {ws}")
    rec = gwf.output.budget().get_data(text="CHD")[-1]
    nodes = np.asarray(rec["node"]) - 1
    q = np.asarray(rec["q"])
    return float(q[np.isin(nodes, riv_nodes)].sum())


def main():
    tilts = (0, 15, 30, 45)
    work = HERE / "_d_solve"; work.mkdir(exist_ok=True)
    cm0 = bm.build(0.0)
    active = np.arange(cm0.gx.size)          # all valid columns (FEM keeps all too)
    riv = river_columns(cm0)
    print(f"columns: {active.size}; river: {riv.size}")
    rows = []
    print(f"\n{'tilt':>5} {'FEM':>10} {'plan':>10} {'area':>10} {'full':>10}  "
          f"{'plan err%':>9} {'full err%':>9}")
    print("-" * 74)
    for th in tilts:
        cm = bm.build(float(th))
        q_fem = abs(solve_fem(cm, riv))
        d = build_disu_arrays(cm, active)
        modes = corrected_hwva(d)
        q = {m: abs(run_disu(d, hw, riv, work / f"t{th}_{m}"))
             for m, hw in modes.items()}
        ep = 100 * (q["plan"] - q_fem) / q_fem
        ef = 100 * (q["full"] - q_fem) / q_fem
        print(f"{th:5d} {q_fem:10.1f} {q['plan']:10.1f} {q['area']:10.1f} "
              f"{q['full']:10.1f}  {ep:8.1f}% {ef:8.1f}%")
        rows.append(dict(tilt=th, fem=q_fem, plan=q["plan"], area=q["area"],
                         full=q["full"], plan_err=ep, full_err=ef))
    with (HERE / "benchmark_d_baseflow_vs_tilt.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tilt_deg", "fem", "plan", "area", "full",
                    "plan_err_pct", "full_err_pct"])
        for r in rows:
            w.writerow([r["tilt"], f"{r['fem']:.3f}", f"{r['plan']:.3f}",
                        f"{r['area']:.3f}", f"{r['full']:.3f}",
                        f"{r['plan_err']:.3f}", f"{r['full_err']:.3f}"])
    print("\nwrote benchmark_d_baseflow_vs_tilt.csv")


if __name__ == "__main__":
    main()
