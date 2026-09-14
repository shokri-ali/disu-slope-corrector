"""catchment demonstration (HEAD-DRIVEN, confined leakage) on the real steep topo.

Why this exists
---------------
The recharge-driven `catchment_solve.py` cannot show the slope correction: in steady
state the river baseflow equals the recharge by mass balance, so FEM / plan /
area / full all report the same flux and the only thing left to compare is the
fragile per-column head drop `dh` -- which is further contaminated by
desaturation of the steep walls (water table to -162 m).  The result is that the
"full" correction looks WORSE than plan (see _catchment.log).

This script ports the *validated* mechanism from `benchmark_d_solve.py` onto the
real catchment window, dropping the three things that hid the signal:

  * driver       : recharge  ->  a held head difference (deep regional head vs
                   river stage).  Baseflow is then leakage, set by the aquitard
                   vertical conductance == the sec^2(alpha) quantity.
  * diagnostic   : per-column `dh`  ->  river baseflow, an integral flux measured
                   identically (sum of flux into the river) in FEM and DISU.
  * saturation   : phreatic / dewatering  ->  confined (icelltype=0), well-mixed
                   high-K aquifers + tight aquitard, so the aquitard stays
                   saturated everywhere and the cos^2 law holds cleanly.

Everything else is kept: real catchment topography, naturally steep dip, large
sec^2(alpha), FEM corner grid as truth.  The FEM/DISU machinery is imported
unchanged from `benchmark_d_solve` (it is geometry-agnostic).

Outputs
-------
  * a baseflow table  FEM vs plan / area / full  (headline result)
  * per-column aquitard leakage vs local cos^2(alpha)  ->  catchment_leakage_vs_dip.csv
    and figure fig_catchment_leakage_vs_dip.png
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import catchment_geom                              # sets bm layers 350/50/100, CELLS=[7,2,2]
import benchmark_d_mesh as bm
import benchmark_d_solve as bd               # confined, head-driven machinery
from benchmark_d_solve import (build_disu_arrays, corrected_hwva, solve_fem,
                               H_BOT, H_RIV, KH, KZZ_AQ, K_AQT)


# ---------------------------------------------------------------------------
# River channel by D8 flow routing on the window DEM.  Priority-flood fill
# (Barnes 2014) removes pits so every cell drains to the boundary; D8 then gives
# each cell a steepest-descent receiver; flow accumulation picks out the valley
# network (cells draining more than `accum_frac` of the catchment).  This
# follows the real channel, unlike the old plane-fit thalweg.
# ---------------------------------------------------------------------------
_NB = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def _grid(cm):
    xs = np.unique(cm.gx); ys = np.unique(cm.gy)
    ny, nx = ys.size, xs.size
    Z = np.full((ny, nx), np.nan)
    col_of = np.full((ny, nx), -1, int)
    ix = np.searchsorted(xs, cm.gx); iy = np.searchsorted(ys, cm.gy)
    Z[iy, ix] = cm.zsurf
    col_of[iy, ix] = np.arange(cm.gx.size)
    return xs, ys, Z, col_of


def river_columns_d8(cm, accum_frac=0.012):
    import heapq
    xs, ys, Z, col_of = _grid(cm)
    ny, nx = Z.shape
    valid = np.isfinite(Z)

    # priority-flood depression fill -> `filled`
    filled = np.where(valid, Z, np.inf)
    done = np.zeros((ny, nx), bool)
    heap = []
    for r in range(ny):
        for c in range(nx):
            if valid[r, c] and (r in (0, ny - 1) or c in (0, nx - 1)):
                heapq.heappush(heap, (Z[r, c], r, c)); done[r, c] = True
    while heap:
        z, r, c = heapq.heappop(heap)
        for dr, dc in _NB:
            rr, cc = r + dr, c + dc
            if 0 <= rr < ny and 0 <= cc < nx and valid[rr, cc] and not done[rr, cc]:
                filled[rr, cc] = max(Z[rr, cc], z)
                done[rr, cc] = True
                heapq.heappush(heap, (filled[rr, cc], rr, cc))

    # D8 steepest-descent receiver on the filled surface
    recv = np.full((ny, nx, 2), -1, int)
    for r in range(ny):
        for c in range(nx):
            if not valid[r, c]:
                continue
            best, best_slope = None, 0.0
            for dr, dc in _NB:
                rr, cc = r + dr, c + dc
                if 0 <= rr < ny and 0 <= cc < nx and valid[rr, cc]:
                    s = (filled[r, c] - filled[rr, cc]) / ((dr * dr + dc * dc) ** 0.5
                                                           * cm.dxy)
                    if s > best_slope:
                        best_slope, best = s, (rr, cc)
            if best is not None:
                recv[r, c] = best

    # flow accumulation: add each cell's area to its receiver, high elevation first
    acc = np.ones((ny, nx))
    cells = sorted(((filled[r, c], r, c) for r in range(ny) for c in range(nx)
                    if valid[r, c]), key=lambda t: -t[0])
    for _z, r, c in cells:
        rr, cc = recv[r, c]
        if rr >= 0:
            acc[rr, cc] += acc[r, c]

    thr = accum_frac * valid.sum()
    riv_mask = valid & (acc >= thr)
    return np.array(sorted(col_of[riv_mask].tolist()))


# ---------------------------------------------------------------------------
# DISU run that also returns the per-column vertical leakage through the
# aquitard (flow between the lower aquifer cell and the aquitard cell).
# Adapted from benchmark_d_solve.run_disu so the working file stays untouched.
# ---------------------------------------------------------------------------
def run_disu_leaky(d, hwva, riv_cols, ws):
    import flopy
    mf6 = os.environ.get("MF6_EXE", "mf6")
    ncol = d["ncell"] // 3

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
    for c in range(ncol):
        chd.append([(cid(c, 2),), H_BOT])              # deep regional reservoir
        if c in nset:
            chd.append([(cid(c, 0),), H_RIV]); riv_nodes.append(cid(c, 0))
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd, save_flows=True)
    flopy.mf6.ModflowGwfoc(gwf, budget_filerecord="d.cbc",
                           saverecord=[("BUDGET", "ALL")])
    sim.write_simulation(silent=True)
    ok, _ = sim.run_simulation(silent=True)
    if not ok:
        raise RuntimeError(f"MF6 failed in {ws}")

    bud = gwf.output.budget()
    rec = bud.get_data(text="CHD")[-1]
    nodes = np.asarray(rec["node"]) - 1
    q = np.asarray(rec["q"])
    baseflow = float(q[np.isin(nodes, riv_nodes)].sum())

    # per-column vertical leakage: flow on the face between aquitard (li=1) and
    # lower aquifer (li=2), read from FLOW-JA-FACE.
    fja = bud.get_data(text="FLOW-JA-FACE")[-1].squeeze()
    ja = d["ja"]; iac = d["iac"]
    rowstart = np.concatenate([[0], np.cumsum(iac)])
    leak = np.full(ncol, np.nan)
    for c in range(ncol):
        n = cid(c, 1)                                  # aquitard cell
        m = cid(c, 2)                                  # lower aquifer cell
        for p in range(rowstart[n], rowstart[n + 1]):
            if ja[p] == m:
                leak[c] = abs(float(fja[p]))
                break
    return baseflow, leak


# ---------------------------------------------------------------------------
# Per-column local dip (cos alpha) of the aquitard, straight from the corrector.
# ---------------------------------------------------------------------------
def column_cos_alpha(d):
    from disu_slope_corrector.correction import DisuGeometry, apply_correction
    geom = DisuGeometry(xc=d["xc"], yc=d["yc"], zc=d["zc"], top=d["top"],
                        bot=d["bot"], area=d["area"], iac=d["iac"],
                        ja=d["ja"] + 1, ihc=d["ihc"], cl12=d["cl12"], hwva=d["hwva"])
    res = apply_correction(geom, neighbours=8, max_dip_deg=89.0, code="mf6")
    ncol = d["ncell"] // 3
    iac = d["iac"]; ja = d["ja"]
    rowstart = np.concatenate([[0], np.cumsum(iac)])
    cosa = np.full(ncol, np.nan)
    for c in range(ncol):
        n = c * 3 + 1; m = c * 3 + 2
        for p in range(rowstart[n], rowstart[n + 1]):
            if ja[p] == m:
                cosa[c] = res.cos_alpha[p]
                break
    return cosa


# ---------------------------------------------------------------------------
# Resolution-matched DISU: refine each geological layer into the SAME sub-layers
# as the FEM (bm.CELLS = [7,2,2] -> 11 cells/column), so DISU and FEM share the
# vertical discretisation and the ONLY remaining difference is the connection
# geometry (the slope correction).  A "fairer" companion to the 1-cell-per-layer
# benchmark.
# ---------------------------------------------------------------------------
def build_disu_sub(cm, active):
    active = np.asarray(active)
    remap = {int(o): n for n, o in enumerate(active)}
    ncol = active.size
    depths = bm._node_depths()                    # (nsub+1,)
    nsub = depths.size - 1
    cells = bm.CELLS
    geol = [0] * cells[0] + [1] * cells[1] + [2] * cells[2]
    dxy = cm.dxy; area = dxy * dxy
    look = {(int(cm.col_ij[o][0]), int(cm.col_ij[o][1])): remap[int(o)] for o in active}
    gx = cm.gx[active]; gy = cm.gy[active]; zsurf = cm.zsurf[active]
    col_rc = cm.col_ij[active]
    iface = zsurf[:, None] - depths[None, :]      # (ncol, nsub+1)
    ncell = ncol * nsub

    def cid(c, li):
        return c * nsub + li

    xc = np.repeat(gx, nsub); yc = np.repeat(gy, nsub)
    top = np.empty(ncell); bot = np.empty(ncell); zc = np.empty(ncell)
    kh = np.empty(ncell); k33 = np.empty(ncell)
    for c in range(ncol):
        for li in range(nsub):
            n = cid(c, li)
            top[n] = iface[c, li]; bot[n] = iface[c, li + 1]
            zc[n] = 0.5 * (top[n] + bot[n])
            kv = K_AQT if geol[li] == 1 else KH
            kz = K_AQT if geol[li] == 1 else KZZ_AQ
            kh[n] = kv; k33[n] = kz

    iac = np.zeros(ncell, dtype=int)
    ja = []; ihc = []; cl12 = []; hwva = []
    for c in range(ncol):
        r, col = int(col_rc[c][0]), int(col_rc[c][1])
        for li in range(nsub):
            n = cid(c, li); t = top[n] - bot[n]
            conns = [(n, 0, 0.0, 0.0)]
            if li > 0:
                conns.append((cid(c, li - 1), 0, t / 2, area))
            if li < nsub - 1:
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
                remap=remap, nsub=nsub)


def run_disu_leaky_gen(d, hwva, riv_cols, ws, leak_pair):
    """Generalised run for nsub cells/column; leak_pair = (li_aquitard, li_below)."""
    import flopy
    mf6 = os.environ.get("MF6_EXE", "mf6")
    nsub = d["nsub"]; ncol = d["ncell"] // nsub

    def cid(c, li):
        return c * nsub + li

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
    chd = []; riv_nodes = []
    nset = {d["remap"][int(c)] for c in riv_cols if int(c) in d["remap"]}
    for c in range(ncol):
        chd.append([(cid(c, nsub - 1),), H_BOT])       # deep reservoir (bottom cell)
        if c in nset:
            chd.append([(cid(c, 0),), H_RIV]); riv_nodes.append(cid(c, 0))
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd, save_flows=True)
    flopy.mf6.ModflowGwfoc(gwf, budget_filerecord="d.cbc",
                           saverecord=[("BUDGET", "ALL")])
    sim.write_simulation(silent=True)
    ok, _ = sim.run_simulation(silent=True)
    if not ok:
        raise RuntimeError(f"MF6 failed in {ws}")
    bud = gwf.output.budget()
    rec = bud.get_data(text="CHD")[-1]
    nodes = np.asarray(rec["node"]) - 1; q = np.asarray(rec["q"])
    baseflow = float(q[np.isin(nodes, riv_nodes)].sum())
    fja = bud.get_data(text="FLOW-JA-FACE")[-1].squeeze()
    ja = d["ja"]; rowstart = np.concatenate([[0], np.cumsum(d["iac"])])
    leak = np.full(ncol, np.nan)
    la, lb = leak_pair
    for c in range(ncol):
        n, m = cid(c, la), cid(c, lb)
        for p in range(rowstart[n], rowstart[n + 1]):
            if ja[p] == m:
                leak[c] = abs(float(fja[p])); break
    return baseflow, leak


def main():
    work = HERE / "_catchment_leaky"; work.mkdir(exist_ok=True)
    cm = catchment_geom.build_catchment(50.0)
    active = np.arange(cm.gx.size)
    riv = river_columns_d8(cm)
    ground = cm.zsurf

    # window dip stats (the aquitard is topo-parallel, so this is its dip)
    Z = np.full((cm.ny, cm.nx), np.nan)
    Z[cm.col_ij[:, 0], cm.col_ij[:, 1]] = cm.zsurf
    gyv, gxv = np.gradient(Z, cm.dxy, cm.dxy)
    slope = np.degrees(np.arctan(np.hypot(gxv, gyv)))
    sl = slope[np.isfinite(slope)]
    print(f"cells {cm.gx.size}; river/thalweg cells {riv.size}; "
          f"topo relief {np.ptp(ground):.0f} m")
    print(f"aquitard dip deg: median {np.median(sl):.1f}, 95th "
          f"{np.percentile(sl,95):.1f}, max {sl.max():.1f}  "
          f"(sec^2 median {1/np.cos(np.radians(np.median(sl)))**2:.2f}, "
          f"95th {1/np.cos(np.radians(np.percentile(sl,95)))**2:.2f})")
    print(f"driver: deep head {H_BOT} m vs river stage {H_RIV} m (confined); "
          f"KH={KH} Kzz={KZZ_AQ} aquitard={K_AQT} m/d\n")

    q_fem = abs(solve_fem(cm, riv))

    # (1) one cell per geological layer (the headline coarse benchmark)
    d = build_disu_arrays(cm, active)
    modes = corrected_hwva(d)
    res = {m: run_disu_leaky(d, hw, riv, work / m) for m, hw in modes.items()}
    cosa = column_cos_alpha(d)

    # (2) resolution-matched DISU: same sub-layers as the FEM (bm.CELLS)
    d2 = build_disu_sub(cm, active)
    cells = bm.CELLS
    leak_pair = (cells[0] + cells[1] - 1, cells[0] + cells[1])   # aquitard->lower
    modes2 = corrected_hwva(d2)
    res2 = {m: run_disu_leaky_gen(d2, hw, riv, work / ("ref_" + m), leak_pair)
            for m, hw in modes2.items()}

    print(f"{'':>26}{'baseflow':>10}{'err vs FEM':>12}")
    print(f"{'FEM truth':>26}{q_fem:10.2f}{'--':>12}")
    for tag, rr in (("coarse (1 cell/layer)", res), ("refined (FEM sub-layers)", res2)):
        for m in ("plan", "area", "full"):
            qm = abs(rr[m][0])
            print(f"{m+' DISU, '+tag:>26}"[:26] + f"{qm:10.2f}"
                  f"{100*(qm-q_fem)/q_fem:11.1f}%")

    # baseflow summary (both resolutions) for the results figure (reproducible)
    bf = HERE / "catchment_baseflow.csv"
    with bf.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "resolution", "baseflow_m3d", "err_pct_vs_fem"])
        w.writerow(["fem", "-", f"{q_fem:.3f}", "0.0"])
        for tag, rr in (("coarse", res), ("refined", res2)):
            for m in ("plan", "area", "full"):
                qm = abs(rr[m][0])
                w.writerow([m, tag, f"{qm:.3f}", f"{100*(qm-q_fem)/q_fem:.2f}"])
    print(f"wrote {bf.name}")

    # per-column leakage vs local cos^2(alpha)
    leak_plan = res["plan"][1]; leak_full = res["full"][1]
    sec2 = 1.0 / np.where(cosa > 0, cosa, np.nan) ** 2
    out = HERE / "catchment_leakage_vs_dip.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["col", "cos_alpha", "sec2_alpha", "leak_plan", "leak_area",
                    "leak_full"])
        for c in range(cm.gx.size):
            w.writerow([c, f"{cosa[c]:.4f}", f"{sec2[c]:.4f}",
                        f"{leak_plan[c]:.5f}", f"{res['area'][1][c]:.5f}",
                        f"{leak_full[c]:.5f}"])
    print(f"\nwrote {out.name}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ok = np.isfinite(sec2) & np.isfinite(leak_plan) & (leak_plan > 0)
        ratio = leak_full[ok] / leak_plan[ok]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
        ax[0].scatter(sec2[ok], leak_plan[ok], s=8, alpha=.4, label="plan")
        ax[0].scatter(sec2[ok], res['area'][1][ok], s=8, alpha=.4, label="area")
        ax[0].scatter(sec2[ok], leak_full[ok], s=8, alpha=.4, label="full")
        ax[0].set_xlabel(r"$\sec^2\alpha$ (local aquitard dip)")
        ax[0].set_ylabel("per-column leakage [m$^3$/d]")
        ax[0].set_title("Aquitard leakage vs local dip"); ax[0].legend(fontsize=8)
        ax[1].scatter(sec2[ok], ratio, s=8, alpha=.4)
        lim = np.linspace(1, np.nanmax(sec2[ok]), 50)
        ax[1].plot(lim, lim, "k--", lw=1, label=r"$y=\sec^2\alpha$")
        ax[1].set_xlabel(r"$\sec^2\alpha$"); ax[1].set_ylabel("full / plan leakage")
        ax[1].set_title("Correction factor vs geometry"); ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(HERE / "fig_catchment_leakage_vs_dip.png", dpi=160)
        print("wrote fig_catchment_leakage_vs_dip.png")
    except Exception as e:                              # noqa: BLE001
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
