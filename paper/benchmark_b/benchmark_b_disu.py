"""D1: Benchmark B rebuilt -- two-aquifer leaky system on a layered DISU grid.

Replaces the lost mesh3d-prototype pipeline. Set-up follows Section 2.6 of the
manuscript: two confined layers of vertical thickness b = 50 m, plan domain
L = 500 m by W = 200 m, dipping at alpha in +x; Kh = 5 m/d, Kv = 0.1 m/d;
CHD h1(0)=110, h1(L)=100, h2(0)=105, h2(L)=102; all other faces no-flow.

Grid: structured layered DISU, NCOL = 51 columns with centroids at x = 0, dx, ..., L
(dx = 10 m) so the CHD columns sit exactly on x = 0 and x = L, NROW = 4 (dy = 50 m)
so the local plane fit is well posed. One cell per layer.

Formulations (Section 2.3 encoding for MF6; TOP/BOT always the true dipping surfaces):
  plan : as built
  area : vertical HWVA x sec(alpha_fit)
  full : vertical HWVA x sec^2(alpha_fit)
alpha_fit per connection from the released package (apply_correction, k = 8).

Reference solutions (Hemker 1984, Eqs 13-16, T1 = T2 = Kh b):
  own   : leakance of the formulation (L_plan, L_areas, L_full)
  correct: L_full = Kv / (b cos^2 alpha)
Outputs: results_benchmarkB.csv (RMS over interior cells), profiles_benchmarkB.csv.
"""
import math
import os
import pathlib as pl
import sys

import numpy as np
import pandas as pd

HERE = pl.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopy  # noqa: E402
from disu_slope_corrector.correction import DisuGeometry, apply_correction  # noqa: E402

MF6 = os.environ.get("MF6_EXE", "mf6")
RUNS = pl.Path(os.environ.get("A5_RUNS", pl.Path(__import__("tempfile").gettempdir()) / "d1_runs")) / "benchmarkB"

L, W, B, KH, KV = 500.0, 200.0, 50.0, 5.0, 0.1
H1, H2 = (110.0, 100.0), (105.0, 102.0)
DX, NROW = 10.0, 4
NCOL = int(round(L / DX)) + 1
DY = W / NROW
NLAY = 2


def nid(lay, row, col):
    return lay * NROW * NCOL + row * NCOL + col


def build(alpha):
    ta = math.tan(alpha)
    n = NLAY * NROW * NCOL
    xc = np.zeros(n); yc = np.zeros(n); zc = np.zeros(n); top = np.zeros(n); bot = np.zeros(n)
    for lay in range(NLAY):
        for r in range(NROW):
            for c in range(NCOL):
                k = nid(lay, r, c)
                xc[k] = c * DX; yc[k] = (r + 0.5) * DY
                top[k] = -xc[k] * ta - lay * B
                bot[k] = top[k] - B
                zc[k] = top[k] - 0.5 * B
    area = np.full(n, DX * DY)
    iac = np.zeros(n, int); ja = []; ihc = []; cl12 = []; hwva = []
    for lay in range(NLAY):
        for r in range(NROW):
            for c in range(NCOL):
                k = nid(lay, r, c)
                con = [(k, 0, 0.0, 0.0)]
                if lay > 0:
                    con.append((nid(lay - 1, r, c), 0, B / 2, DX * DY))
                if lay < NLAY - 1:
                    con.append((nid(lay + 1, r, c), 0, B / 2, DX * DY))
                if c > 0:
                    con.append((nid(lay, r, c - 1), 1, DX / 2, DY))
                if c < NCOL - 1:
                    con.append((nid(lay, r, c + 1), 1, DX / 2, DY))
                if r > 0:
                    con.append((nid(lay, r - 1, c), 1, DY / 2, DX))
                if r < NROW - 1:
                    con.append((nid(lay, r + 1, c), 1, DY / 2, DX))
                con = [con[0]] + sorted(con[1:], key=lambda t: t[0])
                iac[k] = len(con)
                for m, h, cl, hw in con:
                    ja.append(m); ihc.append(h); cl12.append(cl); hwva.append(hw)
    return dict(n=n, xc=xc, yc=yc, zc=zc, top=top, bot=bot, area=area, iac=iac,
                ja=np.array(ja), ihc=np.array(ihc), cl12=np.array(cl12, float), hwva=np.array(hwva, float))


def formulations(d):
    geom = DisuGeometry(xc=d["xc"], yc=d["yc"], zc=d["zc"], top=d["top"], bot=d["bot"],
                        area=d["area"], iac=d["iac"], ja=d["ja"] + 1, ihc=d["ihc"],
                        cl12=d["cl12"], hwva=d["hwva"])
    res = apply_correction(geom, neighbours=8)
    vert = d["ihc"] == 0
    sec = 1.0 / np.where(res.cos_alpha > 0, res.cos_alpha, 1.0)
    hw_area = d["hwva"].copy(); hw_area[vert] *= sec[vert]
    hw_full = d["hwva"].copy(); hw_full[vert] *= sec[vert] ** 2
    alpha_fit = np.degrees(np.arccos(np.clip(res.cos_alpha[vert], -1, 1)))
    return {"plan": d["hwva"], "area": hw_area, "full": hw_full}, alpha_fit


def run(d, hwva, ws):
    sim = flopy.mf6.MFSimulation(sim_name="b", sim_ws=str(ws), exe_name=MF6)
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1.0, 1, 1.0)])
    flopy.mf6.ModflowIms(sim, complexity="SIMPLE", outer_dvclose=1e-9, inner_dvclose=1e-10,
                         linear_acceleration="BICGSTAB")
    gwf = flopy.mf6.ModflowGwf(sim, modelname="b")
    flopy.mf6.ModflowGwfdisu(gwf, nodes=d["n"], nja=d["ja"].size, top=d["top"], bot=d["bot"],
                             area=d["area"], iac=d["iac"], ja=d["ja"], ihc=d["ihc"],
                             cl12=d["cl12"], hwva=hwva)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=KH, k33=KV)
    flopy.mf6.ModflowGwfic(gwf, strt=105.0)
    chd = []
    for r in range(NROW):
        chd += [[(nid(0, r, 0),), H1[0]], [(nid(0, r, NCOL - 1),), H1[1]],
                [(nid(1, r, 0),), H2[0]], [(nid(1, r, NCOL - 1),), H2[1]]]
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd)
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="b.hds", saverecord=[("HEAD", "ALL")])
    sim.write_simulation(silent=True)
    ok, _ = sim.run_simulation(silent=True)
    if not ok:
        raise RuntimeError(ws)
    return np.asarray(gwf.output.head().get_data()).ravel()


def hemker(x, leak, T=KH * B):
    g = math.sqrt(leak * (2.0 / T))
    psi = 0.5 * (H1[0] + H2[0]) + 0.5 * ((H1[1] + H2[1]) - (H1[0] + H2[0])) * x / L
    ph0, phL = H1[0] - H2[0], H1[1] - H2[1]
    A = ph0
    Bc = (phL - A * math.cosh(g * L)) / math.sinh(g * L)
    phi = A * np.cosh(g * x) + Bc * np.sinh(g * x)
    return psi + 0.5 * phi, psi - 0.5 * phi, g


def main():
    dips = list(range(0, 61, 5))
    rows, prof = [], []
    for dip in dips:
        a = math.radians(dip)
        ca = math.cos(a)
        d = build(a)
        hws, afit = formulations(d)
        leaks = {"plan": KV / B, "area": KV / (B * ca), "full": KV / (B * ca * ca)}
        lay = np.arange(d["n"]) // (NROW * NCOL)
        col = np.arange(d["n"]) % NCOL
        interior = (col > 0) & (col < NCOL - 1)
        x = d["xc"]
        h1c, h2c, gc = hemker(x, leaks["full"])
        correct = np.where(lay == 0, h1c, h2c)
        for mode, hw in hws.items():
            ws = RUNS / f"dip_{dip:02d}" / mode
            ws.mkdir(parents=True, exist_ok=True)
            h = run(d, hw, ws)
            h1o, h2o, go = hemker(x, leaks[mode])
            own = np.where(lay == 0, h1o, h2o)
            i = interior
            rows.append(dict(dip=dip, mode=mode,
                             rms_vs_own_mm=1e3 * np.sqrt(np.mean((h[i] - own[i]) ** 2)),
                             rms_vs_correct_mm=1e3 * np.sqrt(np.mean((h[i] - correct[i]) ** 2)),
                             max_vs_correct_mm=1e3 * np.abs(h[i] - correct[i]).max(),
                             gamma_own=go, gamma_correct=gc,
                             alpha_fit_min=afit.min(), alpha_fit_max=afit.max()))
            for lyr in range(NLAY):
                sel = lay == lyr
                hm = pd.Series(h[sel]).groupby(col[sel]).mean().values
                for cc in range(NCOL):
                    prof.append(dict(dip=dip, mode=mode, layer=lyr + 1, x_m=cc * DX, disu_m=hm[cc],
                                     analytical_own_m=(h1o if lyr == 0 else h2o)[nid(lyr, 0, cc)],
                                     analytical_correct_m=(h1c if lyr == 0 else h2c)[nid(lyr, 0, cc)]))
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(HERE / "results_benchmarkB.csv", index=False, float_format="%.6g")
    pd.DataFrame(prof).to_csv(HERE / "profiles_benchmarkB.csv", index=False, float_format="%.8g")


if __name__ == "__main__":
    main()
