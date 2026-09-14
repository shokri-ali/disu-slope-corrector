"""End-to-end MODFLOW 6 test: the corrected geometry must scale the *simulated*
vertical flux by sec^2(alpha).

Unlike the array-ratio tests, this runs MF6 and measures real fluxes, so it
guards against the subtlety that MF6 ignores CL12 for vertical conductance: the
``code="mf6"`` correction folds sec^2 into HWVA, and only an actual MF6 run
confirms the conductance (hence the flux) really changes by sec^2.

Skipped automatically if flopy or an mf6 executable is not available.
"""
from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

flopy = pytest.importorskip("flopy")

from disu_slope_corrector.correction import DisuGeometry, apply_correction


def _find_mf6():
    """Locate an mf6 executable: env var, PATH, flopy bin, or repo ./bin."""
    cand = [os.environ.get("MF6_EXE"), shutil.which("mf6"), shutil.which("mf6.exe")]
    here = Path(__file__).resolve()
    for up in here.parents:
        cand.append(str(up / "bin" / "mf6.exe"))
        cand.append(str(up / "bin" / "mf6"))
    try:
        w = flopy.which("mf6")
        if w:
            cand.append(w)
    except Exception:
        pass
    for c in cand:
        if c and Path(c).exists():
            return c
    return None


MF6_EXE = _find_mf6()
needs_mf6 = pytest.mark.skipif(MF6_EXE is None, reason="mf6 executable not found")


def _build_two_layer_disu(alpha_deg, ncol=4, nrow=4, spacing=10.0, t=10.0, k=1.0):
    """Dipping 2-layer DISU (vertical connections only), plan-view geometry.

    Returns a dict of flopy-ready arrays (ja 0-based) plus the DisuGeometry the
    corrector consumes (ja 1-based).
    """
    ta = math.tan(math.radians(alpha_deg))
    ncell = 2 * nrow * ncol
    area = spacing * spacing

    def nid(lay, row, col):
        return lay * nrow * ncol + row * ncol + col

    xc = np.zeros(ncell); yc = np.zeros(ncell); zc = np.zeros(ncell)
    top = np.zeros(ncell); bot = np.zeros(ncell)
    for lay in range(2):
        for row in range(nrow):
            for col in range(ncol):
                x = col * spacing
                ztop0 = -ta * x
                n = nid(lay, row, col)
                xc[n] = x; yc[n] = row * spacing
                top[n] = ztop0 - lay * t
                bot[n] = ztop0 - (lay + 1) * t
                zc[n] = 0.5 * (top[n] + bot[n])

    iac = np.zeros(ncell, dtype=int)
    ja = []; ihc = []; cl12 = []; hwva = []
    for lay in range(2):
        for row in range(nrow):
            for col in range(ncol):
                n = nid(lay, row, col)
                conns = [(n, 0, 0.0, 0.0)]
                other = nid(1 - lay, row, col)
                conns.append((other, 0, t / 2, area))   # vertical connection
                conns = [conns[0]] + sorted(conns[1:], key=lambda c: c[0])
                iac[n] = len(conns)
                for (m, h, cl, hw) in conns:
                    ja.append(m); ihc.append(h); cl12.append(cl); hwva.append(hw)

    ja = np.array(ja, dtype=int)
    ihc = np.array(ihc, dtype=int)
    cl12 = np.array(cl12, dtype=float)
    hwva = np.array(hwva, dtype=float)
    geom = DisuGeometry(xc=xc, yc=yc, zc=zc, top=top, bot=bot,
                        area=np.full(ncell, area), iac=iac, ja=ja + 1,
                        ihc=ihc, cl12=cl12, hwva=hwva)
    return dict(ncell=ncell, top=top, bot=bot, area=np.full(ncell, area),
                iac=iac, ja=ja, ihc=ihc, cl12=cl12, hwva=hwva, k=k,
                nrow=nrow, ncol=ncol, geom=geom)


def _run_vertical_flux(d, cl12, hwva, ws):
    """All-CHD model: top layer h=10, bottom layer h=0; return total |flux|."""
    sim = flopy.mf6.MFSimulation(sim_name="t", sim_ws=str(ws), exe_name=MF6_EXE)
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1.0, 1, 1.0)])
    flopy.mf6.ModflowIms(sim, complexity="SIMPLE", outer_dvclose=1e-9,
                         inner_dvclose=1e-10)
    gwf = flopy.mf6.ModflowGwf(sim, modelname="t", save_flows=True)
    flopy.mf6.ModflowGwfdisu(gwf, nodes=d["ncell"], nja=d["ja"].size,
                             top=d["top"], bot=d["bot"], area=d["area"],
                             iac=d["iac"], ja=d["ja"], ihc=d["ihc"],
                             cl12=cl12, hwva=hwva)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=d["k"])
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)
    ncell = d["ncell"]
    half = ncell // 2
    chd = [[(n,), 10.0] for n in range(half)]            # top layer
    chd += [[(n,), 0.0] for n in range(half, ncell)]     # bottom layer
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd, save_flows=True)
    flopy.mf6.ModflowGwfoc(gwf, budget_filerecord="t.cbc",
                           saverecord=[("BUDGET", "ALL")])
    sim.write_simulation(silent=True)
    ok, _ = sim.run_simulation(silent=True)
    assert ok, f"MF6 failed in {ws}"
    chd_rec = gwf.output.budget().get_data(text="CHD")[-1]
    q = np.asarray(chd_rec["q"])
    return float(np.abs(q).sum() / 2.0)


@needs_mf6
@pytest.mark.parametrize("dip_deg", [30.0, 45.0, 60.0])
def test_mf6_vertical_flux_scales_as_sec_squared(dip_deg, tmp_path):
    d = _build_two_layer_disu(dip_deg)
    res = apply_correction(d["geom"], neighbours=8, max_dip_deg=89.0, code="mf6")

    q_plan = _run_vertical_flux(d, d["cl12"], d["hwva"], tmp_path / "plan")
    q_full = _run_vertical_flux(d, res.cl12, res.hwva, tmp_path / "full")

    sec2 = 1.0 / math.cos(math.radians(dip_deg)) ** 2
    assert q_full / q_plan == pytest.approx(sec2, rel=2e-3)


@needs_mf6
def test_mf6_cl12_alone_does_not_change_flux(tmp_path):
    """Guard the core finding: scaling only CL12 leaves the MF6 vertical flux
    unchanged (so the mfusg-style CL12 correction is inert in MF6)."""
    d = _build_two_layer_disu(45.0)
    q_ref = _run_vertical_flux(d, d["cl12"], d["hwva"], tmp_path / "ref")
    q_clhalf = _run_vertical_flux(d, d["cl12"] * 0.5, d["hwva"], tmp_path / "clhalf")
    assert q_clhalf == pytest.approx(q_ref, rel=1e-6)
