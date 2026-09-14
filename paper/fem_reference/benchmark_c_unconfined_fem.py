"""Benchmark C (unconfined): variably-saturated 3-D FEM truth, dipping 3 layers.

No analytical solution and NO Dupuit-Forchheimer assumption.  This solves steady
variably-saturated (free-surface) flow on the genuinely tilted mesh by Picard
iteration on a smoothed relative conductivity kr(psi), psi = h - z:

    kr(psi) = kr_min + (1 - kr_min) * 0.5*(1 + tanh(psi/delta))

so kr -> 1 below the water table and -> kr_min above it, with a smooth ~delta
transition.  This is a fully 3-D free-surface model (the water table is where
psi = 0); it makes none of the layered / Dupuit assumptions of the DISU model.

Configuration:
  L1 aquifer  (top)  : 40 m, K = 5    m/d   -- unconfined, recharged
  L2 aquitard (mid)  : 20 m, K = 5e-4 m/d   -- tight, dipping bottleneck
  L3 aquifer  (bot)  : 40 m, K = 5    m/d
Recharge R on the (dipping) top surface; specified head at the basal toe (x=L);
no-flow elsewhere.  Recharge must leak down through the dipping aquitard to the
basal outlet, so the head drop across the aquitard is set by the aquitard
leakance -- the sec^2 quantity under test.

Diagnostic: head drop across the aquitard, dh = h(top-of-aquitard) -
h(base-of-aquitard).  With the true (full) leakance dh ~ R*t*cos^2(a)/Kv
(decreases with dip); plan-view geometry keeps dh dip-independent.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from skfem import Basis, ElementTetP0, ElementTetP1, MeshTet, asm, condense, solve
from skfem import BilinearForm
from skfem.helpers import dot, grad

import benchmark_c_confined_fem as femc   # reuse mesh builder + layer interfaces

LAYERS_K = [5.0, 5.0e-4, 5.0]             # isotropic K per layer [m/d]
L = femc.L
W = femc.W
N_DIP = femc.N_DIP
N_STRIKE = femc.N_STRIKE
_ZIFACE = femc._ZIFACE                     # [0,40,60,100] zeta-from-top interfaces

R = 5.0e-4                                 # recharge [m/d] per horizontal area
KR_MIN = 1.0e-3
DELTA = 2.0                                # water-table smoothing band [m]
OMEGA = 0.35                               # Picard under-relaxation on head
TOE_DRAWDOWN = 30.0                        # toe head below ground surface at x=L


def solve_unconfined(alpha_rad, h_init=None, cells=(8, 4, 6)):
    femc.CELLS = list(cells)
    mesh = femc.build_mesh(alpha_rad)
    elem = ElementTetP1()
    basis = Basis(mesh, elem)
    basis0 = basis.with_element(ElementTetP0())
    ta = math.tan(alpha_rad)
    ca = math.cos(alpha_rad)

    cen = mesh.p[:, mesh.t].mean(axis=1)
    zeta_e = (-cen[0] * ta) - cen[2]
    layer_e = femc._layer_of_zeta(zeta_e)
    k_e = np.array([LAYERS_K[li] for li in layer_e])
    z_e = cen[2]

    @BilinearForm
    def form(u, v, w):
        return w["c"] * dot(grad(u), grad(v))

    # recharge on the (inclined) top surface: mass-balance normal flux R*cos(a)
    def on_top(x):
        zeta = (-x[0] * ta) - x[2]
        return np.abs(zeta) < 1e-6
    top_facets = mesh.facets_satisfying(on_top, boundaries_only=True)
    from skfem import FacetBasis, LinearForm
    fb = FacetBasis(mesh, elem, facets=top_facets)

    @LinearForm
    def rch(v, w):
        return R * ca * v
    b = asm(rch, fb)

    # basal toe Dirichlet
    xc, zc = mesh.p[0], mesh.p[2]
    zeta_n = (-xc * ta) - zc
    z_top_L = -L * ta
    h_toe = z_top_L - TOE_DRAWDOWN
    toe = (np.abs(xc - L) < 1e-6) & (zeta_n > _ZIFACE[-2] + 1e-6)
    D = np.where(toe)[0]

    # Picard iteration on kr(psi).  Start from the continuation guess if given
    # (previous dip's solution, elevation-shifted), else a topography-following
    # water table 15 m below the ground surface so no region starts "dry".
    h = h_init.copy() if h_init is not None else (-xc * ta) - 15.0
    h[D] = h_toe
    for it in range(300):
        h_e = h[mesh.t].mean(axis=0)        # element-mean head
        psi_e = h_e - z_e                   # pressure head at element centroid
        kr = KR_MIN + (1.0 - KR_MIN) * 0.5 * (1.0 + np.tanh(psi_e / DELTA))
        cond_e = k_e * kr
        A = asm(form, basis, c=basis0.interpolate(cond_e))
        x = np.full(basis.N, h_toe)
        x[D] = h_toe
        h_new = solve(*condense(A, b, x=x, D=D))
        dmax = np.max(np.abs(h_new - h))
        h = (1.0 - OMEGA) * h + OMEGA * h_new
        if dmax < 1e-4:
            break

    return mesh, h, it, dmax


def aquitard_head_drop(mesh, h, alpha_rad):
    """Mean head drop across the aquitard: h(zeta=40) - h(zeta=60)."""
    ta = math.tan(alpha_rad)
    xc = mesh.p[0]
    zeta = (-xc * ta) - mesh.p[2]
    top_iface = np.abs(zeta - _ZIFACE[1]) < 1e-6      # top of aquitard
    bot_iface = np.abs(zeta - _ZIFACE[2]) < 1e-6      # base of aquitard
    return float(h[top_iface].mean() - h[bot_iface].mean())


def main():
    out_dir = Path(__file__).resolve().parent
    rows = []
    print(f"{'dip':>5} {'dh aquitard (m)':>16} {'dh/dh(0)':>10} {'cos^2':>8} "
          f"{'iters':>6} {'resid':>10}")
    print("-" * 60)
    report = {0, 15, 30, 45}
    dh0 = None
    h_prev, ta_prev, x_nodes = None, 0.0, None
    for dip_deg in range(0, 46, 5):                      # continuation in dip
        a = math.radians(dip_deg)
        ta = math.tan(a)
        if h_prev is not None:
            h_init = h_prev + (-x_nodes) * (ta - ta_prev)  # elevation shift
        else:
            h_init = None
        mesh, h, it, resid = solve_unconfined(a, h_init=h_init)
        x_nodes = mesh.p[0]
        h_prev, ta_prev = h, ta
        if dip_deg in report:
            dh = aquitard_head_drop(mesh, h, a)
            if dh0 is None:
                dh0 = dh
            print(f"{dip_deg:5d} {dh:16.4f} {dh/dh0:10.4f} {math.cos(a)**2:8.4f} "
                  f"{it:6d} {resid:10.2e}")
            rows.append(dict(dip_deg=dip_deg, dh=dh))

    csv_path = out_dir / "benchmark_c_unconfined_fem.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dip_deg", "aquitard_head_drop_m"])
        for r in rows:
            w.writerow([r["dip_deg"], f"{r['dh']:.6f}"])
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
