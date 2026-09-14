"""Benchmark C (unconfined): saturated free-surface FEM truth (MODFLOW-style).

This replaces the Richards-flavoured (unsaturated-zone) solver, which was brittle
on the dipping geometry.  Following how MODFLOW treats an unconfined aquifer, we
solve SATURATED groundwater flow only: the water table is the TOP of the
saturated domain (no unsaturated zone), recharge is applied AT the water table,
and a seepage cap prevents it rising above ground.  The free-surface position is
found by iteration (h = z on the water table).  The vertical is fully resolved,
so this is NOT a Dupuit model -- it is an independent, geometry-resolving truth.

Strike-uniform, so solved as an exact 2-D x-z cross-section (per unit width).

Geometry (dipping uniformly at alpha):
  ground surface  z_g(x)  = -x tan(a)
  aquitard top    z_at(x) = z_g - 40      (base of top aquifer)
  aquitard base   z_ab(x) = z_g - 60
  base of system  z_bs(x) = z_g - 100
K: top & bottom aquifer 5 m/d, tight dipping aquitard 5e-4 m/d.
Recharge R at the water table; basal toe head fixed at x = L; elsewhere no-flow.

Diagnostic: head drop across the aquitard, dh = h(z_at) - h(z_ab), which scales
as cos^2(a) for the true geometry and stays dip-independent for plan-view DISU.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from skfem import (Basis, ElementTriP1, ElementTriP0, FacetBasis, MeshTri,
                   asm, condense, solve, BilinearForm, LinearForm)
from skfem.helpers import dot, grad

L = 500.0
NX = 60
N_TOP, N_AQT, N_BOT = 6, 4, 6          # vertical cells: top-sat / aquitard / bottom
T_TOP, T_AQT, T_BOT = 40.0, 20.0, 40.0
K_AQ, K_AQT = 5.0, 5.0e-4
R = 5.0e-4
TOE_DRAWDOWN = 25.0                     # toe head below ground surface at x = L
OMEGA_WT = 0.5                          # water-table relaxation


def _surfaces(x, ta):
    z_g = -x * ta
    return z_g, z_g - T_TOP, z_g - T_TOP - T_AQT, z_g - T_TOP - T_AQT - T_BOT


def build_mesh(xs, z_wt, ta):
    """Triangulated x-z mesh of the saturated domain; top = water table."""
    nx = xs.size
    nz = N_TOP + N_AQT + N_BOT
    pts = np.empty((2, nx * (nz + 1)))

    def nid(i, k):
        return i * (nz + 1) + k

    for i, x in enumerate(xs):
        z_g, z_at, z_ab, z_bs = _surfaces(x, ta)
        ztop = min(z_wt[i], z_g)                       # seepage cap at surface
        col = np.concatenate([
            np.linspace(ztop, z_at, N_TOP + 1),
            np.linspace(z_at, z_ab, N_AQT + 1)[1:],
            np.linspace(z_ab, z_bs, N_BOT + 1)[1:],
        ])
        for k in range(nz + 1):
            pts[:, nid(i, k)] = (x, col[k])

    tris = []
    for i in range(nx - 1):
        for k in range(nz):
            a, b = nid(i, k), nid(i + 1, k)
            c, d = nid(i + 1, k + 1), nid(i, k + 1)
            tris.append([a, b, d]); tris.append([b, c, d])
    return MeshTri(pts, np.array(tris, dtype=np.int64).T)


def solve_free_surface(alpha_rad, verbose=False):
    ta, ca = math.tan(alpha_rad), math.cos(alpha_rad)
    xs = np.linspace(0.0, L, NX + 1)
    z_g = -xs * ta
    z_wt = z_g - 5.0                                   # initial guess: 5 m deep

    elem = ElementTriP1()
    h_toe = (-L * ta) - TOE_DRAWDOWN

    @BilinearForm
    def form(u, v, w):
        return w["c"] * dot(grad(u), grad(v))

    @LinearForm
    def rch(v, w):
        return R * ca * v

    nz = N_TOP + N_AQT + N_BOT
    nzp = nz + 1
    top_node_ids = np.arange(NX + 1) * nzp            # k = 0 of each column
    top_set = set(top_node_ids.tolist())
    for sweep in range(60):
        mesh = build_mesh(xs, z_wt, ta)
        basis = Basis(mesh, elem)
        basis0 = basis.with_element(ElementTriP0())

        # element conductivity by layer (fixed interfaces follow the dip)
        cen = mesh.p[:, mesh.t].mean(axis=1)
        z_at = (-cen[0] * ta) - T_TOP
        z_ab = z_at - T_AQT
        k_e = np.where((cen[1] <= z_at) & (cen[1] >= z_ab), K_AQT, K_AQ)
        A = asm(form, basis, c=basis0.interpolate(k_e))

        # recharge on the top (water-table) surface: boundary facets whose
        # both endpoints are top-of-column nodes
        bf = mesh.boundary_facets()
        fe = mesh.facets[:, bf]
        is_top = np.isin(fe[0], top_node_ids) & np.isin(fe[1], top_node_ids)
        fb = FacetBasis(mesh, elem, facets=bf[is_top])
        b = asm(rch, fb)

        # basal toe Dirichlet at x = L, bottom aquifer
        xc, zc = mesh.p[0], mesh.p[1]
        z_ab_n = (-xc * ta) - T_TOP - T_AQT
        toe = (np.abs(xc - L) < 1e-6) & (zc < z_ab_n + 1e-6)
        D = np.where(toe)[0]
        x0 = np.full(basis.N, h_toe)
        x0[D] = h_toe
        h = solve(*condense(A, b, x=x0, D=D))

        # update water table = head along the current top surface (h = z there)
        h_top = h[top_node_ids]                        # ordered by column / x
        z_wt_new = np.minimum(h_top, z_g)              # seepage cap
        z_wt_new = np.maximum(z_wt_new, z_g - T_TOP + 0.5)  # keep WT in top aquifer
        dwt = np.max(np.abs(z_wt_new - z_wt))
        z_wt = (1 - OMEGA_WT) * z_wt + OMEGA_WT * z_wt_new
        if verbose:
            print(f"  sweep {sweep:2d} dWT={dwt:8.4f} WT[min,max]="
                  f"[{z_wt.min():.2f},{z_wt.max():.2f}]")
        if dwt < 1e-3:
            break

    return mesh, h, sweep, dwt


def aquitard_head_drop(mesh, h, alpha_rad):
    ta = math.tan(alpha_rad)
    xc, zc = mesh.p[0], mesh.p[1]
    z_at = (-xc * ta) - T_TOP
    z_ab = z_at - T_AQT
    top = np.abs(zc - z_at) < 1e-6
    bot = np.abs(zc - z_ab) < 1e-6
    return float(h[top].mean() - h[bot].mean())


def main():
    out = Path(__file__).resolve().parent
    rows = []
    print(f"{'dip':>5} {'dh aquitard (m)':>16} {'dh/dh(0)':>10} {'cos^2':>8} "
          f"{'sweeps':>7} {'dWT':>9}")
    print("-" * 60)
    dh0 = None
    for dip in (0, 15, 30, 45):
        a = math.radians(dip)
        mesh, h, sw, dwt = solve_free_surface(a)
        dh = aquitard_head_drop(mesh, h, a)
        if dh0 is None:
            dh0 = dh
        print(f"{dip:5d} {dh:16.4f} {dh/dh0:10.4f} {math.cos(a)**2:8.4f} "
              f"{sw:7d} {dwt:9.2e}")
        rows.append((dip, dh))
    with (out / "benchmark_c_unconfined_fem.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dip_deg", "aquitard_head_drop_m"])
        for dip, dh in rows:
            w.writerow([dip, f"{dh:.6f}"])
    print("\nwrote benchmark_c_unconfined_fem.csv")


if __name__ == "__main__":
    main()
