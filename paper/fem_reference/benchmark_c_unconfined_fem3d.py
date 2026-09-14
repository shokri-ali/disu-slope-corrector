"""Benchmark C (unconfined): FULL 3-D saturated free-surface FEM truth.

Same physics as the 2-D x-z version (MODFLOW-style: saturated flow with a free
water table, recharge at the water table, seepage cap, free-surface iteration)
but solved on a genuine 3-D tetrahedral mesh with the strike (y) direction
resolved -- so the unconfined benchmark is 3-D-vs-3-D like A / B / confined-C.

The water table is found by iteration: solve -> set the top surface to the head
along it (h = z) -> relax -> rebuild the mesh.  Strike-uniform, so the converged
water table is uniform in y (the water-table guess is tracked per x-column).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from skfem import (Basis, ElementTetP1, ElementTetP0, FacetBasis, MeshTet,
                   asm, condense, solve, BilinearForm, LinearForm)
from skfem.helpers import dot, grad

L, W = 500.0, 300.0
NX, NY = 40, 4
N_TOP, N_AQT, N_BOT = 6, 4, 6
T_TOP, T_AQT, T_BOT = 40.0, 20.0, 40.0
K_AQ, K_AQT = 5.0, 5.0e-4
R = 5.0e-4
TOE_DRAWDOWN = 25.0
OMEGA_WT = 0.5
NZ = N_TOP + N_AQT + N_BOT


def _surfaces(x, ta):
    z_g = -x * ta
    return z_g, z_g - T_TOP, z_g - T_TOP - T_AQT, z_g - T_TOP - T_AQT - T_BOT


def _nid(i, j, k):
    return (i * (NY + 1) + j) * (NZ + 1) + k


def build_mesh(xs, ys, z_wt_x, ta):
    """3-D tet mesh of the saturated domain; top of each column = water table."""
    pts = np.empty((3, (NX + 1) * (NY + 1) * (NZ + 1)))
    for i, x in enumerate(xs):
        z_g, z_at, z_ab, z_bs = _surfaces(x, ta)
        ztop = min(z_wt_x[i], z_g)                       # seepage cap
        col = np.concatenate([
            np.linspace(ztop, z_at, N_TOP + 1),
            np.linspace(z_at, z_ab, N_AQT + 1)[1:],
            np.linspace(z_ab, z_bs, N_BOT + 1)[1:],
        ])
        for j, y in enumerate(ys):
            for k in range(NZ + 1):
                pts[:, _nid(i, j, k)] = (x, y, col[k])

    split = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
             (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
    tets = []
    for i in range(NX):
        for j in range(NY):
            for k in range(NZ):
                v = [_nid(i, j, k), _nid(i + 1, j, k), _nid(i + 1, j + 1, k),
                     _nid(i, j + 1, k), _nid(i, j, k + 1), _nid(i + 1, j, k + 1),
                     _nid(i + 1, j + 1, k + 1), _nid(i, j + 1, k + 1)]
                for a, b, c, e in split:
                    tets.append([v[a], v[b], v[c], v[e]])
    return MeshTet(pts, np.array(tets, dtype=np.int64).T)


def solve_free_surface(alpha_rad, verbose=False):
    ta, ca = math.tan(alpha_rad), math.cos(alpha_rad)
    xs = np.linspace(0.0, L, NX + 1)
    ys = np.linspace(0.0, W, NY + 1)
    z_g_x = -xs * ta
    z_wt_x = z_g_x - 5.0
    elem = ElementTetP1()
    h_toe = (-L * ta) - TOE_DRAWDOWN

    top_node_ids = np.array([_nid(i, j, 0)
                             for i in range(NX + 1) for j in range(NY + 1)])
    # node index -> originating x-column, for averaging the water table over y
    top_col_of_node = np.array([i for i in range(NX + 1) for j in range(NY + 1)])
    at_node_ids = np.array([_nid(i, j, N_TOP)
                            for i in range(NX + 1) for j in range(NY + 1)])
    ab_node_ids = np.array([_nid(i, j, N_TOP + N_AQT)
                            for i in range(NX + 1) for j in range(NY + 1)])

    @BilinearForm
    def form(u, v, w):
        return w["c"] * dot(grad(u), grad(v))

    @LinearForm
    def rch(v, w):
        return R * ca * v

    h = None
    for sweep in range(60):
        mesh = build_mesh(xs, ys, z_wt_x, ta)
        basis = Basis(mesh, elem)
        basis0 = basis.with_element(ElementTetP0())

        cen = mesh.p[:, mesh.t].mean(axis=1)
        z_at = (-cen[0] * ta) - T_TOP
        z_ab = z_at - T_AQT
        k_e = np.where((cen[2] <= z_at) & (cen[2] >= z_ab), K_AQT, K_AQ)
        A = asm(form, basis, c=basis0.interpolate(k_e))

        bf = mesh.boundary_facets()
        fe = mesh.facets[:, bf]
        is_top = np.all(np.isin(fe, top_node_ids), axis=0)
        b = asm(rch, FacetBasis(mesh, elem, facets=bf[is_top]))

        xc, zc = mesh.p[0], mesh.p[2]
        z_ab_n = (-xc * ta) - T_TOP - T_AQT
        toe = (np.abs(xc - L) < 1e-6) & (zc < z_ab_n + 1e-6)
        D = np.where(toe)[0]
        x0 = np.full(basis.N, h_toe)
        h = solve(*condense(A, b, x=x0, D=D))

        # water table = head along the top; average over y per x-column
        h_top = h[top_node_ids]
        z_wt_new = np.full(NX + 1, np.nan)
        for i in range(NX + 1):
            z_wt_new[i] = h_top[top_col_of_node == i].mean()
        z_wt_new = np.minimum(z_wt_new, z_g_x)               # seepage cap
        z_wt_new = np.maximum(z_wt_new, z_g_x - T_TOP + 0.5)  # keep WT in top aquifer
        dwt = np.max(np.abs(z_wt_new - z_wt_x))
        z_wt_x = (1 - OMEGA_WT) * z_wt_x + OMEGA_WT * z_wt_new
        if verbose:
            print(f"  sweep {sweep:2d} dWT={dwt:.4f}")
        if dwt < 1e-3:
            break

    dh = float(h[at_node_ids].mean() - h[ab_node_ids].mean())
    return dh, sweep, dwt


def main():
    out = Path(__file__).resolve().parent
    rows = []
    print(f"{'dip':>5} {'dh aquitard (m)':>16} {'dh/dh(0)':>10} {'cos^2':>8} "
          f"{'sweeps':>7}")
    print("-" * 52)
    dh0 = None
    for dip in (0, 15, 30, 45):
        a = math.radians(dip)
        dh, sw, dwt = solve_free_surface(a)
        if dh0 is None:
            dh0 = dh
        print(f"{dip:5d} {dh:16.4f} {dh/dh0:10.4f} {math.cos(a)**2:8.4f} {sw:7d}")
        rows.append((dip, dh))
    with (out / "benchmark_c_unconfined_fem3d.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dip_deg", "aquitard_head_drop_m"])
        for dip, dh in rows:
            w.writerow([dip, f"{dh:.6f}"])
    print("\nwrote benchmark_c_unconfined_fem3d.csv")


if __name__ == "__main__":
    main()
