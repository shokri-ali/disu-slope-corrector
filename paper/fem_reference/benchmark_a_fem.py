"""Independent 3-D finite-element reference for Benchmark A (dipping confined slab).

This is a FEFLOW-style check built from open parts:

* a body-fitted tetrahedral mesh of the *actual tilted slab* (built here, no gmsh),
* a standard Galerkin FEM solve of confined Darcy flow on that mesh (scikit-fem).

The point of this script is to remove the circularity in the paper's original
validation.  The analytical per-layer head drop  t*R*cos^2(a)/K  already contains
the cos^2 factor by construction, so matching DISU to it does not, by itself,
prove the cos^2 law.  Here the FEM never sees a conductance formula and never has
cos^2 inserted by hand.  The only geometric facts supplied are:

  (1) the genuinely tilted mesh geometry, and
  (2) the recharge flux through the inclined top surface, which by *mass balance*
      with the plan-view recharge R (per unit horizontal/map area) must be
      q_n = R*cos(a) per unit inclined area  (inclined top area = A_plan/cos a).

If the cos^2 head-drop law emerges from solving Darcy's equation on the tilted
mesh, the law is a property of the physics, not of the analytical ansatz.  One
cos comes from the recharge BC (mass balance); the other emerges from the solve
through the bedding-normal path length -- exactly the two factors the paper
attributes to area and length.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from skfem import (
    Basis,
    ElementTetP1,
    FacetBasis,
    MeshTet,
    asm,
    condense,
    solve,
)
from skfem import BilinearForm, LinearForm
from skfem.helpers import dot, grad


# ---------------------------------------------------------------------------
# Benchmark A configuration (matches the manuscript, section 2.2)
# ---------------------------------------------------------------------------
T_LAYER = 50.0          # vertical thickness of each layer [m]
N_LAYERS = 3
R = 1.0e-3              # recharge per unit horizontal (map) area [m/d]
K = 1.0                # homogeneous hydraulic conductivity [m/d]
H_BOTTOM = 0.0         # constant head on the basal surface [m]

L_DIP = 500.0          # down-dip (bedding-parallel) extent [m]
L_STRIKE = 200.0       # along-strike extent [m]

# mesh resolution
N_DIP = 10             # cells down-dip
N_STRIKE = 4           # cells along strike
CELLS_PER_LAYER = 4    # cells through each layer (bedding-normal)


def build_tilted_tet_mesh(alpha_rad: float):
    """Structured tetrahedral mesh of a slab dipping by ``alpha_rad`` in +x.

    Built in bedding coordinates (xi=down-dip, eta=strike, zeta=bedding-normal)
    and rotated into physical (x, y, z).  Because the basis vectors are
    orthonormal and the origin is 0, zeta = P . n_hat exactly, which makes it
    trivial to tag the top / bottom faces and the layer interfaces later.

    Returns the mesh plus the orthonormal basis vectors and total bedding-normal
    thickness.
    """
    ca, sa = math.cos(alpha_rad), math.sin(alpha_rad)
    d_hat = np.array([ca, 0.0, -sa])   # down-dip, bedding-parallel
    s_hat = np.array([0.0, 1.0, 0.0])  # strike
    n_hat = np.array([sa, 0.0, ca])    # bedding-normal (upward)

    nzc = N_LAYERS * CELLS_PER_LAYER
    nxp, nyp, nzp = N_DIP + 1, N_STRIKE + 1, nzc + 1

    zeta_total = N_LAYERS * T_LAYER * ca   # bedding-normal total thickness
    dxi = L_DIP / N_DIP
    deta = L_STRIKE / N_STRIKE
    dzeta = zeta_total / nzc

    # node coordinates
    pts = np.empty((3, nxp * nyp * nzp))

    def gid(i, j, k):
        return (i * nyp + j) * nzp + k

    for i in range(nxp):
        for j in range(nyp):
            for k in range(nzp):
                xi, eta, zeta = i * dxi, j * deta, k * dzeta
                pts[:, gid(i, j, k)] = xi * d_hat + eta * s_hat + zeta * n_hat

    # hexahedra -> 6 tets (Freudenthal split on diagonal v0-v6, conforming)
    tets = []
    split = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
             (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
    for i in range(N_DIP):
        for j in range(N_STRIKE):
            for k in range(nzc):
                v = [
                    gid(i, j, k),       gid(i + 1, j, k),
                    gid(i + 1, j + 1, k), gid(i, j + 1, k),
                    gid(i, j, k + 1),   gid(i + 1, j, k + 1),
                    gid(i + 1, j + 1, k + 1), gid(i, j + 1, k + 1),
                ]
                for a, b, c, e in split:
                    tets.append([v[a], v[b], v[c], v[e]])

    cells = np.array(tets, dtype=np.int64).T
    mesh = MeshTet(pts, cells)
    return mesh, n_hat, zeta_total


def solve_head(alpha_rad: float):
    """Solve confined Darcy flow on the tilted slab; return mesh, basis, head."""
    mesh, n_hat, zeta_total = build_tilted_tet_mesh(alpha_rad)
    elem = ElementTetP1()
    basis = Basis(mesh, elem)

    # zeta coordinate of every point = P . n_hat
    zeta_pts = n_hat @ mesh.p

    # bilinear form: integral K grad(h) . grad(v)
    @BilinearForm
    def stiffness(u, v, w):
        return K * dot(grad(u), grad(v))

    A = asm(stiffness, basis)

    # Neumann recharge on the inclined TOP surface.
    # Mass balance with plan-view recharge R (per map area): the same total water
    # R*A_plan must enter through the inclined top of area A_plan/cos(a), so the
    # normal influx per unit inclined area is q_n = R*cos(a).
    ca = math.cos(alpha_rad)
    q_top = R * ca

    def on_top(x):
        zeta = n_hat[0] * x[0] + n_hat[1] * x[1] + n_hat[2] * x[2]
        return np.abs(zeta - zeta_total) < 1e-6

    def on_bottom(x):
        zeta = n_hat[0] * x[0] + n_hat[1] * x[1] + n_hat[2] * x[2]
        return np.abs(zeta) < 1e-6

    top_facets = mesh.facets_satisfying(on_top, boundaries_only=True)
    fb_top = FacetBasis(mesh, elem, facets=top_facets)

    @LinearForm
    def neumann(v, w):
        return q_top * v

    b = asm(neumann, fb_top)

    # Dirichlet (constant head) on the basal surface
    bottom_dofs = basis.get_dofs(facets=mesh.facets_satisfying(on_bottom,
                                                               boundaries_only=True))
    x = np.full(basis.N, H_BOTTOM, dtype=float)
    x = solve(*condense(A, b, x=x, D=bottom_dofs))

    return mesh, basis, x, n_hat, zeta_total


def per_layer_drops(mesh, head, n_hat, zeta_total):
    """Mean head on each interface plane and the per-layer drops (top->down)."""
    zeta_pts = n_hat @ mesh.p
    ca = zeta_total / (N_LAYERS * T_LAYER)  # = cos(alpha)
    interfaces = [l * T_LAYER * ca for l in range(N_LAYERS + 1)]  # zeta of planes
    mean_h = []
    for z in interfaces:
        sel = np.abs(zeta_pts - z) < 1e-6
        mean_h.append(float(head[sel].mean()))
    # interfaces are ordered bottom(0) -> top(N); drop across a layer is
    # h(upper interface) - h(lower interface)
    drops = [mean_h[l + 1] - mean_h[l] for l in range(N_LAYERS)]
    return drops, mean_h


def main():
    out_dir = Path(__file__).resolve().parent
    rows = []
    print(f"{'dip':>5} {'FEM drop':>12} {'analytic':>12} {'plan-view':>12} "
          f"{'FEM/analytic':>13} {'sec^2 bias':>11}")
    print("-" * 72)
    for dip_deg in range(0, 86, 5):
        a = math.radians(dip_deg)
        mesh, basis, head, n_hat, zeta_total = solve_head(a)
        drops, mean_h = per_layer_drops(mesh, head, n_hat, zeta_total)
        fem_drop = float(np.mean(drops))                 # m
        analytic = T_LAYER * R * math.cos(a) ** 2 / K    # m
        planview = T_LAYER * R / K                        # dip-independent
        ratio = fem_drop / analytic
        sec2 = 1.0 / math.cos(a) ** 2
        print(f"{dip_deg:5d} {fem_drop*1e3:11.4f}m {analytic*1e3:11.4f}m "
              f"{planview*1e3:11.4f}m {ratio:13.5f} {sec2:11.4f}")
        rows.append(dict(dip_deg=dip_deg,
                         fem_drop_mm=fem_drop * 1e3,
                         analytic_drop_mm=analytic * 1e3,
                         planview_drop_mm=planview * 1e3,
                         fem_over_analytic=ratio,
                         sec2_alpha=sec2,
                         per_layer_drops_mm=[d * 1e3 for d in drops]))

    csv_path = out_dir / "benchmark_a_fem_vs_analytic.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dip_deg", "fem_drop_mm", "analytic_drop_mm",
                    "planview_drop_mm", "fem_over_analytic", "sec2_alpha"])
        for r in rows:
            w.writerow([r["dip_deg"], f"{r['fem_drop_mm']:.6f}",
                        f"{r['analytic_drop_mm']:.6f}",
                        f"{r['planview_drop_mm']:.6f}",
                        f"{r['fem_over_analytic']:.6f}",
                        f"{r['sec2_alpha']:.6f}"])
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
