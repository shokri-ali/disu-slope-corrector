"""Benchmark C (confined): independent 3-D FEM truth for a dipping 3-layer system.

No analytical solution.  The FEM is the reference for the three DISU
formulations (plan-view / area-only / full slope-aware).

Conceptual model matched to a layered DISU grid so that the ONLY thing the
benchmark probes is the vertical (cross-formational) conductance across the
dipping aquitard -- the quantity the slope correction targets:

  * aquifers: well-mixed vertical prisms (axis-aligned high vertical K, so each
    prism carries ~one head as a one-cell-per-layer DISU cell does) with
    horizontal conductivity Kh acting on vertical side faces;
  * aquitard: isotropic K, so its bedding-normal leakance follows the tilted
    geometry as Kv/(t cos^2 a) automatically -- no cosine inserted by hand.

Configuration ("outcrop-to-discharge" dipping system):
  L1 aquifer  (top)  : 40 m, Kh = 5    m/d
  L2 aquitard (mid)  : 20 m, K  = 0.05 m/d
  L3 aquifer  (bot)  : 40 m, Kh = 5    m/d
  plan 1000 m (down-dip) x 300 m (strike); confined top & bottom.

Boundary conditions force vertical flow through the dipping aquitard:
  * specified head H_in  on the TOP aquifer at the up-dip end   (x = 0)
  * specified head H_out on the BASAL aquifer at the down-dip toe (x = L)
  * all other faces / end-face patches: no-flow
Water entering the top aquifer has no down-dip outlet of its own, so it must
descend through the dipping aquitard to reach the basal outlet.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from skfem import Basis, ElementTetP1, MeshTet, asm, condense, solve
from skfem import BilinearForm
from skfem.helpers import grad


# (vertical thickness [m], horizontal Kh [m/d]); also used by the DISU script.
# The aquitard is tight (a true confining unit) so it is the flow bottleneck and
# the sec^2 leakance signal is not masked by aquifer horizontal transmission.
LAYERS = [(40.0, 5.0), (20.0, 0.0005), (40.0, 5.0)]
# vertical conductivity for the FEM truth: aquifers well-mixed, aquitard isotropic
KZZ = [1000.0, 0.0005, 1000.0]
CELLS = [5, 4, 5]
L = 1000.0
W = 300.0
N_DIP = 40
N_STRIKE = 3

H_IN = 100.0
H_OUT = 85.0

TOTAL_T = sum(t for t, _ in LAYERS)
_ZIFACE = np.cumsum([0.0] + [t for t, _ in LAYERS])     # [0,40,60,100]


def _zeta_planes():
    planes = [np.linspace(_ZIFACE[i], _ZIFACE[i + 1], CELLS[i] + 1)
              for i in range(len(LAYERS))]
    out = [planes[0]]
    for p in planes[1:]:
        out.append(p[1:])
    return np.concatenate(out)


def build_mesh(alpha_rad):
    ta = math.tan(alpha_rad)
    zp = _zeta_planes()
    nzc = len(zp) - 1
    nxp, nyp, nzp = N_DIP + 1, N_STRIKE + 1, nzc + 1
    xs = np.linspace(0.0, L, nxp)
    ys = np.linspace(0.0, W, nyp)
    pts = np.empty((3, nxp * nyp * nzp))

    def gid(i, j, k):
        return (i * nyp + j) * nzp + k

    for i in range(nxp):
        ztop = -xs[i] * ta
        for j in range(nyp):
            for k in range(nzp):
                pts[:, gid(i, j, k)] = (xs[i], ys[j], ztop - zp[k])

    split = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
             (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
    tets = []
    for i in range(N_DIP):
        for j in range(N_STRIKE):
            for k in range(nzc):
                v = [gid(i, j, k), gid(i + 1, j, k), gid(i + 1, j + 1, k),
                     gid(i, j + 1, k), gid(i, j, k + 1), gid(i + 1, j, k + 1),
                     gid(i + 1, j + 1, k + 1), gid(i, j + 1, k + 1)]
                for a, b, c, e in split:
                    tets.append([v[a], v[b], v[c], v[e]])
    return MeshTet(pts, np.array(tets, dtype=np.int64).T)


def _layer_of_zeta(zeta):
    return np.clip(np.searchsorted(_ZIFACE, zeta, side="right") - 1,
                   0, len(LAYERS) - 1)


def solve_head(alpha_rad):
    mesh = build_mesh(alpha_rad)
    elem = ElementTetP1()
    basis = Basis(mesh, elem)
    ta = math.tan(alpha_rad)

    cen = mesh.p[:, mesh.t].mean(axis=1)
    zeta_e = (-cen[0] * ta) - cen[2]
    layer_e = _layer_of_zeta(zeta_e)

    def kform(kxx, kyy, kzz):
        @BilinearForm
        def f(u, v, w):
            du, dv = grad(u), grad(v)
            return kxx * du[0] * dv[0] + kyy * du[1] * dv[1] + kzz * du[2] * dv[2]
        return f

    A = None
    for li, (_, kh) in enumerate(LAYERS):
        elems = np.where(layer_e == li)[0]
        if elems.size == 0:
            continue
        Ai = asm(kform(kh, kh, KZZ[li]), Basis(mesh, elem, elements=elems))
        A = Ai if A is None else A + Ai

    xc, zc = mesh.p[0], mesh.p[2]
    zeta = (-xc * ta) - zc
    tol = 1e-6
    in_top = zeta < (_ZIFACE[1] - tol)                 # L1 (top aquifer)
    in_bot = zeta > (_ZIFACE[-2] + tol)                # L3 (basal aquifer)
    x0, xL = np.abs(xc) < tol, np.abs(xc - L) < tol

    x = np.full(basis.N, np.nan)
    x[x0 & in_top] = H_IN                               # top aquifer, up-dip
    x[xL & in_bot] = H_OUT                              # basal aquifer, toe
    D = np.where(~np.isnan(x))[0]
    out_dofs = np.where(xL & in_bot)[0]
    x[np.isnan(x)] = 0.0

    head = solve(*condense(A, np.zeros(basis.N), x=x, D=D))
    q_out = float((A @ head)[out_dofs].sum())
    return mesh, head, q_out


def main():
    out_dir = Path(__file__).resolve().parent
    rows = []
    print(f"{'dip':>5} {'outlet Q (m3/d)':>17} {'Q/Q(0)':>9} {'sec^2':>8}")
    print("-" * 44)
    q0 = None
    for dip_deg in (0, 15, 30, 45, 60):
        a = math.radians(dip_deg)
        mesh, head, q_out = solve_head(a)
        if q0 is None:
            q0 = abs(q_out)
        print(f"{dip_deg:5d} {q_out:17.5f} {abs(q_out)/q0:9.4f} "
              f"{1/math.cos(a)**2:8.4f}")
        rows.append(dict(dip_deg=dip_deg, outlet_q=q_out))

    csv_path = out_dir / "benchmark_c_confined_fem.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dip_deg", "outlet_q_m3d"])
        for r in rows:
            w.writerow([r["dip_deg"], f"{r['outlet_q']:.6f}"])
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
