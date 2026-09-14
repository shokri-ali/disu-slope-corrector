"""Independent 3-D continuum FEM reference for Benchmark B (two-aquifer leaky).

Faithful continuum representation of the benchmark's conceptual model:

  * two well-mixed confined aquifers (each vertical thickness B, transmissivity
    Kh*B), realised by giving the aquifer a large bedding-normal conductivity so
    its head is ~uniform through thickness, exactly as a one-cell-per-layer DISU
    model assumes;
  * the inter-layer leakance concentrated in a thin dipping aquitard of vertical
    thickness E and *isotropic* conductivity Ka = E*Kv/B, so that in the flat
    limit the leakance is Ka/E = Kv/B (Hemker's L_plan).

The dip is introduced ONLY through the tilted mesh geometry.  Because the
aquitard is isotropic, no cosine is inserted anywhere: the inclined aquitard has
area A_plan/cos(a) and bedding-normal thickness E*cos(a), so its leakance per
plan area becomes Ka/(E*cos^2 a) = Kv/(B*cos^2 a) = L_full automatically.  If the
continuum FEM matches the L_full analytical and diverges from L_plan as dip
grows, the cos^2 leakance law is confirmed by an independent method that never
sees a leakance formula.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from skfem import Basis, ElementTetP1, MeshTet, asm, condense, solve
from skfem import BilinearForm
from skfem.helpers import grad


# ---------------------------------------------------------------------------
# Benchmark B configuration (manuscript section 2.6)
# ---------------------------------------------------------------------------
B = 50.0          # vertical thickness of each aquifer [m]
L = 500.0         # plan-view length in dip direction [m]
W = 200.0         # along-strike width [m]
KH = 5.0          # aquifer horizontal (bedding-parallel) conductivity [m/d]
KV = 0.1          # inter-layer (bedding-normal) conductivity defining leakance [m/d]
KV_AQ = 500.0     # large within-aquifer vertical K -> well-mixed aquifers

H1_0, H1_L = 110.0, 100.0   # layer-1 (upper) CHD at x=0, x=L
H2_0, H2_L = 105.0, 102.0   # layer-2 (lower) CHD at x=0, x=L

E_AQUITARD = 2.0            # thin aquitard vertical thickness [m]
KA = E_AQUITARD * KV / B    # aquitard isotropic K giving leakance Kv/B when flat

N_DIP = 30
N_STRIKE = 2
CELLS_PER_AQUIFER = 6
CELLS_IN_AQUITARD = 2


def _zeta_planes():
    """Vertical (downward) node positions from the top, with the thin aquitard."""
    a1 = np.linspace(0.0, B, CELLS_PER_AQUIFER + 1)
    aq = np.linspace(B, B + E_AQUITARD, CELLS_IN_AQUITARD + 1)[1:]
    a2 = np.linspace(B + E_AQUITARD, 2 * B + E_AQUITARD,
                     CELLS_PER_AQUIFER + 1)[1:]
    return np.concatenate([a1, aq, a2])


def build_mesh(alpha_rad: float):
    ta = math.tan(alpha_rad)
    zplanes = _zeta_planes()
    nzc = len(zplanes) - 1
    nxp, nyp, nzp = N_DIP + 1, N_STRIKE + 1, nzc + 1
    xs = np.linspace(0.0, L, nxp)
    ys = np.linspace(0.0, W, nyp)

    pts = np.empty((3, nxp * nyp * nzp))

    def gid(i, j, k):
        return (i * nyp + j) * nzp + k

    for i in range(nxp):
        z_top = -xs[i] * ta
        for j in range(nyp):
            for k in range(nzp):
                pts[:, gid(i, j, k)] = (xs[i], ys[j], z_top - zplanes[k])

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
    return MeshTet(pts, cells), zplanes


def _make_form(kxx, kyy, kzz, kxz):
    @BilinearForm
    def f(u, v, w):
        du, dv = grad(u), grad(v)
        return ((kxx * du[0] + kxz * du[2]) * dv[0]
                + (kyy * du[1]) * dv[1]
                + (kxz * du[0] + kzz * du[2]) * dv[2])
    return f


def solve_head(alpha_rad: float):
    mesh, zplanes = build_mesh(alpha_rad)
    elem = ElementTetP1()
    basis = Basis(mesh, elem)
    ca, sa = math.cos(alpha_rad), math.sin(alpha_rad)
    ta = math.tan(alpha_rad)

    # classify elements by zeta_from_top of their centroid
    cen = mesh.p[:, mesh.t].mean(axis=1)          # (3, nelem)
    zeta_e = (-cen[0] * ta) - cen[2]
    is_aquitard = (zeta_e > B) & (zeta_e < B + E_AQUITARD)

    # aquifer: AXIS-ALIGNED anisotropy -- high vertical K makes the vertical
    # prism well-mixed (one head per column, as a 1-cell-per-layer DISU cell
    # assumes) while keeping horizontal transmissivity Kh*b exactly.  No
    # rotation, so the horizontal term is not contaminated at dip.
    kxx_a = KH
    kzz_a = KV_AQ
    kxz_a = 0.0
    aquifer_form = _make_form(kxx_a, KH, kzz_a, kxz_a)
    # aquitard: isotropic Ka (dip enters only through geometry)
    aquitard_form = _make_form(KA, KA, KA, 0.0)

    aquifer_elems = np.where(~is_aquitard)[0]
    aquitard_elems = np.where(is_aquitard)[0]
    A = asm(aquifer_form, Basis(mesh, elem, elements=aquifer_elems))
    A = A + asm(aquitard_form, Basis(mesh, elem, elements=aquitard_elems))

    # Dirichlet on the two vertical end faces, per aquifer (aquitard left free)
    xc, zc = mesh.p[0], mesh.p[2]
    zeta = (-xc * ta) - zc
    tol = 1e-6
    x0, xL = np.abs(xc) < tol, np.abs(xc - L) < tol
    in_l1 = zeta < (B - tol)
    in_l2 = zeta > (B + E_AQUITARD + tol)

    x = np.full(basis.N, np.nan)
    x[x0 & in_l1] = H1_0
    x[x0 & in_l2] = H2_0
    x[xL & in_l1] = H1_L
    x[xL & in_l2] = H2_L
    D = np.where(~np.isnan(x))[0]
    x[np.isnan(x)] = 0.0

    head = solve(*condense(A, np.zeros(basis.N), x=x, D=D))
    return mesh, head


def layer_profiles(mesh, head, alpha_rad):
    ta = math.tan(alpha_rad)
    xc = mesh.p[0]
    zeta = (-xc * ta) - mesh.p[2]
    xs = np.unique(np.round(xc, 6))
    h1, h2 = [], []
    for xv in xs:
        col = np.abs(xc - xv) < 1e-6
        h1.append(head[col & (zeta < B - 1e-9)].mean())
        h2.append(head[col & (zeta > B + E_AQUITARD + 1e-9)].mean())
    return xs, np.array(h1), np.array(h2)


def hemker(alpha_rad, xs, mode):
    ca = math.cos(alpha_rad)
    lk = {"plan": KV / B,
          "areas": KV / (B * ca),
          "full": KV / (B * ca * ca)}[mode]
    T1 = T2 = KH * B
    gamma = math.sqrt(lk * (1.0 / T1 + 1.0 / T2))
    psi0 = (T1 * H1_0 + T2 * H2_0) / (T1 + T2)
    psiL = (T1 * H1_L + T2 * H2_L) / (T1 + T2)
    psi = psi0 + (psiL - psi0) * xs / L
    phi0, phiL = H1_0 - H2_0, H1_L - H2_L
    Acoef = phi0
    Bcoef = (phiL - Acoef * math.cosh(gamma * L)) / math.sinh(gamma * L)
    phi = Acoef * np.cosh(gamma * xs) + Bcoef * np.sinh(gamma * xs)
    h1 = psi + (T2 / (T1 + T2)) * phi
    h2 = psi - (T1 / (T1 + T2)) * phi
    return h1, h2


def rms(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    out_dir = Path(__file__).resolve().parent
    rows = []
    print(f"{'dip':>5}  RMS head error of continuum FEM vs analytical leakance (mm)")
    print(f"{'':>5} {'vs plan':>10} {'vs area-only':>13} {'vs full':>10}  "
          f"{'closest':>8}")
    print("-" * 56)
    for dip_deg in (0, 15, 30, 45, 60):
        a = math.radians(dip_deg)
        mesh, head = solve_head(a)
        xs, h1f, h2f = layer_profiles(mesh, head, a)
        res = {}
        for mode in ("plan", "areas", "full"):
            h1a, h2a = hemker(a, xs, mode)
            res[mode] = rms(np.concatenate([h1f, h2f]),
                            np.concatenate([h1a, h2a])) * 1e3
        closest = min(res, key=res.get)
        print(f"{dip_deg:5d} {res['plan']:10.3f} {res['areas']:13.3f} "
              f"{res['full']:10.3f}  {closest:>8}")
        rows.append(dict(dip_deg=dip_deg, **{f"fem_vs_{m}_mm": res[m] for m in res}))

    csv_path = out_dir / "benchmark_b_fem_vs_analytic.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dip_deg", "fem_vs_plan_mm", "fem_vs_areas_mm", "fem_vs_full_mm"])
        for r in rows:
            w.writerow([r["dip_deg"], f"{r['fem_vs_plan_mm']:.6f}",
                        f"{r['fem_vs_areas_mm']:.6f}", f"{r['fem_vs_full_mm']:.6f}"])
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
