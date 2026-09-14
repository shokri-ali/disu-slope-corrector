"""Journal-format versions of the revised figures (Hydrogeology Journal artwork rules).

Width 8.6 cm (one column) or 17.6 cm (page), Arial lettering 8 pt at final size, lines
>= 0.3 pt, RGB, 600 dpi. No titles inside figures; panels carry only a letter and a short
label, the detail is in the captions. Files are named by the FINAL figure number after
renumbering (Fig. 6 moved to the ESM, Fig. 15 removed):

  v2/v3 number   final file      content
  Fig. 7         Fig06           Benchmark A per-layer head drop
  Fig. 8         Fig07           Benchmark A RMS error, uniform and heterogeneous K
  Fig. 9         Fig08           Benchmark A head increments, heterogeneous K
  Fig. 10        Fig09           Benchmark B profiles at 30 deg
  Fig. 11        Fig10           Benchmark B RMS error
  Fig. 16        Fig14           Horizontal resolution, 2 x 2
  Fig. S1        FigS1           Along-dip (ESM)
  Fig. S2        FigS2           Cross-section (ESM)

Writes revision/figures_v3/journal/<name>.tif (LZW) and <name>.png (for embedding in the docx).
Data sources are the same as make_revision_figures.py, make_figure_resolution.py,
make_figure_S1.py and make_figure_S2.py.
"""
import io
import pathlib as pl

import matplotlib
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = pl.Path(__file__).resolve().parent
REV = HERE.parent
A25 = REV / "benchmark_a"
OUT = HERE / "journal"
OUT.mkdir(exist_ok=True)

CM = 1 / 2.54
COL, PAGE = 8.6 * CM, 17.6 * CM
DPI = 600
FLOOR = 1e-3

C_PLAN, C_AREA, C_FULL, C_REF = "#c0392b", "#e67e22", "#27ae60", "#7f8c8d"
C_XT, C_XTF, C_PROV = "#8e44ad", "#2980b9", "#1f4e79"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "mathtext.fontset": "custom", "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "lines.linewidth": 1.0,
    "lines.markersize": 3.5, "grid.linewidth": 0.4, "axes.titlelocation": "left",
    "legend.handlelength": 2.2, "savefig.facecolor": "white",
})
GRID = dict(ls=":", alpha=0.6)

A = pd.read_csv(A25 / "results_benchmarkA_25d.csv")
B = pd.read_csv(REV / "benchmark_b" / "results_benchmarkB.csv")
BP = pd.read_csv(REV / "benchmark_b" / "profiles_benchmarkB.csv")


def save(fig, name):
    """Write 600 dpi RGB (no alpha channel) TIFF with LZW compression, and PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    im.save(OUT / f"{name}.tif", dpi=(DPI, DPI), compression="tiff_lzw")
    im.save(OUT / f"{name}.png", dpi=(DPI, DPI))
    print("wrote", name, im.size, im.mode)


def sel(case, variant, xt3d=False, code="mf6"):
    d = A[(A.code == code) & (A.case == case) & (A.variant == variant) & (A.xt3d == xt3d)]
    return d[d.status == "ok"].sort_values("dip")


def fig06_drop():
    fig, ax = plt.subplots(figsize=(COL, 6.4 * CM), layout="constrained")
    s = np.linspace(0, 60, 300)
    c = np.cos(np.radians(s))
    ax.plot(s, np.full_like(s, 50.0), ":", color=C_PLAN)
    ax.plot(s, 50.0 * c, "--", color=C_AREA)
    ax.plot(s, 50.0 * c ** 2, "-", color=C_REF, lw=1.3)
    for var, mk, col, ms in (("plan", "o", C_PLAN, 3.5), ("area", "s", C_AREA, 3.5), ("full", "D", C_FULL, 3.8)):
        d = sel("homog", var)
        ax.plot(d.dip, d.drop_mm, mk, color=col, ms=ms, zorder=4)
    ax.text(1, 52.3, r"plan-view, $tR/K$", color=C_PLAN)
    ax.text(33, 45.5, r"area-only, $tR\cos\alpha/K$", color=C_AREA)
    ax.text(2, 14, "full correction and analytical\n" r"solution, $tR\cos^2\alpha/K$", color="#1e8449")
    ax.set_xlim(-1.5, 61.5)
    ax.set_ylim(0, 58)
    ax.set_xticks(range(0, 61, 10))
    ax.set_xlabel("Dip angle (°)")
    ax.set_ylabel("Mean per-layer head drop (mm)")
    ax.grid(True, **GRID)
    save(fig, "Fig06")


def fig07_rms_A():
    fig, axes = plt.subplots(1, 2, figsize=(PAGE, 6.8 * CM), sharey=True, layout="constrained")
    for ax, case, title in ((axes[0], "homog", "(a) Uniform K"), (axes[1], "hetK", "(b) Heterogeneous K")):
        for var, col, mk, lab in (("plan", C_PLAN, "o", "Plan-view"), ("area", C_AREA, "s", "Area-only"),
                                  ("full", C_FULL, "D", "Full slope correction")):
            d = sel(case, var)
            d = d[d.dip > 0]
            ax.plot(d.dip, np.maximum(d.rms_mm, FLOOR), mk + "-", color=col, label=lab)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Dip angle (°)")
        ax.set_xticks(range(10, 61, 10))
        ax.grid(True, which="both", **GRID)
    axes[0].set_ylabel("RMS head error (mm)")
    axes[0].set_ylim(5e-4, 1e4)
    axes[0].legend(frameon=False, loc="upper left")
    save(fig, "Fig07")


def fig08_increments():
    K = np.array([1.0, 0.01, 1.0])
    fig, axes = plt.subplots(1, 2, figsize=(PAGE, 8.2 * CM), sharey=True, layout="constrained")
    s = np.linspace(0, 62, 300)
    c = np.cos(np.radians(s))
    for L, ax in enumerate(axes):
        harm = 0.5 * 50.0 * 1e-3 * (1 / K[L] + 1 / K[L + 1]) * 1e3
        ax.plot(s, np.full_like(s, harm), ":", color=C_PLAN, lw=1.1, label="Plan-view (independent of dip)")
        ax.plot(s, harm * c, "--", color=C_AREA, label=r"Area-only closed form, $\propto\cos\alpha$")
        ax.plot(s, harm * c ** 2, "-", color=C_REF, lw=1.3, label=r"Analytical, $\propto\cos^2\alpha$")
        col_m, col_s = ("inc12_mm", "inc12_sd_mm") if L == 0 else ("inc23_mm", "inc23_sd_mm")
        for var, col, mk, lab in (("area", C_AREA, "s", "Area-only, simulated"),
                                  ("full", C_FULL, "D", "Full correction, simulated")):
            d = sel("hetK", var)
            ax.fill_between(d.dip, d[col_m] - d[col_s], d[col_m] + d[col_s], color=col, alpha=0.18, lw=0)
            ax.plot(d.dip, d[col_m], mk, color=col, label=lab)
        ax.set_title(f"({'ab'[L]}) Interface between layers {L + 1} and {L + 2}")
        ax.set_xlabel("Dip angle (°)")
        ax.set_xticks(range(0, 61, 10))
        ax.set_ylim(0, 2800)
        ax.grid(True, **GRID)
    axes[0].set_ylabel("Head increment (mm)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="outside lower center", ncol=3)
    save(fig, "Fig08")


def fig09_profiles(dip=30):
    fig, axes = plt.subplots(2, 2, figsize=(PAGE, 11.0 * CM), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [2.0, 1.0]})
    for L in (1, 2):
        ax, ex = axes[0, L - 1], axes[1, L - 1]
        ref = BP[(BP.dip == dip) & (BP["mode"] == "full") & (BP.layer == L)]
        ax.plot(ref.x_m, ref.analytical_correct_m, "-", color="#2c3e50", lw=1.3, label="Analytical solution")
        for mode, col, mk, lab in (("plan", C_PLAN, "o", "Plan-view"), ("area", C_AREA, "s", "Area-only"),
                                   ("full", C_FULL, "D", "Full slope correction")):
            d = BP[(BP.dip == dip) & (BP["mode"] == mode) & (BP.layer == L)]
            ev = d.iloc[::5]
            ax.plot(ev.x_m, ev.disu_m, mk, color=col, label=lab)
            ex.plot(d.x_m, 1e3 * (d.disu_m - d.analytical_correct_m), "-", color=col)
            ex.plot(ev.x_m, 1e3 * (ev.disu_m - ev.analytical_correct_m), mk, color=col, ms=3)
        ax.set_title(f"({'ab'[L - 1]}) Layer {L}")
        ax.grid(True, **GRID)
        ex.set_title(f"({'cd'[L - 1]}) Layer {L}, error")
        ex.axhline(0, color="#2c3e50", lw=0.6)
        ex.grid(True, **GRID)
        ex.set_xlabel("Distance in plan, x (m)")
    axes[0, 0].set_ylabel("Head (m)")
    axes[1, 0].set_ylabel("Error (mm)")
    axes[0, 1].legend(frameon=False)
    save(fig, "Fig09")


def fig10_rms_B():
    fig, ax = plt.subplots(figsize=(COL, 6.4 * CM), layout="constrained")
    for mode, col, mk, lab in (("plan", C_PLAN, "o", "Plan-view"), ("area", C_AREA, "s", "Area-only"),
                               ("full", C_FULL, "D", "Full slope correction")):
        d = B[B["mode"] == mode].sort_values("dip")
        ax.plot(d.dip, d.rms_vs_correct_mm, mk + "-", color=col, label=lab)
    ax.set_xlabel("Dip angle (°)")
    ax.set_ylabel("RMS head error (mm)")
    ax.set_xticks(range(0, 61, 10))
    ax.set_ylim(bottom=0)
    ax.grid(True, **GRID)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "Fig10")


def _res_load(n):
    if n == 250:
        d = A[(A.code == "mf6") & (A.case == "homog")]
        p = pd.read_csv(A25 / "results_provost_sublayers_25d.csv")
    else:
        d = pd.read_csv(A25 / f"results_resolution_{n}.csv")
        p = pd.read_csv(A25 / f"results_provost_sublayers_{n}.csv")
    return d[(d.status == "ok") & (d.dip > 0)], p[(p.status == "ok") & (p.dip > 0)]


def fig14_resolution():
    curves = (("full", False, C_FULL, "D-", 5.5, 3.0, 2, 0.45, "Slope-corrected geometry"),
              ("plan", False, C_PLAN, "o-", 3.5, 1.0, 3, 1.0, "Plan-view geometry"),
              ("plan", True, C_XT, "s--", 3.5, 1.0, 3, 1.0, "Plan-view geometry + XT3D"),
              ("full", True, C_XTF, "^-.", 3.5, 1.0, 3, 1.0, "Slope-corrected geometry + XT3D"))
    res = ((250, "190"), (1000, "95"), (4000, "47"), (16000, "24"))
    fig, axes = plt.subplots(2, 2, figsize=(PAGE, 14.5 * CM), sharex=True, sharey=True, layout="constrained")
    for ax, (n, size), letter in zip(axes.ravel(), res, "abcd"):
        d, p = _res_load(n)
        for var, xt, col, style, ms, lw, z, al, lab in curves:
            q = d[(d.variant == var) & (d.xt3d == xt)].sort_values("dip")
            ax.plot(q.dip, np.maximum(q.rms_mm.astype(float), FLOOR), style, color=col, ms=ms, lw=lw,
                    zorder=z, alpha=al, label=lab)
        q2 = p[p.nsub == 2].sort_values("dip")
        ax.plot(q2.dip, np.maximum(q2.rms_mm.astype(float), FLOOR), "v-", color=C_PROV, zorder=4,
                label="Provost et al. (2025) + XT3D")
        ax.set_yscale("log")
        ax.set_ylim(5e-4, 200)
        ax.set_xticks(range(10, 61, 10))
        ax.set_title(f"({letter}) Cells ≈{size} m across")
        ax.grid(True, which="both", **GRID)
    for ax in axes[1]:
        ax.set_xlabel("Dip angle (°)")
    for ax in axes[:, 0]:
        ax.set_ylabel("RMS head error (mm)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    order = [1, 2, 4, 3, 0]
    fig.legend([h[i] for i in order], [lab[i] for i in order], frameon=False, loc="outside lower center", ncol=3)
    save(fig, "Fig14")


def figS1_along_dip():
    s3 = pd.read_csv(REV / "enhanced_connectivity" / "results_provost_benchmark.csv")
    fig, ax = plt.subplots(figsize=(COL, 8.0 * CM), layout="constrained")
    d = s3[(s3.nlay_chan == 3) & (s3.status == "ok")]
    for var, col, mk, ms, lw, z, lab in (
            ("vo-s-corrV", C_FULL, "D-", 5, 2.2, 2, "Layered + slope correction (identical to layered)"),
            ("vo-s", C_PLAN, "o:", 3, 1.0, 3, "Layered connectivity, standard"),
            ("vs-s", "#2e86c1", "s--", 3, 1.0, 3, "Full connectivity, standard"),
            ("vs-x", C_PROV, "s-", 3, 1.0, 3, "Full connectivity + XT3D")):
        q = d[(d.variant == var) & (d.dip <= 40)].sort_values("dip")
        ax.plot(q.dip, q.qmag_err_pct.abs(), mk, color=col, ms=ms, lw=lw, zorder=z, label=lab)
    ax.set_xlabel("Dip angle (°)")
    ax.set_ylabel("Flux-magnitude error (%)")
    ax.set_xticks(range(0, 41, 10))
    ax.grid(True, **GRID)
    fig.legend(frameon=False, loc="outside lower center", ncol=1)
    save(fig, "FigS1")


def figS2_cross_section():
    step2 = REV / "enhanced_connectivity"
    curves = (("corrected", C_FULL, "D-", 5.5, 3.0, 2, 0.45, "Slope-corrected geometry"),
              ("planview", C_PLAN, "o-", 3.5, 1.0, 3, 1.0, "Plan-view geometry"),
              ("planview_xt3d", C_XT, "s--", 3.5, 1.0, 3, 1.0, "Plan-view geometry + XT3D"),
              ("provost_xt3d", C_PROV, "v-", 3.5, 1.0, 4, 1.0, "Provost et al. (2025) + XT3D"))
    fig, axes = plt.subplots(1, 2, figsize=(PAGE, 7.6 * CM), layout="constrained")
    for ax, fname, title, ytop in ((axes[0], "results_step2.csv", "(a) Uniform K", 200),
                                   (axes[1], "results_step2_hetK.csv", "(b) Heterogeneous K", 5000)):
        df = pd.read_csv(step2 / fname)
        d = df[(df.bc == "analytical") & (df.status == "ok") & (df.nsub == 1) & (df.dip > 0)]
        for var, col, style, ms, lw, z, al, lab in curves:
            q = d[d.variant == var].sort_values("dip")
            ax.plot(q.dip, np.maximum(q.rms_all_mm, FLOOR), style, color=col, ms=ms, lw=lw, zorder=z,
                    alpha=al, label=lab)
        ax.set_yscale("log")
        ax.set_ylim(5e-4, ytop)
        ax.set_title(title)
        ax.set_xlabel("Dip angle (°)")
        ax.set_xticks(range(10, 61, 10))
        ax.grid(True, which="both", **GRID)
    axes[0].set_ylabel("RMS head error (mm)")
    h, lab = axes[0].get_legend_handles_labels()
    order = [1, 2, 3, 0]
    fig.legend([h[i] for i in order], [lab[i] for i in order], frameon=False, loc="outside lower center", ncol=2)
    save(fig, "FigS2")


if __name__ == "__main__":
    for f in (fig06_drop, fig07_rms_A, fig08_increments, fig09_profiles, fig10_rms_B, fig14_resolution,
              figS1_along_dip, figS2_cross_section):
        f()
