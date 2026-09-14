"""Horizontal-resolution study for Fig. 16(a): Benchmark A on the 2.5-D Voronoi grid
with 250 (published panel), 1000 (~95 m) and 4000 (~47 m, about the layer thickness)
cells per layer. Uniform K, MF6 6.5.0, same boundary conditions and scripts.

Configurations per dip (5..60 by 5): plan-view, plan-view + XT3D, slope-corrected,
slope-corrected + XT3D, Provost et al. (2025) enhanced connectivity + XT3D (1 layer/unit).

Usage: python resolution_study.py 1000   (one resolution per call; env A5_RUNS = run root)
"""
import multiprocessing as mp
import os
import pathlib as pl
import sys
import time

HERE = pl.Path(__file__).resolve().parent


def main():
    ncell = sys.argv[1]
    os.environ["A25_NCELL"] = ncell            # inherited by the worker processes
    sys.path.insert(0, str(HERE))
    import benchmarkA_25d as b                 # importable by name, so workers can unpickle run_mf6
    t0 = time.time()
    verts, cell2d = b.load_grid()
    print(f"grid {len(cell2d)} cells/layer, {len(verts)} vertices ({time.time()-t0:.0f} s)", flush=True)
    root = os.environ.get("A5_RUNS", str(pl.Path(__import__("tempfile").gettempdir()) / "a25_runs"))
    tasks = [("homog", dp, v, x, root) for dp in range(5, 61, 5)
             for v, x in (("plan", False), ("plan", True), ("full", False), ("full", True), ("provost", True))]
    rows = []
    with mp.Pool(int(os.environ.get("A25_PROCS", "6"))) as pool:
        for i, r in enumerate(pool.imap_unordered(b.run_mf6, tasks), 1):
            r["ncell"] = int(ncell)
            rows.append(r)
            if i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  ({time.time()-t0:.0f} s)", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows).sort_values(["variant", "xt3d", "dip"])
    out = HERE / f"results_resolution_{ncell}.csv"
    df.to_csv(out, index=False, float_format="%.6g")
    df["cfg"] = df.variant + df.xt3d.map({True: "+xt3d", False: ""})
    print(df.pivot_table(index="dip", columns="cfg", values="rms_mm").round(3).to_string())
    print("failed:", df[df.status != "ok"][["cfg", "dip"]].to_string(index=False))


if __name__ == "__main__":
    main()
