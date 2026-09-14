"""Command-line entry point for the slope-aware DISU corrector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="disu-slope-correct",
        description="Slope-aware geometric correction for layered DISU "
                    "groundwater models (MF6 and MFUSG).",
    )
    p.add_argument("--target", choices=["mf6", "mfusg"], required=True,
                   help="Model format.")
    p.add_argument("--input", required=True, type=Path,
                   help="MF6: simulation workspace containing mfsim.nam. "
                        "MFUSG: model workspace.")
    p.add_argument("--output", required=True, type=Path,
                   help="Output workspace (will be created if missing).")
    p.add_argument("--nam-file", type=str,
                   help="MFUSG: name of the .nam file inside the workspace.")
    p.add_argument("--model-name", type=str, default=None,
                   help="MF6: name of the GWF model. If omitted, the first "
                        "GWF model in the simulation is used.")
    p.add_argument("--centroids", type=Path,
                   help="MFUSG: CSV with per-cell (xc, yc) centroids in node order.")
    p.add_argument("--neighbours", type=int, default=8,
                   help="Number of nearest columns used to fit the local "
                        "interface plane. Default 8.")
    p.add_argument("--max-dip-deg", type=float, default=85.0,
                   help="Maximum permitted local dip (degrees). cos(alpha) "
                        "is clipped to this limit. Default 85.")
    p.add_argument("--pinchout-tol", type=float, default=0.0,
                   help="Vertical overlap, and cell thickness, at or below "
                        "this value count as zero. Default 0.")
    p.add_argument("--pinchout-mode", choices=["thickness", "overlap"],
                   default="thickness",
                   help="How to treat horizontal connections whose cells do "
                        "not overlap. 'thickness' (default) zeroes only a "
                        "genuine pinch-out, where a cell has no thickness; "
                        "'overlap' zeroes every non-overlapping pair, which "
                        "severs flow along a steeply dipping unit.")
    p.add_argument("--column-tol", type=float, default=1.0e-6,
                   help="Horizontal tolerance for grouping cells into the "
                        "same column. Default 1e-6.")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    correction_kwargs = dict(
        neighbours=args.neighbours,
        max_dip_deg=args.max_dip_deg,
        pinchout_tol_m=args.pinchout_tol,
        pinchout_mode=args.pinchout_mode,
        column_tol_m=args.column_tol,
    )

    if args.target == "mf6":
        from .mf6 import correct_mf6
        result = correct_mf6(
            sim_ws=args.input,
            output_ws=args.output,
            model_name=args.model_name,
            **correction_kwargs,
        )
    else:
        if args.nam_file is None:
            parser.error("--nam-file is required for --target mfusg")
        if args.centroids is None:
            parser.error("--centroids is required for --target mfusg")
        from .mfusg import correct_mfusg
        result = correct_mfusg(
            nam_file=args.nam_file,
            model_ws=args.input,
            output_ws=args.output,
            centroids_csv=args.centroids,
            **correction_kwargs,
        )

    n_vert = int((result.cos_alpha < 1.0).sum())
    n_pinch = int(result.is_pinched.sum())
    n_offset = int(result.is_offset.sum())
    print(
        f"corrected: {n_vert} vertical connections, {n_pinch} pinch-outs, "
        f"{n_offset} offset faces left unchanged",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
