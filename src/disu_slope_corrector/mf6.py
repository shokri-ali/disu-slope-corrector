"""MODFLOW 6 DISU adapter for the slope-aware correction.

Reads a DISU package from a FloPy simulation, applies the correction, and
writes the modified arrays back. Requires FloPy >= 3.6.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .correction import CorrectionResult, DisuGeometry, apply_correction


def _require_flopy():
    try:
        import flopy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "flopy is required for the MF6 adapter; install with `pip install flopy`"
        ) from exc


def load_mf6_disu(sim_ws: str | Path,
                  model_name: Optional[str] = None) -> tuple[DisuGeometry, object]:
    """Load DISU geometry from a MODFLOW 6 simulation workspace.

    Parameters
    ----------
    sim_ws : path
        Simulation workspace containing ``mfsim.nam``.
    model_name : str, optional
        Name of the groundwater flow model. If omitted, the first GWF model
        found in the simulation is used.

    Returns
    -------
    geom : DisuGeometry
        Geometry and connectivity arrays in the package's native order.
    gwf : flopy.mf6.ModflowGwf
        The underlying FloPy GWF model object, returned so the caller can
        write modified arrays back via ``save_mf6_disu``.
    """
    _require_flopy()
    import flopy

    sim = flopy.mf6.MFSimulation.load(sim_ws=str(sim_ws), verbosity_level=0)
    gwf_names = sim.model_names
    if model_name is None:
        if not gwf_names:
            raise RuntimeError(f"no GWF models found in {sim_ws}")
        model_name = gwf_names[0]
    gwf = sim.get_model(model_name)
    disu = gwf.get_package("disu")
    if disu is None:
        raise RuntimeError(f"model {model_name} has no DISU package")

    # Pull arrays. FloPy returns numpy arrays for these.
    top = np.asarray(disu.top.array, dtype=float)
    bot = np.asarray(disu.bot.array, dtype=float)
    area = np.asarray(disu.area.array, dtype=float)
    iac = np.asarray(disu.iac.array, dtype=int)
    ja = np.asarray(disu.ja.array, dtype=int)
    ihc = np.asarray(disu.ihc.array, dtype=int)
    cl12 = np.asarray(disu.cl12.array, dtype=float)
    hwva = np.asarray(disu.hwva.array, dtype=float)

    # Centroids: read from the cell2d block. xc, yc are required; zc is the
    # cell centre, taken as midpoint of (top, bot).
    cell2d = disu.cell2d.array
    if cell2d is None:
        raise RuntimeError(
            "DISU package does not expose CELL2D centroids; the corrector "
            "requires (xc, yc) per cell"
        )
    xc = np.asarray(cell2d["xc"], dtype=float)
    yc = np.asarray(cell2d["yc"], dtype=float)
    zc = 0.5 * (top + bot)

    geom = DisuGeometry(
        xc=xc, yc=yc, zc=zc,
        top=top, bot=bot, area=area,
        iac=iac, ja=ja, ihc=ihc, cl12=cl12, hwva=hwva,
    )
    return geom, gwf


def save_mf6_disu(gwf, result: CorrectionResult,
                  output_ws: str | Path) -> None:
    """Write the corrected DISU arrays back to a new simulation workspace.

    The simulation containing ``gwf`` is reused; only the DISU's ``cl12``
    and ``hwva`` arrays are replaced, and the simulation is rewritten to
    ``output_ws``.
    """
    _require_flopy()
    disu = gwf.get_package("disu")
    disu.cl12.set_data(result.cl12)
    disu.hwva.set_data(result.hwva)

    sim = gwf.simulation
    sim.set_sim_path(str(output_ws))
    sim.write_simulation(silent=True)


def correct_mf6(sim_ws: str | Path,
                output_ws: str | Path,
                model_name: Optional[str] = None,
                **correction_kwargs) -> CorrectionResult:
    """Convenience: load, correct, save in one call.

    Returns the CorrectionResult so callers can inspect per-connection
    cos(alpha) and pinch-out flags.
    """
    geom, gwf = load_mf6_disu(sim_ws, model_name=model_name)
    result = apply_correction(geom, **correction_kwargs)
    save_mf6_disu(gwf, result, output_ws)
    return result
