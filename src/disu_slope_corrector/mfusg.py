"""MODFLOW-USG DISU adapter for the slope-aware correction.

MFUSG cannot use XT3D, so for a layered MFUSG model the slope-aware
geometric correction is the only available remedy for dipping-interface
conductance bias.

This adapter reads a MFUSG model via FloPy, applies the correction in
arrays equivalent to the MF6 DISU layout, and writes the corrected model
to a new workspace. Centroids are required; if the DISU package does not
already carry per-cell ``XC``/``YC``, the caller must supply them through
``centroids_csv``.
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
            "flopy is required for the MFUSG adapter; install with `pip install flopy`"
        ) from exc


def _read_centroids_csv(path: str | Path, n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Read per-cell (xc, yc) centroids from a two- or three-column CSV.

    Acceptable layouts:
        xc, yc                      (rows ordered by node)
        node, xc, yc                (1-based or 0-based; auto-detected)
    """
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] == 2:
        if arr.shape[0] != n_cells:
            raise ValueError(
                f"centroids CSV has {arr.shape[0]} rows but model has {n_cells} cells"
            )
        return arr[:, 0].astype(float), arr[:, 1].astype(float)
    if arr.shape[1] == 3:
        order = np.argsort(arr[:, 0].astype(int))
        arr = arr[order]
        return arr[:, 1].astype(float), arr[:, 2].astype(float)
    raise ValueError(f"unsupported centroids CSV shape {arr.shape}")


def load_mfusg_disu(nam_file: str | Path,
                    model_ws: str | Path,
                    centroids_csv: Optional[str | Path] = None
                    ) -> tuple[DisuGeometry, object]:
    """Load DISU geometry from a MODFLOW-USG name file.

    Parameters
    ----------
    nam_file : path
        MFUSG name file (typically ``*.nam``).
    model_ws : path
        Model workspace.
    centroids_csv : path, optional
        CSV with per-cell (xc, yc) centroids. Required because MFUSG DISU
        does not natively carry per-cell horizontal centroids in a format
        FloPy exposes directly.
    """
    _require_flopy()
    import flopy

    mf = flopy.modflow.Modflow.load(
        str(nam_file),
        model_ws=str(model_ws),
        version="mfusg",
        check=False,
        verbose=False,
    )
    disu = mf.get_package("DISU")
    if disu is None:
        raise RuntimeError(f"model in {model_ws} has no DISU package")

    nodes = int(disu.nodes)
    top = np.asarray(disu.top.array, dtype=float).ravel()
    bot = np.asarray(disu.bot.array, dtype=float).ravel()
    area = np.asarray(disu.area.array, dtype=float).ravel()
    iac = np.asarray(disu.iac.array, dtype=int).ravel()
    ja = np.asarray(disu.ja.array, dtype=int).ravel()
    ivc = np.asarray(disu.ivc.array, dtype=int).ravel()
    cl12 = np.asarray(disu.cl1.array, dtype=float).ravel()
    hwva = np.asarray(disu.fahl.array, dtype=float).ravel()

    # MFUSG uses IVC (1 = vertical, 0 = horizontal); MF6 uses IHC (0 = vertical,
    # 1 = horizontal). Convert.
    ihc = np.where(ivc == 1, 0, 1).astype(int)

    if centroids_csv is None:
        raise ValueError(
            "MFUSG corrector requires `centroids_csv`: a CSV with per-cell "
            "(xc, yc) centroids in node order."
        )
    xc, yc = _read_centroids_csv(centroids_csv, nodes)
    zc = 0.5 * (top + bot)

    geom = DisuGeometry(
        xc=xc, yc=yc, zc=zc,
        top=top, bot=bot, area=area,
        iac=iac, ja=ja, ihc=ihc, cl12=cl12, hwva=hwva,
    )
    return geom, mf


def save_mfusg_disu(mf, result: CorrectionResult, output_ws: str | Path) -> None:
    """Write the corrected MFUSG model to a new workspace."""
    _require_flopy()
    disu = mf.get_package("DISU")
    disu.cl1 = result.cl12
    disu.fahl = result.hwva
    mf.change_model_ws(str(output_ws))
    mf.write_input()


def correct_mfusg(nam_file: str | Path,
                  model_ws: str | Path,
                  output_ws: str | Path,
                  centroids_csv: str | Path,
                  **correction_kwargs) -> CorrectionResult:
    """Convenience: load, correct, save in one call."""
    geom, mf = load_mfusg_disu(nam_file, model_ws, centroids_csv=centroids_csv)
    result = apply_correction(geom, **correction_kwargs)
    save_mfusg_disu(mf, result, output_ws)
    return result
