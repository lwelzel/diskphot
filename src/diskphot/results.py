"""Observed-fit loading and physical result tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import constants
from .config import Variant
from .data import EpochData

#: Extinction opacity kappa = 3 Q_ext / (4 a rho) for the adopted grains
#: (a = 3.7e-4 cm, rho = 1.5 g/cm^3, Q_ext = 2) ~ 2.7e3 cm^2/g.
KAPPA_EXT_CGS: float = float(getattr(
    constants, "KAPPA_EXT_CGS",
    3.0 * getattr(constants, "Q_EXT", 2.0)
    / (4.0 * getattr(constants, "GRAIN_RADIUS_CM", 3.7e-4)
       * getattr(constants, "GRAIN_DENSITY_CGS", 1.5)),
))


def column_density_cgs(tau: float | np.ndarray) -> float | np.ndarray:
    """Dust column density [g/cm^2] from an extinction optical depth."""
    return np.asarray(tau, dtype=float) / KAPPA_EXT_CGS


@dataclass
class ObservedFit:
    variant: Variant
    logl_null: float
    theta_null: np.ndarray
    n_weighted_pixels: int
    param_names: np.ndarray | None = None
    theta_mle: np.ndarray | None = None
    logl_mle: float | None = None
    q0: float | None = None
    tau_curve_grid: np.ndarray | None = None
    logl_curve: np.ndarray | None = None

    def as_payload(self) -> dict:
        out = {"logl_null": self.logl_null}
        for key in ("logl_mle", "q0", "logl_curve"):
            v = getattr(self, key)
            if v is not None:
                out[key] = v
        return out


def load_fit(run: Path) -> ObservedFit:
    with np.load(Path(run) / "diskphot_fit.npz", allow_pickle=False) as z:
        get = lambda k: z[k] if k in z.files else None
        return ObservedFit(
            variant=Variant(str(z["variant"])),
            logl_null=float(z["logl_null"]),
            theta_null=z["theta_null"],
            n_weighted_pixels=int(z["n_weighted_pixels"]),
            param_names=get("param_names"),
            theta_mle=get("theta_mle"),
            logl_mle=None if get("logl_mle") is None else float(z["logl_mle"]),
            q0=None if get("q0") is None else float(z["q0"]),
            tau_curve_grid=get("tau_curve_grid"),
            logl_curve=get("logl_curve"),
        )


def load_calibration_summary(run: Path) -> dict:
    return json.loads((Path(run) / "limits_summary.json").read_text())


def epoch_table(summary: dict, data: EpochData, out_path: Path | None = None):
    """Per-epoch sightline table: D-hat, tau upper limits, column density."""
    from astropy.table import Table

    dhat = data.disk_hat_at_galaxy()
    rows = []
    for i in range(data.n_epochs):
        row = {"epoch": i, "date": data.dates[i] if i < len(data.dates) else "",
               "dhat": float(dhat[i])}
        for cl, entry in summary.get("limits", {}).items():
            ul = entry.get("upper_limit_observed")
            if ul is not None:
                row[f"tau_ul_{cl}"] = float(ul) * float(dhat[i])
                row[f"sigma_dust_{cl}_gcm2"] = float(column_density_cgs(row[f"tau_ul_{cl}"]))
        rows.append(row)
    table = Table(rows=rows)
    if out_path is not None:
        table.write(out_path, format="ascii.ecsv", overwrite=True)
    return table


def summary_lines(summary: dict) -> list[str]:
    lines = [
        f"detection threshold c_det = {summary['c_det']:.3f} "
        f"(alpha = {summary['alpha_det']:g}, n_null = {summary['n_null']})",
    ]
    if "observed_q0" in summary:
        lines.append(
            f"observed q0 = {summary['observed_q0']:.3f} -> "
            f"p = {summary['observed_q0_pvalue']:.4f} "
            f"({'DETECTED' if summary['observed_detected'] else 'not detected'})"
        )
    lines.append(
        f"sensitivity (>= {100 * 0.95:.0f}% power): tau' = "
        f"{summary['sensitivity_tau']:.3e} "
        f"(conservative {summary['sensitivity_tau_conservative']:.3e})"
    )
    for cl, entry in summary.get("limits", {}).items():
        ul = entry.get("upper_limit_observed")
        if ul is not None:
            lines.append(f"upper limit ({float(cl) * 100:.1f}% CL): tau' <= {ul:.3e}")
    if "warning_n_null" in summary:
        lines.append("WARNING: " + summary["warning_n_null"])
    return lines
