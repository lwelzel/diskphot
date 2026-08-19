"""Result assembly and the paper tables.

The fitted ``tau'`` multiplies the disk model normalized at the ring
surface-brightness peak, so ``tau'`` is the physical optical depth at the SB
peak; the optical depth along the galaxy sightline at epoch *i* is
``tau_i = tau' * dhat_i`` with ``dhat_i`` the normalized disk SB at the
galaxy position (mean within 2 px, as in the manuscript).

All quantitative statements are frequentist and simulation-calibrated:

* the detection statistic ``q0``, its p-value against the simulated null
  ensemble, and the detection threshold ``c_det``;
* the calibrated (Neyman) confidence upper limits on ``tau'`` at the
  configured one-sided confidence levels, with the measured coverage;
* the injection sensitivity (smallest ``tau'`` detected with the target
  probability).

The per-epoch table scales the ``tau'`` upper limit by ``dhat_i``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.table import Table

from . import constants
from .config import FitConfig, Variant
from .data import EpochData

__all__ = [
    "ObservedFit",
    "column_density_cgs",
    "epoch_table",
    "load_calibration_summary",
    "load_fit",
    "summary_lines",
]


class ObservedFit:
    """Lazy view over the ``diskphot_fit.npz`` written by ``run_fit``."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        payload = np.load(self.run_dir / "diskphot_fit.npz", allow_pickle=False)
        self.variant = Variant(str(payload["variant"]))
        self.logl_null = float(payload["logl_null"])
        self.theta_null = payload["theta_null"]
        self.n_weighted_pixels = int(payload["n_weighted_pixels"])
        self.param_names: list[str] = (
            [str(n) for n in payload["param_names"]] if "param_names" in payload else []
        )
        self.theta_mle = payload["theta_mle"] if "theta_mle" in payload else None
        self.logl_mle = float(payload["logl_mle"]) if "logl_mle" in payload else None
        self.q0 = float(payload["q0"]) if "q0" in payload else None
        self.tau_grid = payload["tau_curve_grid"] if "tau_curve_grid" in payload else None
        self.logl_curve = payload["logl_curve"] if "logl_curve" in payload else None

    @property
    def config(self) -> FitConfig:
        return FitConfig.from_json(self.run_dir / "config.json")

    @property
    def tau_mle(self) -> float | None:
        if self.theta_mle is None or self.variant is Variant.NULL:
            return None
        n_off = sum(1 for n in self.param_names if n.startswith(("dx", "dy")))
        return float(self.theta_mle[n_off])

    def q_curve(self) -> np.ndarray:
        """Observed profile-likelihood-ratio curve ``q(tau')`` on the grid.

        ``q(tau') = 2 [max lnL_p - lnL_p(tau')]`` clipped at zero, with the
        maximum taken over the free MLE, the null fit, and the curve itself
        (the boundary MLE ``tau' = 0`` is admissible).
        """
        if self.logl_curve is None:
            raise ValueError("this run holds no profile curve (variant != common)")
        lnl_hat = max(float(self.logl_mle), self.logl_null, float(np.max(self.logl_curve)))
        return np.clip(2.0 * (lnl_hat - np.asarray(self.logl_curve)), 0.0, None)

    def as_payload(self) -> dict:
        """Dict view consumed by :func:`diskphot.calibrate.analyze_calibration`."""
        out: dict = {"logl_null": self.logl_null}
        if self.logl_mle is not None:
            out |= {"logl_mle": self.logl_mle, "q0": self.q0}
        if self.logl_curve is not None:
            out |= {"tau_curve_grid": self.tau_grid, "logl_curve": self.logl_curve}
        return out


def load_fit(run_dir: Path | str) -> ObservedFit:
    return ObservedFit(Path(run_dir))


def load_calibration_summary(path: Path | str) -> dict:
    """Read the JSON written by :func:`diskphot.calibrate.analyze_calibration`."""
    return json.loads(Path(path).read_text())


def _upper_limits_from_summary(summary: dict) -> dict[float, float]:
    out: dict[float, float] = {}
    for entry in summary.get("limits", {}).values():
        if "upper_limit" in entry:
            out[float(entry["cl"])] = float(entry["upper_limit"])
    return out


def epoch_table(
    summary: dict,
    data: EpochData,
    aperture_radius_px: float = 2.0,
    out_path: Path | None = None,
) -> Table:
    """Per-epoch physical optical-depth limits and geometry (the paper table).

    ``summary`` is the calibration summary dict (including observed limits);
    each per-epoch sightline limit is the ``tau'`` upper limit scaled by
    ``dhat_i`` (exact for the common-``tau'`` model, in which the sightline
    optical depth is proportional to ``dhat_i`` by construction).
    """
    dhat = data.disk_hat_at_galaxy(aperture_radius_px)
    r_arcsec = data.galaxy_radii_arcsec()
    r_au = np.asarray(constants.arcsec_to_au(r_arcsec))

    columns: dict[str, np.ndarray | list] = {
        "epoch": np.arange(data.n_epochs),
        "date": list(data.dates),
        "r_arcsec": np.round(r_arcsec, 3),
        "r_au": np.round(r_au, 1),
        "dhat_gal": np.round(dhat, 4),
    }
    uls = _upper_limits_from_summary(summary)
    for cl, ul in sorted(uls.items()):
        col = f"tau_ul_{int(round(1000 * cl)):03d}"
        columns[col] = ul * dhat
    sens = summary.get("sensitivity")
    if sens is not None and np.isfinite(sens):
        columns["tau_sens"] = float(sens) * dhat

    table = Table(columns)
    for name in table.colnames:
        if name.startswith("tau_"):
            table[name].info.format = ".3e"
    table.meta.update(
        {
            "distance_pc": constants.DISTANCE_PC,
            "pixel_scale_arcsec": constants.PIXEL_SCALE_ARCSEC,
            "disk_peak_sb": data.disk_peak_sb,
            "disk_peak_radius_px": data.disk_peak_radius_px,
            "c_det": summary.get("c_det"),
            "alpha_det": summary.get("alpha_det"),
            "power_target": summary.get("power_target"),
            "noise_model": summary.get("noise_model"),
            "tau_prime_upper_limits": {f"{k:g}": v for k, v in uls.items()},
            "tau_prime_sensitivity": summary.get("sensitivity"),
        }
    )
    if out_path is not None:
        table.write(out_path, format="ascii.ecsv", overwrite=True)
    return table


def summary_lines(summary: dict) -> list[str]:
    """Human-readable one-liners of the calibrated statistics."""
    lines = [
        f"detection threshold c_det = {summary['c_det']:.3f} "
        f"(alpha_det = {summary['alpha_det']:g}, n_null = {summary['n_null']})",
    ]
    if "observed" in summary:
        obs = summary["observed"]
        lines.append(
            f"observed q0 = {obs['q0']:.3f}, p-value = {obs['p_value']:.4f} "
            f"({'DETECTION' if obs['q0'] > summary['c_det'] else 'no detection'})"
        )
    sens = summary.get("sensitivity")
    if sens is not None:
        lines.append(
            f"sensitivity (P_det >= {summary['power_target']:g}): tau' = {sens:.3e} "
            f"(conservative: {summary.get('sensitivity_conservative', float('nan')):.3e})"
        )
    for entry in summary.get("limits", {}).values():
        if "upper_limit" in entry:
            lines.append(
                f"tau' upper limit ({100 * entry['cl']:.1f}% CL, Neyman): "
                f"{entry['upper_limit']:.3e}"
            )
    if "warning_n_null" in summary:
        lines.append("WARNING: " + summary["warning_n_null"])
    return lines


def column_density_cgs(
    tau: np.ndarray | float,
    grain_radius_cm: float = constants.GRAIN_RADIUS_MEAN_CM,
    solid_density: float = constants.GRAIN_SOLID_DENSITY,
    q_ext: float = constants.Q_EXT,
) -> np.ndarray | float:
    """Dust column density ``Sigma = tau / kappa`` [g cm^-2].

    Uses the geometric-optics opacity per unit grain mass
    ``kappa = 3 Q_ext / (4 a rho_s)`` (~2.7e3 cm^2/g for 3.7 um astrosilicate
    grains), i.e. the inverse of the manuscript's column-density relation.
    """
    kappa = 3.0 * q_ext / (4.0 * grain_radius_cm * solid_density)
    return np.asarray(tau) / kappa
