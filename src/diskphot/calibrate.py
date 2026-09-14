"""Injection--recovery calibration of the profile-likelihood statistics.

``run_calibration`` simulates null and injected realizations, fits every
one exactly as the data (all realizations of one injection batched through
the GPU), and writes per-realization tables.  ``analyze_calibration`` turns
those tables plus the observed fit into the reported numbers: detection
threshold and p-value, Clopper--Pearson power curve and sensitivity, Neyman
critical curves, per-realization upper limits, and measured one-sided
coverage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from astropy.table import Table
from scipy.stats import binomtest

from .config import FitConfig, Variant
from .data import EpochData
from .inference import fit_mle, fit_offsets, profile_curve, tau_grid
from .model import JointLogLikelihood
from .synthetic import empirical_template, simulate_epochs

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ binomial
def clopper_pearson(k: int, n: int, cl: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper--Pearson) two-sided binomial interval via scipy."""
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=cl, method="exact")
    return float(ci.low), float(ci.high)


def detection_curve(trials: Table, c_det: float) -> Table:
    """Detection fraction vs injected tau with exact binomial intervals."""
    rows = []
    for tau in np.unique(np.asarray(trials["tau_inj"])):
        sel = np.asarray(trials["tau_inj"]) == tau
        n = int(sel.sum())
        k = int((np.asarray(trials["q0"])[sel] > c_det).sum())
        lo, hi = clopper_pearson(k, n)
        rows.append({"tau_inj": float(tau), "n": n, "k": k,
                     "frac": k / n, "frac_lo": lo, "frac_hi": hi})
    return Table(rows=rows)


def sensitivity_threshold(curve: Table, target: float = 0.95,
                          column: str = "frac") -> float:
    """Smallest injected tau whose ``column`` reaches ``target``.

    Log-interpolated between bracketing grid points; the first grid point if
    already above target there; NaN if the target is never reached.
    """
    taus = np.asarray(curve["tau_inj"], dtype=float)
    frac = np.asarray(curve[column], dtype=float)
    order = np.argsort(taus)
    taus, frac = taus[order], frac[order]
    if frac[0] >= target:
        return float(taus[0])
    above = np.nonzero(frac >= target)[0]
    if len(above) == 0:
        return float("nan")
    j = int(above[0])
    x0, x1 = np.log(taus[j - 1]), np.log(taus[j])
    f0, f1 = frac[j - 1], frac[j]
    t = 0.0 if f1 == f0 else (target - f0) / (f1 - f0)
    return float(np.exp(x0 + t * (x1 - x0)))


def upper_limit(grid: np.ndarray, q_obs: np.ndarray, crit: np.ndarray) -> float:
    """Neyman-inversion upper limit: largest tau with ``q(tau) <= c(tau)``.

    Log-interpolates the last accepted-to-rejected crossing; ``inf`` when no
    grid point is rejected (limit above the scanned range) and ``0`` when
    every point is rejected (below the range).
    """
    acc = np.asarray(q_obs) <= np.asarray(crit)
    if acc.all():
        return float("inf")
    if not acc.any():
        return 0.0
    j = int(np.nonzero(acc)[0].max())  # last accepted grid index
    if j == len(grid) - 1:
        return float(grid[j])
    x0, x1 = np.log(grid[j]), np.log(grid[j + 1])
    d0 = float(crit[j] - q_obs[j])          # >= 0 at the accepted point
    d1 = float(crit[j + 1] - q_obs[j + 1])  # < 0 at the rejected point
    t = 0.5 if d0 == d1 else d0 / (d0 - d1)
    return float(np.exp(x0 + np.clip(t, 0.0, 1.0) * (x1 - x0)))


def _critical_curve(trials: Table, grid: np.ndarray, cl: float) -> np.ndarray:
    """CL-quantile of ``q(tau)`` among realizations injected at that tau."""
    crit = np.empty(len(grid))
    tau_inj = np.asarray(trials["tau_inj"], dtype=float)
    for j, tau in enumerate(grid):
        sel = np.isclose(tau_inj, tau)
        crit[j] = np.quantile(np.asarray(trials[f"q_grid_{j}"])[sel], cl)
    return crit


# ----------------------------------------------------------------- campaign
def _fit_batch(data: EpochData, obs: torch.Tensor, cfg: FitConfig,
               grid: np.ndarray, truth) -> list[dict]:
    """Null fit, free fit and profile curve for one batch of realizations."""
    null = JointLogLikelihood(data, Variant.NULL, cfg.solve, observations=obs)
    res_null = fit_offsets(null, cfg.profile)
    model = JointLogLikelihood(data, Variant.COMMON, cfg.solve, observations=obs)
    res_free = fit_mle(model, cfg.profile, seed=cfg.seed)
    curve = profile_curve(model, cfg.profile, grid)   # (R, len(grid))
    lnl_hat = np.maximum(res_free.logl, res_null.logl)
    lnl_hat = np.maximum(lnl_hat, curve.max(axis=1))
    q0 = np.clip(2.0 * (lnl_hat - res_null.logl), 0.0, None)
    q_grid = np.clip(2.0 * (lnl_hat[:, None] - curve), 0.0, None)
    off = res_free.theta[:, : model.n_offset_params].reshape(len(q0), -1, 2)
    err = off - truth.offsets_xy[:, 1:]
    rows = []
    for r in range(len(q0)):
        row = {
            "q0": float(q0[r]),
            "tau_hat": float(res_free.theta[r, -1]),
            "offset_rms_err": float(np.sqrt(np.mean(err[r] ** 2))),
        }
        row.update({f"q_grid_{j}": float(q_grid[r, j]) for j in range(len(grid))})
        rows.append(row)
    return rows


def run_calibration(cfg: FitConfig, data: EpochData, out_dir: Path) -> None:
    """Simulate, fit and tabulate the null ensemble and every injection."""
    if cfg.variant is not Variant.COMMON:
        raise ValueError("calibration is defined for the COMMON science variant")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = np.asarray(cfg.calib.taus, dtype=float) if cfg.calib.taus else tau_grid(cfg.profile)

    template, _ = empirical_template(data, cfg.profile, cfg.solve)

    # Null ensemble -> detection-threshold calibration.
    obs, truth = simulate_epochs(
        data, 0.0, template, cfg.calib.n_null, Variant.COMMON,
        cfg.calib, add_noise=True, seed=cfg.seed,
    )
    rows = _fit_batch(data, obs, cfg, grid, truth)
    nulls = Table(rows=[{"q0": r["q0"]} for r in rows])
    nulls.write(out_dir / "nulls.ecsv", format="ascii.ecsv", overwrite=True)
    log.info("null ensemble done (%d realizations)", cfg.calib.n_null)

    # Injections.
    all_rows: list[dict] = []
    for i, tau in enumerate(grid):
        obs, truth = simulate_epochs(
            data, float(tau), template, cfg.calib.n_repeats, Variant.COMMON,
            cfg.calib, add_noise=True, seed=cfg.seed + 1 + i,
        )
        rows = _fit_batch(data, obs, cfg, grid, truth)
        for r in rows:
            r["tau_inj"] = float(tau)
        all_rows.extend(rows)
        log.info("injection tau=%.3e done (%d realizations)", tau, cfg.calib.n_repeats)
    trials = Table(rows=all_rows)
    trials.meta["tau_grid"] = [float(v) for v in grid]
    trials.write(out_dir / "trials.ecsv", format="ascii.ecsv", overwrite=True)


# ------------------------------------------------------------------ analysis
def analyze_calibration(
    trials: Table,
    nulls: Table,
    cfg: FitConfig,
    observed: dict | None = None,
    out_path: Path | None = None,
) -> dict:
    """Thresholds, power, sensitivity, limits and coverage from the tables."""
    calib = cfg.calib
    grid = np.asarray(trials.meta.get("tau_grid", sorted(set(np.asarray(trials["tau_inj"])))))
    q0_null = np.sort(np.asarray(nulls["q0"], dtype=float))
    c_det = float(np.quantile(q0_null, 1.0 - calib.alpha_det))

    curve = detection_curve(trials, c_det)
    summary: dict = {
        "alpha_det": calib.alpha_det,
        "c_det": c_det,
        "n_null": len(q0_null),
        "tau_grid": [float(v) for v in grid],
        "detection_curve": [dict(zip(curve.colnames, (float(v) for v in row))) for row in curve],
        "sensitivity_tau": sensitivity_threshold(curve, calib.power_target),
        "sensitivity_tau_conservative": sensitivity_threshold(
            curve, calib.power_target, column="frac_lo"
        ),
    }
    if len(q0_null) < 2.0 / calib.alpha_det:
        summary["warning_n_null"] = (
            f"n_null = {len(q0_null)} < 2/alpha_det = {2.0 / calib.alpha_det:.0f}: "
            "the detection threshold quantile is poorly resolved"
        )

    tau_inj = np.asarray(trials["tau_inj"], dtype=float)
    q_grid = np.stack([np.asarray(trials[f"q_grid_{j}"], dtype=float)
                       for j in range(len(grid))], axis=1)
    summary["limits"] = {}
    for cl in calib.cl_levels:
        crit = _critical_curve(trials, grid, cl)
        uls = np.array([upper_limit(grid, q_grid[r], crit) for r in range(len(trials))])
        cov_rows = []
        for tau in grid:
            sel = np.isclose(tau_inj, tau)
            n = int(sel.sum())
            k = int((uls[sel] >= tau).sum())
            lo, hi = clopper_pearson(k, n)
            cov_rows.append({"tau_inj": float(tau), "coverage": k / n,
                             "coverage_lo": lo, "coverage_hi": hi})
        entry: dict = {
            "critical_curve": [float(v) for v in crit],
            "coverage": cov_rows,
        }
        if observed is not None and "logl_curve" in observed:
            lnl_hat = max(float(observed["logl_mle"]), float(observed["logl_null"]),
                          float(np.max(observed["logl_curve"])))
            q_obs = np.clip(2.0 * (lnl_hat - np.asarray(observed["logl_curve"])), 0.0, None)
            entry["upper_limit_observed"] = upper_limit(grid, q_obs, crit)
        summary["limits"][f"{cl:g}"] = entry

    if observed is not None and "q0" in observed:
        k = int((q0_null >= float(observed["q0"])).sum())
        summary["observed_q0"] = float(observed["q0"])
        summary["observed_q0_pvalue"] = (k + 1) / (len(q0_null) + 1)
        summary["observed_detected"] = bool(float(observed["q0"]) > c_det)

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(summary, indent=2))
    return summary
