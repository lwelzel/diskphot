r"""Simulation calibration of the profile-likelihood statistics.

The pipeline's quantitative statements are frequentist and are calibrated
here by injection--recovery on the parametric bootstrap of
:mod:`diskphot.synthetic`.  Three statistics are calibrated, all built from
the exactly profiled log-likelihood :math:`\ln L_{\rm p}` and all free of
any prior on :math:`\tau'`:

Detection statistic
    :math:`q_0 = 2\,[\ln L_{\rm p}(\hat\tau') - \ln L_{\rm p}(0)]`
    (:math:`\hat\tau' \ge 0`).  Its null distribution is estimated from
    ``n_null`` simulations at :math:`\tau' = 0`; the detection threshold
    :math:`c_{\rm det}` is the empirical :math:`(1-\alpha_{\rm det})`
    quantile.

Sensitivity (power)
    For each injected :math:`\tau'` the detection probability
    :math:`P(q_0 > c_{\rm det})` is estimated with a Clopper--Pearson
    binomial interval; the sensitivity :math:`\tau'_{\rm sens}` is the
    smallest injected value whose detection probability reaches the power
    target (default 95%), interpolated in :math:`\log\tau'`.  This is the
    headline exclusion: optical depths above :math:`\tau'_{\rm sens}` would
    have been detected with the stated probability.

Confidence upper limit
    Neyman construction on the profile-likelihood-ratio curve
    :math:`q(\tau') = 2\,[\ln L_{\rm p}(\hat\tau') - \ln L_{\rm p}(\tau')]`:
    the critical value :math:`c_{\rm CL}(\tau')` is the CL-quantile of
    :math:`q(\tau')` among simulations injected *at* :math:`\tau'`, and the
    upper limit is the largest grid :math:`\tau'` not rejected,
    :math:`q_{\rm obs}(\tau') \le c_{\rm CL}(\tau')` (log-interpolated
    crossing).  Because every realization's full curve is fitted, the
    one-sided coverage of the reported limit is measured directly: the
    fraction of realizations at each injected :math:`\tau'` whose limit
    contains the truth.

Every simulated realization is fitted exactly like the data (same weights,
same CG-profiled template, same optimizer settings); the fits are
gradient-based, so hundreds of realizations per grid point are affordable
without a nested-sampling run per realization.  The calibration is
restricted to the common-:math:`\tau'` variant; the per-epoch variant is
validated separately (all coefficients reported) by the round-trip
diagnostics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from astropy.table import Table
from scipy.stats import beta as beta_dist

from .config import FitConfig, Variant
from .data import EpochData
from .inference import fit_mle, fit_offsets, profile_curve, tau_grid
from .model import JointLogLikelihood
from .synthetic import empirical_template, simulate_epochs

log = logging.getLogger(__name__)

__all__ = [
    "analyze_calibration",
    "clopper_pearson",
    "detection_curve",
    "run_calibration",
    "sensitivity_threshold",
    "upper_limit",
]


# --------------------------------------------------------------------------- #
# Campaign
# --------------------------------------------------------------------------- #
def _fit_realizations(
    cfg: FitConfig,
    data: EpochData,
    observations: torch.Tensor,
    with_curve: bool,
    grid: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    """Null fit, free fit and (optionally) the profile curve, batched over R."""
    null = JointLogLikelihood(data, Variant.NULL, cfg.cg, observations=observations)
    free = JointLogLikelihood(data, Variant.COMMON, cfg.cg, observations=observations)
    res_null = fit_offsets(null, cfg.profile)
    res_free = fit_mle(free, cfg.profile, seed=seed)
    out = {
        "logl_null": res_null.logl,
        "logl_mle": res_free.logl,
        "tau_mle": res_free.theta[:, -1],
        "offsets_mle": res_free.theta[:, : free.n_offset_params],
    }
    if with_curve:
        off0 = torch.as_tensor(
            res_free.theta[:, : free.n_offset_params],
            dtype=data.mask.dtype,
            device=data.mask.device,
        )
        out["curve"] = profile_curve(free, cfg.profile, grid, offsets0=off0)  # (G, R)
    return out


def run_calibration(
    cfg: FitConfig,
    data: EpochData,
    out_dir: Path | None = None,
) -> tuple[Table, Table]:
    """Run the full injection--recovery campaign.

    Returns ``(trials, nulls)``: the per-realization table of injected
    trials (with the full profile curve of every realization) and the
    ``q0`` table of the null simulations.  Both are written as ECSV to
    ``out_dir`` together with the grid metadata.
    """
    if cfg.variant is not Variant.COMMON:
        raise ValueError(
            "the calibration campaign is defined for the common-tau variant; "
            "validate other variants with the roundtrip diagnostics"
        )
    calib = cfg.calib
    grid = np.asarray(calib.taus, dtype=float) if calib.taus else tau_grid(cfg.profile)
    out_dir = Path(out_dir) if out_dir is not None else cfg.run_dir / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(out_dir / "config.json")

    template, _ = empirical_template(data, cfg.profile, cfg.cg)
    log.info(
        "calibration: %d injected taus x %d repeats + %d nulls",
        len(grid),
        calib.n_repeats,
        calib.n_null,
    )

    # ------------------------------------------------------- null ensemble --
    obs0, _ = simulate_epochs(
        data, 0.0, template, n_realizations=calib.n_null, variant=Variant.COMMON,
        calib=calib, seed=cfg.seed,
    )
    fits0 = _fit_realizations(cfg, data, obs0, with_curve=False, grid=grid, seed=cfg.seed)
    q0_null = np.clip(2.0 * (fits0["logl_mle"] - fits0["logl_null"]), 0.0, None)
    nulls = Table(
        {
            "rep": np.arange(calib.n_null),
            "logl_null": fits0["logl_null"],
            "logl_mle": fits0["logl_mle"],
            "tau_mle": fits0["tau_mle"],
            "q0": q0_null,
        }
    )
    nulls.meta["noise_model"] = calib.noise_model
    nulls.write(out_dir / "nulls.ecsv", format="ascii.ecsv", overwrite=True)

    # --------------------------------------------------- injected ensembles --
    rows: list[dict] = []
    for j, tau in enumerate(grid):
        seed = cfg.seed + 1000 * (j + 1)
        obs, truth = simulate_epochs(
            data, float(tau), template, n_realizations=calib.n_repeats,
            variant=Variant.COMMON, calib=calib, seed=seed,
        )
        fits = _fit_realizations(cfg, data, obs, with_curve=True, grid=grid, seed=seed)
        lnl_hat = np.maximum(
            np.maximum(fits["logl_mle"], fits["logl_null"]), fits["curve"].max(axis=0)
        )
        q0 = np.clip(2.0 * (lnl_hat - fits["logl_null"]), 0.0, None)
        q_curve = np.clip(2.0 * (lnl_hat[None, :] - fits["curve"]), 0.0, None)  # (G, R)
        off_err = np.sqrt(
            np.mean(
                (fits["offsets_mle"].reshape(-1, data.n_epochs - 1, 2)
                 - truth.offsets_xy[:, 1:]) ** 2,
                axis=(1, 2),
            )
        )
        for r in range(calib.n_repeats):
            row = {
                "tau_inj": float(tau),
                "rep": r,
                "seed": seed,
                "logl_null": float(fits["logl_null"][r]),
                "logl_mle": float(lnl_hat[r]),
                "tau_mle": float(fits["tau_mle"][r]),
                "q0": float(q0[r]),
                "offset_rms_err": float(off_err[r]),
            }
            row |= {f"q_grid_{g}": float(q_curve[g, r]) for g in range(len(grid))}
            rows.append(row)
        log.info(
            "tau_inj=%.3e: median q0=%.2f, median tau_mle=%.3e",
            tau, float(np.median(q0)), float(np.median(fits["tau_mle"])),
        )
        trials = Table(rows=rows)
        trials.meta["tau_grid"] = [float(v) for v in grid]
        trials.meta["noise_model"] = calib.noise_model
        trials.write(out_dir / "trials.ecsv", format="ascii.ecsv", overwrite=True)

    return trials, nulls


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def clopper_pearson(k: int, n: int, cl: float = 0.95) -> tuple[float, float]:
    """Two-sided Clopper--Pearson interval for a binomial proportion."""
    alpha = 1.0 - cl
    lo = 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def detection_curve(trials: Table, c_det: float, cl: float = 0.95) -> Table:
    """Detection probability per injected tau with Clopper--Pearson bounds."""
    rows = []
    for tau in np.unique(trials["tau_inj"]):
        sel = trials["tau_inj"] == tau
        n = int(sel.sum())
        k = int((np.asarray(trials["q0"][sel]) > c_det).sum())
        lo, hi = clopper_pearson(k, n, cl)
        rows.append(
            {"tau_inj": float(tau), "n": n, "k": k,
             "frac": k / n, "frac_lo": lo, "frac_hi": hi}
        )
    return Table(rows=rows)


def sensitivity_threshold(
    curve: Table, target: float = 0.95, column: str = "frac"
) -> float:
    """Smallest injected tau whose detection probability reaches ``target``.

    Log-interpolates the (assumed monotone) power curve; returns ``nan``
    when the target is never reached and the smallest grid value when even
    the lowest injection exceeds the target.  Use ``column="frac_lo"`` for
    the conservative (binomial lower-bound) threshold.
    """
    taus = np.asarray(curve["tau_inj"], dtype=float)
    frac = np.asarray(curve[column], dtype=float)
    order = np.argsort(taus)
    taus, frac = taus[order], frac[order]
    above = frac >= target
    if not above.any():
        return float("nan")
    j = int(np.argmax(above))
    if j == 0:
        return float(taus[0])
    x0, x1 = np.log(taus[j - 1]), np.log(taus[j])
    y0, y1 = frac[j - 1], frac[j]
    if y1 == y0:
        return float(taus[j])
    return float(np.exp(x0 + (target - y0) * (x1 - x0) / (y1 - y0)))


def _critical_curve(trials: Table, grid: np.ndarray, cl: float) -> np.ndarray:
    """CL-quantile of q(tau') among realizations injected at each grid tau'."""
    crit = np.empty(len(grid))
    for g, tau in enumerate(grid):
        sel = np.asarray(trials["tau_inj"]) == tau
        q_at_truth = np.asarray(trials[f"q_grid_{g}"][sel], dtype=float)
        crit[g] = float(np.quantile(q_at_truth, cl))
    return crit


def upper_limit(grid: np.ndarray, q_curve: np.ndarray, crit: np.ndarray) -> float:
    """Largest tau' not rejected: ``q(tau') <= c(tau')`` (log-interpolated).

    Returns ``inf`` when no grid point is rejected (limit above the grid)
    and ``0`` when every grid point is rejected (limit below the grid).
    """
    excess = np.asarray(q_curve, dtype=float) - np.asarray(crit, dtype=float)
    accepted = excess <= 0
    if accepted.all():
        return float("inf")
    if not accepted.any():
        return 0.0
    j_last = int(np.max(np.nonzero(accepted)))
    if j_last == len(grid) - 1:
        return float(grid[-1])
    x0, x1 = np.log(grid[j_last]), np.log(grid[j_last + 1])
    y0, y1 = excess[j_last], excess[j_last + 1]
    return float(np.exp(x0 - y0 * (x1 - x0) / (y1 - y0)))


def analyze_calibration(
    trials: Table,
    nulls: Table,
    cfg: FitConfig,
    observed: dict | None = None,
    out_path: Path | None = None,
) -> dict:
    """Assemble the calibrated statistics (and observed limits, if given).

    Returns a dict with the detection threshold, the power curve and
    sensitivity, the per-CL critical curves, the measured one-sided coverage
    of the upper-limit construction, and -- when the observed fit payload of
    :func:`diskphot.inference.run_fit` is supplied -- the observed ``q0``
    p-value and upper limits.
    """
    calib = cfg.calib
    grid = np.asarray(trials.meta["tau_grid"], dtype=float)
    q0_null = np.asarray(nulls["q0"], dtype=float)
    c_det = float(np.quantile(q0_null, 1.0 - calib.alpha_det))

    power = detection_curve(trials, c_det)
    out: dict = {
        "alpha_det": calib.alpha_det,
        "c_det": c_det,
        "n_null": len(q0_null),
        "tau_grid": grid.tolist(),
        "power": {
            "tau_inj": [float(v) for v in power["tau_inj"]],
            "frac": [float(v) for v in power["frac"]],
            "frac_lo": [float(v) for v in power["frac_lo"]],
            "frac_hi": [float(v) for v in power["frac_hi"]],
        },
        "sensitivity": sensitivity_threshold(power, calib.power_target),
        "sensitivity_conservative": sensitivity_threshold(
            power, calib.power_target, column="frac_lo"
        ),
        "power_target": calib.power_target,
        "noise_model": calib.noise_model,
    }
    if len(q0_null) < 2.0 / calib.alpha_det:
        out["warning_n_null"] = (
            f"n_null={len(q0_null)} is small for alpha_det={calib.alpha_det}; "
            "the threshold quantile is poorly estimated"
        )

    # Neyman critical curves, per-realization limits, and one-sided coverage.
    q_cols = np.stack(
        [np.asarray(trials[f"q_grid_{g}"], dtype=float) for g in range(len(grid))]
    )  # (G, n_trials)
    taus_inj = np.asarray(trials["tau_inj"], dtype=float)
    out["limits"] = {}
    for cl in calib.cl_levels:
        crit = _critical_curve(trials, grid, cl)
        uls = np.array(
            [upper_limit(grid, q_cols[:, i], crit) for i in range(q_cols.shape[1])]
        )
        coverage = []
        for tau in np.unique(taus_inj):
            sel = taus_inj == tau
            n = int(sel.sum())
            k = int((uls[sel] >= tau).sum())
            lo, hi = clopper_pearson(k, n)
            coverage.append(
                {"tau_inj": float(tau), "coverage": k / n,
                 "coverage_lo": lo, "coverage_hi": hi}
            )
        entry = {
            "cl": cl,
            "critical_curve": crit.tolist(),
            "coverage": coverage,
        }
        if observed is not None and "logl_curve" in observed:
            lnl_hat = max(
                float(observed["logl_mle"]),
                float(observed["logl_null"]),
                float(np.max(observed["logl_curve"])),
            )
            q_obs = np.clip(2.0 * (lnl_hat - np.asarray(observed["logl_curve"])), 0.0, None)
            entry["upper_limit"] = upper_limit(grid, q_obs, crit)
        out["limits"][f"cl_{cl:g}"] = entry

    if observed is not None and "q0" in observed:
        q0_obs = float(observed["q0"])
        k = int((q0_null >= q0_obs).sum())
        out["observed"] = {
            "q0": q0_obs,
            # add-one estimator: never returns exactly zero
            "p_value": (k + 1) / (len(q0_null) + 1),
        }

    if out_path is not None:
        Path(out_path).write_text(json.dumps(out, indent=2))
    return out
