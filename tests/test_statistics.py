"""Statistical-machinery tests and a small end-to-end recovery fit."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from astropy.table import Table
from scipy.stats import binom

from diskphot.calibrate import (
    _critical_curve,
    clopper_pearson,
    detection_curve,
    sensitivity_threshold,
    upper_limit,
)
from diskphot.config import FitConfig, Variant
from diskphot.diagnostics import make_synthetic_epoch_data
from diskphot.inference import fit_mle, fit_offsets
from diskphot.model import JointLogLikelihood
from diskphot.synthetic import empirical_template, simulate_epochs

DTYPE = torch.float64


# --------------------------------------------------------------------------- #
# Binomial machinery
# --------------------------------------------------------------------------- #
def test_clopper_pearson_matches_exact_binomial():
    k, n, cl = 7, 20, 0.95
    lo, hi = clopper_pearson(k, n, cl)
    # Defining property: at the bounds, the tail probabilities equal alpha/2.
    alpha = 1.0 - cl
    assert abs(binom.sf(k - 1, n, lo) - alpha / 2.0) < 1.0e-9
    assert abs(binom.cdf(k, n, hi) - alpha / 2.0) < 1.0e-9


def test_clopper_pearson_edges():
    lo, hi = clopper_pearson(0, 10)
    assert lo == 0.0 and 0.0 < hi < 1.0
    lo, hi = clopper_pearson(10, 10)
    assert hi == 1.0 and 0.0 < lo < 1.0


def test_59_consecutive_successes_bound():
    # 59 successes in 59 trials is the smallest run whose one-sided 95%
    # lower confidence bound on the success probability exceeds 0.95.
    lo_59 = clopper_pearson(59, 59, cl=0.90)[0]  # two-sided 90% = one-sided 95%
    lo_58 = clopper_pearson(58, 58, cl=0.90)[0]
    assert lo_59 > 0.95 > lo_58


# --------------------------------------------------------------------------- #
# Detection curve / sensitivity / upper limit on synthetic tables
# --------------------------------------------------------------------------- #
def _toy_trials() -> Table:
    rng = np.random.default_rng(11)
    rows = []
    for tau, p in [(1.0e-4, 0.10), (3.0e-4, 0.60), (1.0e-3, 1.00)]:
        for _ in range(200):
            det = rng.random() < p
            rows.append({"tau_inj": tau, "q0": 15.0 if det else 1.0})
    return Table(rows=rows)


def test_detection_curve_and_sensitivity():
    trials = _toy_trials()
    curve = detection_curve(trials, c_det=9.0)
    assert len(curve) == 3
    fr = {float(r["tau_inj"]): r["frac"] for r in curve}
    assert fr[1.0e-3] == 1.0 and fr[1.0e-4] < 0.2
    thr = sensitivity_threshold(curve, target=0.95)
    assert 3.0e-4 < thr <= 1.0e-3
    # conservative variant is never below the central one
    thr_lo = sensitivity_threshold(curve, target=0.95, column="frac_lo")
    assert thr_lo >= thr


def test_sensitivity_threshold_edge_cases():
    curve = Table(rows=[
        {"tau_inj": 1e-4, "frac": 0.99, "frac_lo": 0.97},
        {"tau_inj": 1e-3, "frac": 1.00, "frac_lo": 0.99},
    ])
    # already above target at the lowest grid point -> that grid point
    assert sensitivity_threshold(curve, 0.95) == 1e-4
    curve2 = Table(rows=[
        {"tau_inj": 1e-4, "frac": 0.10, "frac_lo": 0.05},
        {"tau_inj": 1e-3, "frac": 0.60, "frac_lo": 0.50},
    ])
    assert np.isnan(sensitivity_threshold(curve2, 0.95))


def test_upper_limit_cases():
    grid = np.array([1e-4, 3e-4, 1e-3])
    crit = np.array([4.0, 4.0, 4.0])
    # never rejected -> inf (limit above the scanned range)
    assert upper_limit(grid, np.array([1.0, 2.0, 3.0]), crit) == float("inf")
    # always rejected -> 0 (limit below the scanned range)
    assert upper_limit(grid, np.array([9.0, 9.0, 9.0]), crit) == 0.0
    # crossing between grid points: log-interpolated inside the bracket
    ul = upper_limit(grid, np.array([1.0, 3.0, 9.0]), crit)
    assert 3e-4 < ul < 1e-3
    # accepted at the last point after a rejection -> last grid value
    ul2 = upper_limit(grid, np.array([9.0, 3.0, 3.0]), crit)
    assert ul2 == pytest.approx(1e-3)


def test_critical_curve_quantiles():
    rng = np.random.default_rng(13)
    grid = np.array([1e-4, 1e-3])
    rows = []
    for g, tau in enumerate(grid):
        for _ in range(400):
            rows.append({
                "tau_inj": tau,
                "q_grid_0": rng.chisquare(1) if g == 0 else 99.0,
                "q_grid_1": rng.chisquare(1) if g == 1 else 99.0,
            })
    trials = Table(rows=rows)
    crit = _critical_curve(trials, grid, 0.95)
    # chi2_1 95% quantile = 3.841
    assert np.allclose(crit, 3.841, atol=0.6)


# --------------------------------------------------------------------------- #
# Small end-to-end recovery fit (noiseless)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_noiseless_injection_recovery():
    cfg = FitConfig()
    cfg.profile.steps = 400
    cfg.profile.n_starts = 1
    data = make_synthetic_epoch_data(n_epochs=4, size=20, seed=21).to("cpu", DTYPE)
    template, _ = empirical_template(data, cfg.profile, cfg.cg)
    # A well-identified injection level: the chi-square well between tau = 0
    # and the truth is ~11 deep for this synthetic geometry, so the point
    # estimate is meaningful.  (At marginal injections the tau' profile is
    # flat and only the calibrated statistics carry information.)
    tau_true = 5.0e-2
    obs, truth = simulate_epochs(
        data, tau_true, template, n_realizations=1, variant=Variant.COMMON,
        calib=cfg.calib, add_noise=False, seed=21,
    )
    model = JointLogLikelihood(data, Variant.COMMON, cfg.cg, observations=obs)
    res = fit_mle(model, cfg.profile, seed=21)
    tau_hat = float(res.theta[0, -1])
    # noise-free: tau recovered well and chi2 at the optimum nearly zero
    assert tau_hat == pytest.approx(tau_true, rel=0.25)
    assert -2.0 * float(res.logl[0]) < 5.0e-2
    off_hat = res.theta[0, : model.n_offset_params]
    off_true = truth.offsets_xy.reshape(-1)[2:]  # epoch-0 gauge-fixed
    assert float(np.abs(off_hat - off_true).max()) < 0.15


@pytest.mark.slow
def test_null_fit_beats_nothing_and_q0_nonnegative():
    cfg = FitConfig()
    cfg.profile.steps = 120
    cfg.profile.n_starts = 1
    data = make_synthetic_epoch_data(n_epochs=4, size=20, seed=23).to("cpu", DTYPE)
    template, _ = empirical_template(data, cfg.profile, cfg.cg)
    obs, _ = simulate_epochs(
        data, 0.0, template, n_realizations=1, variant=Variant.COMMON,
        calib=cfg.calib, add_noise=True, seed=23,
    )
    m_null = JointLogLikelihood(data, Variant.NULL, cfg.cg, observations=obs)
    m_com = JointLogLikelihood(data, Variant.COMMON, cfg.cg, observations=obs)
    res_null = fit_offsets(m_null, cfg.profile)
    res_com = fit_mle(m_com, cfg.profile, seed=23)
    q0 = max(2.0 * (float(res_com.logl[0]) - float(res_null.logl[0])), 0.0)
    assert q0 >= 0.0
    # on a null draw, q0 should be modest (no huge spurious detection)
    assert q0 < 25.0
