"""Calibrated statistics: binomial intervals, power, limits, edge cases."""

import numpy as np
import pytest
from astropy.table import Table

from diskphot.calibrate import (
    _critical_curve,
    clopper_pearson,
    detection_curve,
    sensitivity_threshold,
    upper_limit,
)


def test_clopper_pearson_endpoints():
    lo, hi = clopper_pearson(0, 20)
    assert lo == 0.0 and 0.0 < hi < 0.3
    lo, hi = clopper_pearson(20, 20)
    assert hi == 1.0 and 0.7 < lo < 1.0


def test_59_successes_rule():
    lo, _ = clopper_pearson(59, 59, cl=0.90)  # one-sided 95% lower bound
    assert lo > 0.95
    lo, _ = clopper_pearson(58, 59, cl=0.90)
    assert lo < 0.95


def _toy_trials():
    rows = []
    for tau, frac in ((1e-4, 0.0), (1e-3, 0.5), (1e-2, 1.0)):
        for r in range(20):
            det = r < int(round(frac * 20))
            rows.append({
                "tau_inj": tau,
                "q0": 20.0 if det else 0.5,
                "q_grid_0": 0.1, "q_grid_1": 0.2, "q_grid_2": 0.3,
            })
    t = Table(rows=rows)
    t.meta["tau_grid"] = [1e-4, 1e-3, 1e-2]
    return t


def test_detection_curve_and_sensitivity():
    trials = _toy_trials()
    curve = detection_curve(trials, c_det=5.0)
    assert list(curve["frac"]) == [0.0, 0.5, 1.0]
    sens = sensitivity_threshold(curve, target=0.95)
    assert 1e-3 < sens <= 1e-2


def test_sensitivity_edge_cases():
    curve = Table(rows=[{"tau_inj": 1e-3, "frac": 0.99, "frac_lo": 0.9},
                        {"tau_inj": 1e-2, "frac": 1.0, "frac_lo": 0.99}])
    assert sensitivity_threshold(curve, 0.95) == pytest.approx(1e-3)
    curve2 = Table(rows=[{"tau_inj": 1e-3, "frac": 0.1},
                         {"tau_inj": 1e-2, "frac": 0.6}])
    assert np.isnan(sensitivity_threshold(curve2, 0.95))


def test_upper_limit_edge_cases():
    grid = np.array([1e-4, 1e-3, 1e-2])
    crit = np.array([4.0, 4.0, 4.0])
    assert upper_limit(grid, np.array([1.0, 1.0, 1.0]), crit) == np.inf
    assert upper_limit(grid, np.array([9.0, 9.0, 9.0]), crit) == 0.0
    ul = upper_limit(grid, np.array([1.0, 2.0, 9.0]), crit)
    assert 1e-3 < ul < 1e-2
    assert upper_limit(grid, np.array([9.0, 2.0, 1.0]), crit) == pytest.approx(1e-2)


def test_critical_curve_quantiles():
    trials = _toy_trials()
    crit = _critical_curve(trials, np.array([1e-4, 1e-3, 1e-2]), cl=0.5)
    assert crit == pytest.approx([0.1, 0.2, 0.3])
