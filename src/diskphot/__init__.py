"""diskphot -- debris-disk transit photometry of HD 107146.

Joint forward modelling of the "Vermin" background galaxy through the
HD 107146 debris disk across eight HST/STIS epochs, on the native pixel
grids (the data are never resampled).  The galaxy template is profiled out
exactly (ridge-regularized dense normal equations, one solver everywhere),
and every quantitative statement -- detection statistic, sensitivity, and
confidence upper limits on the disk optical depth -- is frequentist and
calibrated by injection--recovery simulation.

Submodules are imported lazily (PEP 562) so that ``import diskphot`` stays
cheap; matplotlib is only imported inside plotting functions.
"""

from __future__ import annotations

import importlib

__version__ = "0.4.0"

_API = {
    # config
    "Variant": "config", "TauPrior": "config", "DataConfig": "config",
    "SolveConfig": "config", "ProfileConfig": "config", "CalibConfig": "config",
    "ExploreConfig": "config", "FitConfig": "config",
    # data
    "EpochData": "data", "load_epoch_data": "data", "load_full_frames": "data",
    "make_ellipse_mask": "data", "variance_map": "data",
    "read_positions_csv": "data", "estimate_star_position": "data",
    "ring_profile": "data",
    # model / transforms
    "JointLogLikelihood": "model", "shift": "transforms", "translate": "transforms",
    # inference
    "prepare_data": "inference", "fit_offsets": "inference", "fit_mle": "inference",
    "profile_curve": "inference", "run_fit": "inference", "explore": "inference",
    "tau_grid": "inference", "FitResult": "inference",
    # synthetic / calibration
    "SyntheticTruth": "synthetic", "empirical_template": "synthetic",
    "simulate_epochs": "synthetic",
    "run_calibration": "calibrate", "analyze_calibration": "calibrate",
    "clopper_pearson": "calibrate", "detection_curve": "calibrate",
    "sensitivity_threshold": "calibrate", "upper_limit": "calibrate",
    # results / diagnostics
    "ObservedFit": "results", "load_fit": "results",
    "load_calibration_summary": "results", "epoch_table": "results",
    "summary_lines": "results", "column_density_cgs": "results",
    "run_verify": "diagnostics", "run_roundtrip": "diagnostics",
    "nuisance_spectrum": "diagnostics", "make_synthetic_epoch_data": "diagnostics",
}


def __getattr__(name: str):
    if name in _API:
        value = getattr(importlib.import_module(f".{_API[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'diskphot' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_API))
