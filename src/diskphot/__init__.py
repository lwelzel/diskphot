"""diskphot -- debris-disk transit photometry of HD 107146.

Joint forward modelling of the "Vermin" background galaxy through the
HD 107146 debris disk across eight HST/STIS epochs.  The galaxy template is
profiled out *exactly* (conjugate-gradient solve of the template normal
equations with a certified tolerance), the observed data are never
resampled (native-grid cutouts; sub-pixel geometry lives in the model), and
every quantitative statement -- detection statistic, sensitivity, and
confidence upper limits on the disk optical depth -- is frequentist and
calibrated by injection--recovery simulation.
"""

from .calibrate import analyze_calibration, run_calibration, upper_limit
from .config import (
    CalibConfig,
    CGConfig,
    DataConfig,
    ExploreConfig,
    FitConfig,
    ProfileConfig,
    TauPrior,
    Variant,
)
from .data import (
    EpochData,
    estimate_star_position,
    load_epoch_data,
    load_full_frames,
    make_ellipse_mask,
    read_positions_csv,
    ring_profile,
    variance_map,
)
from .diagnostics import (
    make_synthetic_epoch_data,
    nuisance_spectrum,
    run_roundtrip,
    run_verify,
)
from .inference import (
    explore,
    fit_mle,
    fit_offsets,
    prepare_data,
    profile_curve,
    run_fit,
    tau_grid,
)
from .model import JointLogLikelihood
from .results import (
    ObservedFit,
    column_density_cgs,
    epoch_table,
    load_calibration_summary,
    load_fit,
    summary_lines,
)
from .synthetic import empirical_template, simulate_epochs
from .transforms import shift2d, translate

__version__ = "0.3.0"

__all__ = [
    "CGConfig",
    "CalibConfig",
    "DataConfig",
    "EpochData",
    "ExploreConfig",
    "FitConfig",
    "JointLogLikelihood",
    "ObservedFit",
    "ProfileConfig",
    "TauPrior",
    "Variant",
    "analyze_calibration",
    "column_density_cgs",
    "empirical_template",
    "epoch_table",
    "estimate_star_position",
    "explore",
    "fit_mle",
    "fit_offsets",
    "load_calibration_summary",
    "load_epoch_data",
    "load_fit",
    "load_full_frames",
    "make_ellipse_mask",
    "make_synthetic_epoch_data",
    "nuisance_spectrum",
    "prepare_data",
    "profile_curve",
    "read_positions_csv",
    "ring_profile",
    "run_calibration",
    "run_fit",
    "run_roundtrip",
    "run_verify",
    "shift2d",
    "simulate_epochs",
    "summary_lines",
    "tau_grid",
    "translate",
    "upper_limit",
    "variance_map",
]
