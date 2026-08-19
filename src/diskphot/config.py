"""Run configuration.

All knobs of the pipeline live in small frozen dataclasses that serialize to
JSON next to every run directory, so any result on disk carries the exact
configuration that produced it.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from . import constants


class Variant(str, Enum):
    """Extinction-model variants (see :mod:`diskphot.model`)."""

    #: Single tau' scaling the normalized disk surface brightness (primary).
    COMMON = "common"
    #: One tau'_i per epoch, each scaling the normalized disk SB (appendix).
    PER_EPOCH = "per-epoch"
    #: Per-epoch grey screens exp(-gamma_i), gamma_0 = 0, gamma_i signed.
    GREY = "grey"
    #: No extinction (tau = 0); null model of the detection statistic.
    NULL = "null"


class TauPrior(str, Enum):
    """Parameter mapping used only by the optional objective-map explorer."""

    LOG_UNIFORM = "log-uniform"
    UNIFORM = "uniform"


@dataclass(slots=True)
class DataConfig:
    """Paths and preprocessing options."""

    data_dir: Path = Path("data")
    obs_dirname: str = "hd107146_reduced"
    obs_pattern: str = "HD107146_EP*_NUP_MED.fits"
    disk_model_path: Path = Path("hd107146_disk_model/MODEL20_HD107146_16.5_47_h=0.3_NUP.fits")
    #: CSV with approximate per-epoch galaxy centres in full-frame (x, y)
    #: pixel coordinates (columns ``epoch, x, y``; other columns ignored).
    #: Looked up relative to ``data_dir``; ``None`` or a missing file falls
    #: back to automatic estimation.  Positions are refined locally either way.
    galaxy_positions_csv: Path | None = Path("galaxy_positions.csv")

    #: Lower-left corner (x0, y0) and side length of the working region that
    #: contains the galaxy in every epoch (full-frame pixel coordinates).
    region_origin_xy: tuple[int, int] = (300, 60)
    region_size: int = 128
    #: Side length of the final analysis cutout.  Each epoch is cropped at
    #: *integer* pixel offsets (the observed data are never resampled); the
    #: sub-pixel galaxy position within each cutout is carried by the model.
    cutout_size: int = 30

    #: Elliptical analysis aperture (semi-axes [px] and position angle [deg])
    #: centred on the *nominal galaxy position of each epoch* in its native
    #: cutout.  The aperture is enlarged by ``mask_margin_px`` so that it
    #: contains the galaxy for every offset allowed by the fit; it is fixed
    #: (parameter-independent) and identical in the template profiling and
    #: the likelihood.
    mask_semi_major: float = 9.0
    mask_semi_minor: float = 5.0
    mask_angle_deg: float = 35.0
    mask_margin_px: float = 1.5

    #: Per-epoch pixels excluded from the fit, in cutout coordinates
    #: ``(epoch, x, y)`` -- e.g. the EP2 bulge pixel for the robustness rerun.
    mask_pixels: tuple[tuple[int, int, int], ...] = ()

    #: Background RMS per epoch [counts/s/px].  ``None`` -> measured from the
    #: data by iterative sigma clipping (fallback: constants.SIGMA_BG_DEFAULT).
    sigma_bg: float | None = None
    gain: float = constants.GAIN_DEFAULT
    exptimes_s: tuple[float, ...] = constants.EPOCH_EXPTIMES_S

    #: Source term of the per-pixel variance: ``"model"`` (default) predicts
    #: the expected count rate from a null-model fit and rebuilds the
    #: variance from it (one reweighting pass); ``"plugin"`` uses the
    #: observed rate (initialization only -- data-dependent weights);
    #: ``"background"`` uses the background RMS alone.
    variance_source: str = "model"

    #: Optional externally supplied galaxy positions, full-frame (x, y) per
    #: epoch (overrides the CSV).  ``None`` -> CSV or estimated.
    galaxy_positions_xy: tuple[tuple[float, float], ...] | None = None
    #: Optional star position (x, y) in full-frame coordinates.  ``None`` ->
    #: flux centroid of the co-aligned disk model.
    star_xy: tuple[float, float] | None = None
    disk_model_shift_xy: tuple[float, float] = constants.DISK_MODEL_SHIFT_XY


@dataclass(slots=True)
class CGConfig:
    """Template solve settings (direct certified path + loose CG search).

    Every log-likelihood that enters a reported statistic is evaluated with
    an *exact dense direct solve* of the template normal equations (the
    cutout has only ~1e3 pixels), built from the same matrix-free shift
    operator as CG -- see ``JointLogLikelihood._solve_direct``.  The
    conjugate-gradient path with the loose ``opt_*`` stopping settings is
    used only inside optimizer steps, where cheap approximate objectives
    and gradients suffice.
    """

    #: Accuracy target of reported profiled log-likelihoods, used as the
    #: verification tolerance when the direct solve is cross-checked
    #: against an independent dense build (``diskphot verify``).
    tol_chi2: float = 1.0e-3
    #: Relative eigenvalue cutoff of the direct pseudo-inverse.  Near
    #: machine precision: every genuinely constrained mode (however weak)
    #: must enter the profile value; only the numerical null space
    #: (masked checkerboard modes at half-pixel shifts) is dropped.
    rcond: float = 1.0e-13
    #: Loose CG: never stop before this many iterations.
    min_iters: int = 20
    #: Loose CG: trailing-window length of the stopping heuristic.
    window: int = 8
    #: Loose CG: stop once the rate-corrected estimate of the remaining
    #: chi-square decrease falls below this.
    opt_tol_chi2: float = 3.0e-2
    #: Loose CG: iteration cap.
    opt_max_iters: int = 120


@dataclass(slots=True)
class ProfileConfig:
    """Profile-likelihood optimization (offsets profiled at fixed tau)."""

    #: Half-width [px] of the box constraint on the per-epoch offsets
    #: (about the nominal, centroid-derived positions).
    offset_halfwidth_px: float = 1.5
    #: Upper bound of the optical-depth coefficient.
    tau_hi: float = 1.0
    #: Numerical floor of the log10-tau optimizer coordinate (the profile
    #: likelihood is flat below ~1e-6; this is *not* a prior).
    tau_opt_floor: float = 1.0e-8
    #: Log-spaced grid on which the profile curve lnL_p(tau) is evaluated
    #: (tau = 0 is always evaluated in addition, via the null fit).
    tau_grid_lo: float = 1.0e-5
    tau_grid_hi: float = 3.0e-2
    tau_grid_n: int = 13
    #: Adam refinement of the profiled offsets.
    steps: int = 400
    lr: float = 2.0e-2
    lr_tau: float = 5.0e-2
    #: Random multi-starts of the joint MLE (in addition to the zero start).
    n_starts: int = 4
    #: Box half-width of the signed relative grey coefficients gamma_i.
    grey_halfwidth: float = 0.5


@dataclass(slots=True)
class CalibConfig:
    """Injection--recovery calibration of the profile-likelihood statistics."""

    #: Injected tau' grid (the profile-curve grid is used when empty).
    taus: tuple[float, ...] = ()
    #: Noise realizations per injected tau'.  A one-sided 95% binomial lower
    #: bound above 0.95 requires >= 59 successes out of 59, so >= 100 is the
    #: sensible minimum for the 95%-power statement.
    n_repeats: int = 100
    #: Null (tau = 0) realizations used to calibrate the detection threshold.
    n_null: int = 500
    #: Detection threshold: the (1 - alpha_det) quantile of the null q0
    #: distribution.  3-sigma one-sided (1.35e-3) needs n_null >~ 2000.
    alpha_det: float = 0.01
    #: One-sided confidence levels of the reported upper limits.
    cl_levels: tuple[float, ...] = (0.841, 0.95)
    #: Detection-probability target of the sensitivity statement.
    power_target: float = 0.95
    #: True-offset dispersion of the simulated epochs [px].
    offset_sigma_px: float = 0.3
    #: Noise stress model: "white" | "correlated" | "scaled" | "outliers".
    noise_model: str = "white"
    #: Gaussian correlation length [px] for noise_model="correlated".
    corr_sigma_px: float = 1.0
    #: Variance misscaling factor for noise_model="scaled".
    variance_scale: float = 1.5
    #: Outlier fraction and amplitude [sigma] for noise_model="outliers".
    outlier_frac: float = 1.0e-3
    outlier_sigma: float = 8.0
    #: Place all true offsets at half-pixel shifts (worst case) instead of
    #: drawing them from a Gaussian.
    half_pixel_offsets: bool = False


@dataclass(slots=True)
class ExploreConfig:
    """Optional UltraNest mapping of the profiled objective.

    The map is a diagnostic of the objective's geometry (offset--tau
    correlations, multi-modality); it is *not* a Bayesian posterior and its
    integral is *not* an evidence, because the template nuisance is profiled
    rather than marginalized.  All quantitative statements come from the
    calibrated profile-likelihood statistics.
    """

    tau_map: TauPrior = TauPrior.LOG_UNIFORM
    #: Lower edge of the log-uniform tau mapping (a plotting-range choice).
    tau_lo: float = 1.0e-6
    min_num_live_points: int = 512
    dlogz: float = 0.5
    min_ess: int = 512
    update_interval_volume_fraction: float = 0.4
    max_num_improvement_loops: int = 4
    ndraw_min: int = 128
    ndraw_max: int = 512
    popsize: int = 512
    nsteps_per_dim: int = 2
    resume: str = "resume"
    #: Internal likelihood chunk (GPU memory guard).
    chunk: int = 256


@dataclass(slots=True)
class FitConfig:
    """Top-level configuration of a single analysis."""

    variant: Variant = Variant.COMMON
    data: DataConfig = field(default_factory=DataConfig)
    cg: CGConfig = field(default_factory=CGConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    calib: CalibConfig = field(default_factory=CalibConfig)
    explore: ExploreConfig = field(default_factory=ExploreConfig)
    log_dir: Path = Path("outputs/runs")
    run_name: str | None = None
    device: str = "auto"  # "auto" | "cuda" | "cuda:0" | "cpu"
    dtype: str = "float64"  # "float64" | "float32"
    seed: int = 0

    @property
    def run_dir(self) -> Path:
        name = self.run_name or self.default_run_name()
        return Path(self.log_dir) / name

    def default_run_name(self) -> str:
        tag = self.variant.value
        if self.data.mask_pixels:
            tag += "_maskpx"
        return tag

    # ---------------------------------------------------------------- JSON --
    def to_json(self, path: Path | None = None) -> str:
        payload = json.dumps(_serialize(self), indent=2, sort_keys=True)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(payload)
        return payload

    @classmethod
    def from_json(cls, source: str | Path) -> FitConfig:
        text = Path(source).read_text() if Path(str(source)).exists() else str(source)
        return _deserialize(cls, json.loads(text))


def _serialize(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _serialize(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj


def _deserialize(cls: type, payload: Any) -> Any:
    if dataclasses.is_dataclass(cls):
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name not in payload:
                continue
            kwargs[f.name] = _coerce(f.type, payload[f.name])
        return cls(**kwargs)
    return payload


_NESTED = {
    "DataConfig": lambda v: _deserialize(DataConfig, v),
    "CGConfig": lambda v: _deserialize(CGConfig, v),
    "ProfileConfig": lambda v: _deserialize(ProfileConfig, v),
    "CalibConfig": lambda v: _deserialize(CalibConfig, v),
    "ExploreConfig": lambda v: _deserialize(ExploreConfig, v),
    "Variant": Variant,
    "TauPrior": TauPrior,
    "Path": Path,
}


def _coerce(annotation: Any, value: Any) -> Any:
    if value is None:
        return None
    text = str(annotation)
    for key, factory in _NESTED.items():
        if key in text:
            return factory(value)
    if isinstance(value, list):
        return tuple(tuple(v) if isinstance(v, list) else v for v in value)
    return value
