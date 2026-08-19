"""Profile-likelihood inference.

The science statements of the pipeline are frequentist and are all derived
from the exactly profiled log-likelihood of :mod:`diskphot.model`:

* ``fit_mle`` -- multi-start gradient maximization over the offsets and the
  extinction parameters (box constraints; the optical depth is optimized in
  ``log10 tau`` down to a numerical floor far below the sensitivity, with
  ``tau = 0`` covered by the null fit -- there is no prior and hence no
  prior floor in any reported number);
* ``profile_curve`` -- the one-dimensional profile ``lnL_p(tau')`` with the
  offsets re-optimized at every grid point (warm-started);
* ``run_fit`` -- the observed statistics: the null fit, the joint MLE, the
  detection statistic ``q0 = 2 [lnL_p(mle) - lnL_p(tau'=0)]``, and the
  profile curve.  Their sampling distributions (detection threshold,
  confidence limits, coverage) are calibrated by simulation in
  :mod:`diskphot.calibrate` -- see there for the statistical definitions.

``explore`` runs an optional UltraNest mapping of the profiled objective.
The map is a diagnostic of the objective's geometry (offset--tau
correlations, multimodality) and a global cross-check of the optimizer; it
is *not* a Bayesian posterior and its integral is *not* an evidence,
because the template nuisance is profiled rather than marginalized under a
proper prior.  No quantitative result is derived from it.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import FitConfig, ProfileConfig, TauPrior, Variant
from .data import EpochData, load_epoch_data, variance_map
from .model import JointLogLikelihood

log = logging.getLogger(__name__)

__all__ = [
    "FitResult",
    "TorchLikelihood",
    "explore",
    "fit_mle",
    "fit_offsets",
    "prepare_data",
    "profile_curve",
    "resolve_device",
    "run_fit",
    "tau_grid",
]


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype: str) -> torch.dtype:
    return torch.float64 if dtype == "float64" else torch.float32


def tau_grid(cfg: ProfileConfig) -> np.ndarray:
    """Log-spaced tau' grid of the profile curve (tau = 0 handled separately)."""
    return np.geomspace(cfg.tau_grid_lo, cfg.tau_grid_hi, cfg.tau_grid_n)


# --------------------------------------------------------------------------- #
# Data preparation (model-predicted variance pass)
# --------------------------------------------------------------------------- #
def prepare_data(
    cfg: FitConfig,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> EpochData:
    """Load the epochs and, by default, refine the noise model once.

    With ``DataConfig.variance_source == "model"`` the per-pixel variance is
    rebuilt from a *model-predicted* expected count rate (disk model plus the
    null-model galaxy at the fitted offsets) instead of the noisy observed
    pixel values, so that individual noise fluctuations do not set their own
    likelihood weights.
    """
    device = device or resolve_device(cfg.device)
    dtype = dtype or resolve_dtype(cfg.dtype)
    data = load_epoch_data(cfg.data).to(device, dtype)
    if cfg.data.variance_source != "model":
        return data

    null = JointLogLikelihood(data, Variant.NULL, cfg.cg)
    res = fit_offsets(null, cfg.profile)
    theta = torch.as_tensor(res.theta, dtype=dtype, device=device)
    maps = null.maps(theta)
    predicted = data.disk + maps["model_epoch"]
    refined = variance_map(predicted, data.sigma_bg, data.exptimes_s, cfg.data.gain)
    log.info(
        "model-predicted variance pass: median ratio to plug-in = %.3f",
        float(torch.median(refined / data.variance)),
    )
    return data.with_variance(refined)


# --------------------------------------------------------------------------- #
# Gradient optimization of the profiled objective
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class FitResult:
    """Optimized parameters and profiled log-likelihoods, one row per batch."""

    theta: np.ndarray  #: (B, P)
    logl: np.ndarray  #: (B,)
    param_names: list[str]


def _clamp_(offsets: torch.Tensor, u_tau: torch.Tensor | None, model, cfg: ProfileConfig) -> None:
    offsets.clamp_(-cfg.offset_halfwidth_px, cfg.offset_halfwidth_px)
    if u_tau is None:
        return
    if model.variant is Variant.GREY:
        u_tau.clamp_(-cfg.grey_halfwidth, cfg.grey_halfwidth)
    else:
        u_tau.clamp_(np.log10(cfg.tau_opt_floor), np.log10(cfg.tau_hi))


def _assemble(offsets: torch.Tensor, u_tau: torch.Tensor | None, taus_fixed, model) -> torch.Tensor:
    parts = [offsets]
    if u_tau is not None:
        parts.append(u_tau if model.variant is Variant.GREY else 10.0**u_tau)
    elif taus_fixed is not None:
        parts.append(taus_fixed)
    return torch.cat(parts, dim=1)


def _optimize(
    model: JointLogLikelihood,
    cfg: ProfileConfig,
    offsets0: torch.Tensor,
    u_tau0: torch.Tensor | None,
    taus_fixed: torch.Tensor | None = None,
    steps: int | None = None,
) -> FitResult:
    """Batched Adam ascent with box constraints and elementwise best-tracking.

    The template solve inside the likelihood is detached (envelope theorem),
    so gradients with respect to the offsets and extinction parameters are
    those of the exact profiled objective up to the CG tolerance.
    """
    steps = steps or cfg.steps
    offsets = offsets0.clone().requires_grad_(True)
    groups = [{"params": [offsets], "lr": cfg.lr}]
    u_tau = None
    if u_tau0 is not None:
        u_tau = u_tau0.clone().requires_grad_(True)
        groups.append({"params": [u_tau], "lr": cfg.lr_tau})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.05 * cfg.lr)

    best_logl = torch.full((offsets.shape[0],), -np.inf, dtype=offsets.dtype, device=offsets.device)
    best_theta = _assemble(offsets, u_tau, taus_fixed, model).detach().clone()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        theta = _assemble(offsets, u_tau, taus_fixed, model)
        logl = model(theta)
        (-logl.sum()).backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            _clamp_(offsets, u_tau, model, cfg)
            theta = _assemble(offsets, u_tau, taus_fixed, model)
            logl = model(theta)
            better = logl > best_logl
            best_logl = torch.where(better, logl, best_logl)
            best_theta = torch.where(better.view(-1, 1), theta, best_theta)
    with torch.no_grad():
        # Certified evaluation: every reported log-likelihood is the
        # explicit-residual value at the tight CG tolerance.
        best_logl = model(best_theta, tight=True)
    return FitResult(
        theta=best_theta.cpu().numpy(),
        logl=best_logl.cpu().numpy(),
        param_names=model.param_names,
    )


def fit_offsets(
    model: JointLogLikelihood,
    cfg: ProfileConfig,
    taus_fixed: torch.Tensor | None = None,
    offsets0: torch.Tensor | None = None,
    steps: int | None = None,
) -> FitResult:
    """Profile the offsets at fixed extinction parameters (or none).

    ``taus_fixed`` has shape ``(B, n_tau)`` (ignored for the null model);
    ``B`` must equal the model's realization count when data are batched.
    """
    b = taus_fixed.shape[0] if taus_fixed is not None else max(model.n_realizations, 1)
    ref = model.mask
    if offsets0 is None:
        offsets0 = ref.new_zeros((b, model.n_offset_params))
    return _optimize(model, cfg, offsets0, None, taus_fixed=taus_fixed, steps=steps)


def fit_mle(
    model: JointLogLikelihood,
    cfg: ProfileConfig,
    seed: int = 0,
    tau_init: float = 1.0e-3,
) -> FitResult:
    """Multi-start joint MLE over offsets and extinction parameters.

    For single-realization data the starts run as one batch; for batched
    (injection) data the starts run sequentially and the elementwise best is
    kept.  The optical depth of the disk-scaled variants is optimized in
    ``log10 tau`` (numerical floor ``cfg.tau_opt_floor``; the profile is
    flat far above the floor, and ``tau = 0`` is covered by the null fit).
    """
    ref = model.mask
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n_off, n_tau = model.n_offset_params, model.n_tau_params
    r = max(model.n_realizations, 1)

    def start(scale: float) -> tuple[torch.Tensor, torch.Tensor | None]:
        off = scale * cfg.offset_halfwidth_px * (
            torch.rand((r, n_off), generator=gen) * 2.0 - 1.0
        )
        if n_tau == 0:
            return off.to(ref), None
        if model.variant is Variant.GREY:
            u = scale * cfg.grey_halfwidth * (torch.rand((r, n_tau), generator=gen) * 2.0 - 1.0)
        else:
            lo, hi = np.log10(cfg.tau_opt_floor), np.log10(cfg.tau_hi)
            u = torch.full((r, n_tau), np.log10(tau_init))
            if scale > 0:
                u = lo + (hi - lo) * torch.rand((r, n_tau), generator=gen)
        return off.to(ref), u.to(ref)

    starts = [start(0.0)] + [start(0.7) for _ in range(cfg.n_starts)]
    if model.n_realizations == 1:
        off0 = torch.cat([s[0] for s in starts])
        u0 = torch.cat([s[1] for s in starts]) if n_tau else None
        res = _optimize(model, cfg, off0, u0)
        k = int(np.argmax(res.logl))
        return FitResult(res.theta[k : k + 1], res.logl[k : k + 1], res.param_names)

    best: FitResult | None = None
    for off0, u0 in starts:
        res = _optimize(model, cfg, off0, u0)
        if best is None:
            best = res
        else:
            better = res.logl > best.logl
            best.theta[better] = res.theta[better]
            best.logl[better] = res.logl[better]
    assert best is not None
    return best


def profile_curve(
    model: JointLogLikelihood,
    cfg: ProfileConfig,
    taus: np.ndarray,
    offsets0: torch.Tensor | None = None,
) -> np.ndarray:
    """Profile log-likelihood ``lnL_p(tau')`` with offsets re-optimized.

    Only defined for :attr:`Variant.COMMON`.  Returns an array of shape
    ``(len(taus), R)``.  Grid points are warm-started from their neighbour.
    """
    if model.variant is not Variant.COMMON:
        raise ValueError("the profile curve is defined for the common-tau variant")
    ref = model.mask
    r = max(model.n_realizations, 1)
    out = np.empty((len(taus), r))
    off = offsets0.clone() if offsets0 is not None else ref.new_zeros((r, model.n_offset_params))
    for j, tau in enumerate(taus):
        fixed = ref.new_full((r, 1), float(tau))
        res = fit_offsets(model, cfg, taus_fixed=fixed, offsets0=off, steps=cfg.steps if j == 0 else max(cfg.steps // 2, 50))
        out[j] = res.logl
        off = torch.as_tensor(res.theta[:, : model.n_offset_params], dtype=ref.dtype, device=ref.device)
    return out


# --------------------------------------------------------------------------- #
# Observed-data fit
# --------------------------------------------------------------------------- #
def run_fit(
    cfg: FitConfig,
    data: EpochData | None = None,
    run_dir: Path | None = None,
) -> dict:
    """Observed profile-likelihood statistics of one variant.

    Fits the null model (offsets only), the requested variant's joint MLE,
    and -- for the common-tau variant -- the profile curve, then persists
    everything needed by :mod:`diskphot.results` and
    :mod:`diskphot.calibrate`:

    ``config.json``
        The exact :class:`FitConfig` of the run.
    ``diskphot_fit.npz``
        Null and MLE profiled log-likelihoods, the detection statistic
        ``q0 = 2 [lnL_p(mle) - lnL_p(0)]``, the tau grid and profile curve.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype)
    log.info("fitting variant=%s on %s (%s)", cfg.variant.value, device, cfg.dtype)

    if data is None:
        data = prepare_data(cfg, device, dtype)
    else:
        data = data.to(device, dtype)

    null = JointLogLikelihood(data, Variant.NULL, cfg.cg)
    res_null = fit_offsets(null, cfg.profile)
    payload: dict = {
        "variant": cfg.variant.value,
        "logl_null": float(res_null.logl[0]),
        "theta_null": res_null.theta[0],
        "null_param_names": res_null.param_names,
        "n_weighted_pixels": null.n_weighted_pixels,
    }

    if cfg.variant is not Variant.NULL:
        model = JointLogLikelihood(data, cfg.variant, cfg.cg)
        res = fit_mle(model, cfg.profile, seed=cfg.seed)
        q0 = max(2.0 * (float(res.logl[0]) - payload["logl_null"]), 0.0)
        payload |= {
            "param_names": res.param_names,
            "theta_mle": res.theta[0],
            "logl_mle": float(res.logl[0]),
            "q0": q0,
        }
        if cfg.variant is Variant.COMMON:
            grid = tau_grid(cfg.profile)
            off0 = torch.as_tensor(
                res.theta[:, : model.n_offset_params], dtype=data.mask.dtype, device=data.mask.device
            )
            curve = profile_curve(model, cfg.profile, grid, offsets0=off0)[:, 0]
            payload |= {"tau_curve_grid": grid, "logl_curve": curve}

    run_dir = Path(run_dir if run_dir is not None else cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(run_dir / "config.json")
    np.savez_compressed(
        run_dir / "diskphot_fit.npz",
        **{k: np.asarray(v) for k, v in payload.items()},
    )
    if "q0" in payload:
        log.info(
            "lnL_p(null) = %.3f  lnL_p(mle) = %.3f  q0 = %.3f",
            payload["logl_null"],
            payload["logl_mle"],
            payload["q0"],
        )
    return payload


# --------------------------------------------------------------------------- #
# numpy <-> torch bridge (chunked)
# --------------------------------------------------------------------------- #
class TorchLikelihood:
    """Adapter exposing the torch likelihood to vectorized samplers.

    Incoming ``(B, P)`` float64 arrays are evaluated in chunks of at most
    ``chunk`` rows to bound GPU memory (the sampler's proposal batch size is
    thereby decoupled from the model batch size).
    """

    def __init__(
        self,
        model: JointLogLikelihood,
        device: torch.device,
        dtype: torch.dtype,
        chunk: int = 256,
    ):
        self.model = model.to(device=device, dtype=dtype).eval()
        self.device = device
        self.dtype = dtype
        self.chunk = max(int(chunk), 1)

    @torch.inference_mode()
    def __call__(self, theta: np.ndarray) -> np.ndarray:
        theta = np.ascontiguousarray(theta, dtype=np.float64)
        out = np.empty(theta.shape[0])
        for start in range(0, theta.shape[0], self.chunk):
            t = torch.from_numpy(theta[start : start + self.chunk]).to(
                device=self.device, dtype=self.dtype
            )
            out[start : start + self.chunk] = self.model(t).double().cpu().numpy()
        return out


# --------------------------------------------------------------------------- #
# Optional objective map (UltraNest)
# --------------------------------------------------------------------------- #
def explore(
    cfg: FitConfig,
    data: EpochData | None = None,
    run_dir: Path | None = None,
) -> dict:
    """Map the profiled objective with UltraNest (diagnostic only).

    The sampler treats ``exp(lnL_p)`` times the parameter mapping as an
    unnormalized density; the resulting sample cloud visualizes offset--tau
    degeneracies and cross-checks the optimizer's global maximum.  The
    integral is stored as ``log_pseudo_evidence`` and must not be
    interpreted as a Bayesian evidence: the template nuisance is profiled,
    not marginalized, so no nuisance-volume term is present.
    """
    import ultranest
    from ultranest.popstepsampler import (
        PopulationSliceSampler,
        generate_region_oriented_direction,
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype)
    if data is None:
        data = prepare_data(cfg, device, dtype)
    else:
        data = data.to(device, dtype)

    model = JointLogLikelihood(data, cfg.variant, cfg.cg)
    n_off, n_tau = model.n_offset_params, model.n_tau_params
    half = float(cfg.profile.offset_halfwidth_px)
    exp = cfg.explore
    lo, hi = float(exp.tau_lo), float(cfg.profile.tau_hi)
    log_lo, log_hi = np.log10(lo), np.log10(hi)
    log_uniform = exp.tau_map is TauPrior.LOG_UNIFORM
    grey = model.variant is Variant.GREY
    ghw = float(cfg.profile.grey_halfwidth)

    def transform(cube: np.ndarray) -> np.ndarray:
        params = np.empty_like(cube)
        params[:, :n_off] = cube[:, :n_off] * (2.0 * half) - half
        if n_tau:
            u = cube[:, n_off:]
            if grey:
                params[:, n_off:] = u * (2.0 * ghw) - ghw
            elif log_uniform:
                params[:, n_off:] = 10.0 ** (u * (log_hi - log_lo) + log_lo)
            else:
                params[:, n_off:] = u * hi
        return params

    likelihood = TorchLikelihood(model, device, dtype, chunk=exp.chunk)
    run_dir = Path(run_dir if run_dir is not None else cfg.run_dir / "explore")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(run_dir / "config.json")

    sampler = ultranest.ReactiveNestedSampler(
        model.param_names,
        likelihood,
        transform,
        vectorized=True,
        log_dir=str(run_dir),
        resume=exp.resume,
        ndraw_min=exp.ndraw_min,
        ndraw_max=exp.ndraw_max,
    )
    ndim = model.n_params
    if ndim > 4:
        sampler.stepsampler = PopulationSliceSampler(
            popsize=exp.popsize,
            nsteps=exp.nsteps_per_dim * ndim,
            generate_direction=generate_region_oriented_direction,
        )
    result = sampler.run(
        min_num_live_points=exp.min_num_live_points,
        dlogz=exp.dlogz,
        min_ess=exp.min_ess,
        update_interval_volume_fraction=exp.update_interval_volume_fraction,
        max_num_improvement_loops=exp.max_num_improvement_loops,
    )
    sampler.print_results()

    scales = ["linear"] * n_off + (
        ["linear"] * n_tau if grey or not log_uniform else ["log"] * n_tau
    )
    weighted = result["weighted_samples"]
    np.savez_compressed(
        run_dir / "diskphot_map.npz",
        param_names=np.asarray(model.param_names),
        axes_scales=np.asarray(scales),
        samples=np.asarray(result["samples"]),
        weighted_points=np.asarray(weighted["points"]),
        weighted_weights=np.asarray(weighted["weights"]),
        weighted_logl=np.asarray(weighted["logl"]),
        log_pseudo_evidence=float(result["logz"]),
        log_pseudo_evidence_err=float(result["logzerr"]),
        variant=cfg.variant.value,
    )
    (run_dir / "NOT_AN_EVIDENCE.txt").write_text(
        "This directory contains a nested-sampling *map of the profiled "
        "objective*.\nThe template nuisance is profiled, not marginalized, "
        "so the integral\n(log_pseudo_evidence) is not a Bayesian evidence "
        "and differences of it\nbetween variants are not Bayes factors.  "
        "Quantitative statements come\nfrom the calibrated profile-"
        "likelihood statistics (diskphot calibrate / limits).\n"
    )
    result["param_names"] = model.param_names
    return result
