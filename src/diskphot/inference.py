"""Fitting: data preparation, profiled optimization, observed-fit driver.

Optimization is batched block-coordinate descent: the template is re-solved
exactly every ``solve.resolve_every`` steps and held fixed (a rigorous
lower bound of the profile) for the Adam steps in between; best-point
tracking and every reported log-likelihood use a fresh exact solve.  Multi-
start fits and calibration realizations run as one batch on the GPU.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .config import FitConfig, ProfileConfig, SolveConfig, Variant
from .data import EpochData, load_epoch_data, variance_map
from .model import JointLogLikelihood

log = logging.getLogger(__name__)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype: str) -> torch.dtype:
    return {"float64": torch.float64, "float32": torch.float32}[dtype]


def tau_grid(cfg: ProfileConfig) -> np.ndarray:
    return np.geomspace(cfg.tau_grid_lo, cfg.tau_grid_hi, cfg.tau_grid_n)


def prepare_data(cfg: FitConfig) -> EpochData:
    """Load the epochs and (default) install the model-predicted variance.

    The ``"model"`` variance source runs one background-weighted null fit,
    predicts the count rate (disk + registered template) and rebuilds the
    per-pixel variance from it -- weights never depend on the noisy pixel.
    """
    device = resolve_device(cfg.device)
    dt = resolve_dtype(cfg.dtype)
    data = load_epoch_data(cfg.data).to(device, dt)
    if cfg.data.variance_source != "model":
        return data
    bg = torch.zeros_like(data.observations)
    data_bg = data.with_variance(
        variance_map(bg, data.sigma_bg, data.exptimes_s, cfg.data.gain)
    )
    model = JointLogLikelihood(data_bg, Variant.NULL, cfg.solve)
    res = fit_offsets(model, cfg.profile, steps=cfg.profile.steps // 2)
    maps = model.maps(torch.as_tensor(res.theta[0], dtype=dt, device=device))
    predicted = data.disk + maps["model_epoch"]
    return data.with_variance(
        variance_map(predicted, data.sigma_bg, data.exptimes_s, cfg.data.gain)
    )


@dataclasses.dataclass
class FitResult:
    theta: np.ndarray  #: (B, P) best parameters
    logl: np.ndarray  #: (B,) profiled log-likelihood (fresh exact solve)


def _bounds(model: JointLogLikelihood, cfg: ProfileConfig) -> tuple[float, float, float]:
    lo_u = float(np.log10(cfg.tau_opt_floor))
    hi_u = float(np.log10(cfg.tau_hi))
    return cfg.offset_halfwidth_px, lo_u, hi_u


def _assemble(model, offsets: Tensor, tpar: Tensor | None, taus_fixed: Tensor | None) -> Tensor:
    b = offsets.shape[0]
    parts = [offsets.reshape(b, -1)]
    if model.variant is not Variant.NULL:
        if taus_fixed is not None:
            parts.append(taus_fixed.expand(b, -1))
        elif model.variant is Variant.GREY:
            parts.append(tpar)
        else:
            parts.append(10.0 ** tpar)  # log10 coordinate for tau >= 0
    return torch.cat(parts, dim=1)


def _optimize(
    model: JointLogLikelihood,
    cfg: ProfileConfig,
    offsets0: Tensor,
    tpar0: Tensor | None,
    taus_fixed: Tensor | None = None,
    steps: int | None = None,
) -> FitResult:
    """Batched block-descent Adam over (offsets [, tau/gamma])."""
    steps = cfg.steps if steps is None else steps
    solve = model.solve_cfg
    half, lo_u, hi_u = _bounds(model, cfg)
    offsets = offsets0.clone().requires_grad_(True)
    groups = [{"params": [offsets], "lr": cfg.lr}]
    tpar = None
    if tpar0 is not None:
        tpar = tpar0.clone().requires_grad_(True)
        groups.append({"params": [tpar], "lr": cfg.lr_tau})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(steps, 1), eta_min=0.05 * cfg.lr
    )
    b = offsets.shape[0]
    best_logl = torch.full((b,), -torch.inf, dtype=offsets.dtype, device=offsets.device)
    best_theta = _assemble(model, offsets, tpar, taus_fixed).detach()

    def record() -> Tensor:
        nonlocal best_logl, best_theta
        with torch.no_grad():
            theta = _assemble(model, offsets, tpar, taus_fixed)
            tmpl = model.template_for(theta)
            logl = model(theta, template=tmpl)
            better = logl > best_logl
            best_logl = torch.where(better, logl, best_logl)
            best_theta = torch.where(better.view(-1, 1), theta, best_theta)
        return tmpl

    template = record()
    for k in range(steps):
        opt.zero_grad(set_to_none=True)
        theta = _assemble(model, offsets, tpar, taus_fixed)
        loss = -model(theta, template=template).sum()
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            offsets.clamp_(-half, half)
            if tpar is not None:
                if model.variant is Variant.GREY:
                    tpar.clamp_(-cfg.grey_halfwidth, cfg.grey_halfwidth)
                else:
                    tpar.clamp_(lo_u, hi_u)
        if (k + 1) % max(solve.resolve_every, 1) == 0 or k == steps - 1:
            template = record()
    return FitResult(theta=best_theta.cpu().numpy(), logl=best_logl.cpu().numpy())


def _zero_start(model, b: int) -> tuple[Tensor, Tensor | None]:
    p = next(model.buffers())
    offsets = p.new_zeros(b, model.n_offset_params)
    if model.variant is Variant.NULL:
        return offsets, None
    if model.variant is Variant.GREY:
        return offsets, p.new_zeros(b, model.n_tau_params)
    return offsets, p.new_full((b, model.n_tau_params), -3.0)  # tau ~ 1e-3


def fit_offsets(
    model: JointLogLikelihood,
    cfg: ProfileConfig,
    taus_fixed: Tensor | None = None,
    offsets0: Tensor | None = None,
    steps: int | None = None,
) -> FitResult:
    """Profile the offsets (extinction fixed to ``taus_fixed`` or absent)."""
    b = model.y.shape[0]
    off0, _ = _zero_start(model, b)
    if offsets0 is not None:
        off0 = offsets0.to(off0)
    return _optimize(model, cfg, off0, None, taus_fixed=taus_fixed, steps=steps)


def fit_mle(model: JointLogLikelihood, cfg: ProfileConfig, seed: int = 0) -> FitResult:
    """Joint MLE with multi-start (batched when a single realization)."""
    b = model.y.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    half, lo_u, hi_u = _bounds(model, cfg)
    n_starts = max(1, cfg.n_starts)

    def random_start(nb: int) -> tuple[Tensor, Tensor | None]:
        off, tp = _zero_start(model, nb)
        off = off + half * (torch.rand(off.shape, generator=gen) * 2 - 1).to(off) * 0.5
        if tp is not None and model.variant is not Variant.GREY:
            tp = (lo_u + (hi_u - lo_u) * torch.rand(tp.shape, generator=gen)).to(tp)
        return off, tp

    if b == 1:  # batch the starts
        offs, tps = _zero_start(model, 1)
        off_list, tp_list = [offs], [tps]
        for _ in range(n_starts):
            o, t = random_start(1)
            off_list.append(o)
            tp_list.append(t)
        off0 = torch.cat(off_list)
        tp0 = None if tps is None else torch.cat(tp_list)
        res = _optimize(model, cfg, off0, tp0)
        i = int(np.argmax(res.logl))
        return FitResult(theta=res.theta[i : i + 1], logl=res.logl[i : i + 1])

    best: FitResult | None = None
    for s in range(1 + n_starts):
        off0, tp0 = _zero_start(model, b) if s == 0 else random_start(b)
        res = _optimize(model, cfg, off0, tp0)
        if best is None:
            best = res
        else:
            gain = res.logl > best.logl
            best.theta[gain] = res.theta[gain]
            best.logl[gain] = res.logl[gain]
    return best


def profile_curve(
    model: JointLogLikelihood, cfg: ProfileConfig, grid: np.ndarray
) -> np.ndarray:
    """``lnL_p(tau)`` on the grid (offsets profiled; warm-started sweep)."""
    if model.variant is not Variant.COMMON:
        raise ValueError("the profile curve is defined for the COMMON variant")
    b = model.y.shape[0]
    p = next(model.buffers())
    logls = np.empty((b, len(grid)))
    warm: Tensor | None = None
    for j, tau in enumerate(grid):
        taus = p.new_full((1, 1), float(tau))
        res = fit_offsets(
            model, cfg, taus_fixed=taus, offsets0=warm,
            steps=cfg.steps if j == 0 else max(cfg.steps // 2, 50),
        )
        warm = p.new_tensor(res.theta[:, : model.n_offset_params])
        logls[:, j] = res.logl
    return logls


def run_fit(cfg: FitConfig) -> dict:
    """Observed fit of one variant; writes ``config.json`` + ``diskphot_fit.npz``."""
    torch.manual_seed(cfg.seed)
    data = prepare_data(cfg)
    null = JointLogLikelihood(data, Variant.NULL, cfg.solve)
    res_null = fit_offsets(null, cfg.profile)
    payload: dict = {
        "variant": cfg.variant.value,
        "logl_null": float(res_null.logl[0]),
        "theta_null": res_null.theta[0],
        "null_param_names": np.array(null.param_names),
        "n_weighted_pixels": null.n_weighted_pixels,
    }
    if cfg.variant is not Variant.NULL:
        model = JointLogLikelihood(data, cfg.variant, cfg.solve)
        res = fit_mle(model, cfg.profile, seed=cfg.seed)
        logl_mle = max(float(res.logl[0]), payload["logl_null"])
        payload.update(
            param_names=np.array(model.param_names),
            theta_mle=res.theta[0],
            logl_mle=logl_mle,
            q0=max(2.0 * (logl_mle - payload["logl_null"]), 0.0),
        )
        if cfg.variant is Variant.COMMON:
            grid = tau_grid(cfg.profile)
            payload["tau_curve_grid"] = grid
            payload["logl_curve"] = profile_curve(model, cfg.profile, grid)[0]
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(run_dir / "config.json")
    np.savez(run_dir / "diskphot_fit.npz", **payload)
    log.info("fit written to %s", run_dir)
    return payload


# ------------------------------------------------------------------ explore --
def explore(cfg: FitConfig) -> None:
    """Optional UltraNest map of the profiled objective (diagnostic only).

    The integrand is a *profile* likelihood, so the integral is expressly
    not a Bayesian evidence; the run directory receives an explanatory
    ``NOT_AN_EVIDENCE.txt`` and the value is stored as
    ``log_pseudo_evidence``.
    """
    import ultranest  # lazy; optional dependency
    from ultranest.popstepsampler import (
        PopulationSliceSampler,
        generate_region_oriented_direction,
    )

    torch.manual_seed(cfg.seed)
    data = prepare_data(cfg)
    model = JointLogLikelihood(data, cfg.variant, cfg.solve)
    half = cfg.profile.offset_halfwidth_px
    lo, hi = np.log10(cfg.explore.tau_lo), np.log10(cfg.profile.tau_hi)
    ndim = model.n_params
    p = next(model.buffers())

    def transform(u: np.ndarray) -> np.ndarray:
        x = np.array(u, copy=True)
        x[:, : model.n_offset_params] = half * (2 * u[:, : model.n_offset_params] - 1)
        k = model.n_offset_params
        if model.variant is Variant.GREY:
            g = cfg.profile.grey_halfwidth
            x[:, k:] = g * (2 * u[:, k:] - 1)
        elif model.variant is not Variant.NULL:
            x[:, k:] = 10.0 ** (lo + (hi - lo) * u[:, k:])
        return x

    def loglike(x: np.ndarray) -> np.ndarray:
        out = np.empty(len(x))
        chunk = cfg.explore.chunk
        with torch.inference_mode():
            for i0 in range(0, len(x), chunk):
                theta = p.new_tensor(x[i0 : i0 + chunk])
                tmpl = model.template_for(theta)
                out[i0 : i0 + chunk] = model(theta, template=tmpl).cpu().numpy()
        return out

    run_dir = cfg.run_dir / "explore"
    run_dir.mkdir(parents=True, exist_ok=True)
    sampler = ultranest.ReactiveNestedSampler(
        model.param_names, loglike, transform,
        log_dir=str(run_dir), resume=cfg.explore.resume, vectorized=True,
        ndraw_min=cfg.explore.ndraw_min, ndraw_max=cfg.explore.ndraw_max,
    )
    if ndim > 4:
        sampler.stepsampler = PopulationSliceSampler(
            popsize=cfg.explore.popsize,
            nsteps=cfg.explore.nsteps_per_dim * ndim,
            generate_direction=generate_region_oriented_direction,
        )
    result = sampler.run(
        min_num_live_points=cfg.explore.min_num_live_points,
        dlogz=cfg.explore.dlogz, min_ess=cfg.explore.min_ess,
        update_interval_volume_fraction=cfg.explore.update_interval_volume_fraction,
        max_num_improvement_loops=cfg.explore.max_num_improvement_loops,
    )
    np.savez(
        run_dir / "diskphot_map.npz",
        samples=result["samples"],
        param_names=np.array(model.param_names),
        log_pseudo_evidence=result["logz"],
        log_pseudo_evidence_err=result["logzerr"],
    )
    (run_dir / "NOT_AN_EVIDENCE.txt").write_text(
        "This directory maps the PROFILED likelihood (galaxy template profiled\n"
        "out, not marginalized).  Its integral omits the parameter-dependent\n"
        "nuisance-volume term and is NOT a Bayesian evidence; use it only as a\n"
        "geometry/multimodality diagnostic.  All quantitative statements come\n"
        "from the calibrated profile-likelihood statistics (diskphot calibrate\n"
        "/ limits).\n"
    )
