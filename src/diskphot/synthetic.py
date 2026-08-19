"""Synthetic epoch sets for validation and calibration.

The generator is a parametric bootstrap of the fitted model: synthetic
epochs are built from the joint forward model with the *empirical* galaxy
template (the exactly profiled template of the null model at the fitted
offsets, so injections carry the true galaxy morphology, bulge sharpness
included), the measured epoch geometry and disk cutouts, randomized true
sub-pixel offsets, and noise drawn from the same per-pixel variance maps
that weight the fit.

Baseline noise is independent Gaussian per pixel -- matched to the fitted
likelihood, which makes these simulations the correct calibration of the
profile-likelihood statistics *under the stated model*.  To probe
robustness against likelihood misspecification the generator also provides
stress variants (``CalibConfig.noise_model``): spatially correlated noise,
misscaled variance, and non-Gaussian outliers, plus a worst-case option
placing every true offset at a half-pixel shift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from .config import CalibConfig, CGConfig, ProfileConfig, Variant
from .data import EpochData
from .model import JointLogLikelihood
from .transforms import shift2d

log = logging.getLogger(__name__)

__all__ = ["SyntheticTruth", "empirical_template", "simulate_epochs"]


@dataclass(slots=True)
class SyntheticTruth:
    tau: np.ndarray  #: injected tau (scalar in an array, or per-epoch)
    offsets_xy: np.ndarray  #: (R, N, 2) true residual offsets (epoch 0 = 0)


def empirical_template(
    data: EpochData,
    profile: ProfileConfig | None = None,
    cg: CGConfig | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    """Exactly profiled null-model template at the fitted offsets.

    Returns ``(template, delta_hat)`` with the template in the epoch-0
    frame and the fitted null-model offsets (used to centre the injected
    truths on the data's registration solution).
    """
    from .inference import fit_offsets  # local import to avoid a cycle

    profile = profile or ProfileConfig()
    model = JointLogLikelihood(data, Variant.NULL, cg)
    res = fit_offsets(model, profile)
    theta = torch.as_tensor(res.theta, dtype=data.mask.dtype, device=data.mask.device)
    template = model.maps(theta)["template"]
    return template, res.theta[0].reshape(-1, 2)


def _correlated_noise(
    shape: tuple[int, ...],
    sigma_px: float,
    gen: torch.Generator,
    like: torch.Tensor,
) -> torch.Tensor:
    """Unit-variance Gaussian noise with a Gaussian correlation kernel."""
    white = torch.randn(shape, generator=gen).to(like)
    radius = max(1, int(3.0 * sigma_px))
    x = torch.arange(-radius, radius + 1, dtype=like.dtype, device=like.device)
    k = torch.exp(-0.5 * (x / sigma_px) ** 2)
    k = k / torch.sqrt((k * k).sum())  # preserve per-pixel variance
    flat = white.reshape(-1, 1, *shape[-2:])
    flat = torch.nn.functional.conv2d(flat, k.view(1, 1, 1, -1), padding=(0, radius))
    flat = torch.nn.functional.conv2d(flat, k.view(1, 1, -1, 1), padding=(radius, 0))
    return flat.reshape(shape)


def simulate_epochs(
    data: EpochData,
    tau: float | np.ndarray,
    template: torch.Tensor,
    n_realizations: int = 1,
    variant: Variant = Variant.COMMON,
    calib: CalibConfig | None = None,
    add_noise: bool = True,
    seed: int = 0,
) -> tuple[torch.Tensor, SyntheticTruth]:
    """Forward-model synthetic epoch stacks with known truths.

    Returns ``(observations, truth)`` with observations of shape
    ``(R, N, H, W)``: ``disk + T(tau) * shift(template) + noise`` on the
    native grids, using the same disk cutouts and per-pixel variance maps
    as the real data.  The true total shift of epoch ``i`` is the nominal
    geometry plus the drawn residual offset (epoch 0 pinned at zero).
    """
    calib = calib or CalibConfig()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = data.n_epochs
    r = int(n_realizations)
    g = template.to(data.observations)
    h, w = g.shape[-2:]

    offsets = torch.zeros(r, n, 2)
    if calib.half_pixel_offsets:
        signs = torch.where(torch.rand((r, n - 1, 2), generator=gen) < 0.5, -0.5, 0.5)
        offsets[:, 1:] = signs
    else:
        offsets[:, 1:] = calib.offset_sigma_px * torch.randn((r, n - 1, 2), generator=gen)
    offsets = offsets.to(g)
    shifts = data.nominal_shift_xy.unsqueeze(0) + offsets

    tau_arr = np.atleast_1d(np.asarray(tau, dtype=np.float64))
    if variant in (Variant.PER_EPOCH, Variant.GREY) and tau_arr.size == 1:
        tau_arr = np.full(n, tau_arr[0])
    tau_t = torch.as_tensor(tau_arr, dtype=g.dtype, device=g.device)
    if variant is Variant.COMMON:
        exponent = tau_t.view(1, 1, 1, 1) * data.disk_hat.unsqueeze(0)
    elif variant is Variant.PER_EPOCH:
        exponent = tau_t.view(1, n, 1, 1) * data.disk_hat.unsqueeze(0)
    elif variant is Variant.GREY:
        exponent = tau_t.view(1, n, 1, 1).expand(1, n, h, w)
    else:
        exponent = torch.zeros(1, n, h, w, dtype=g.dtype, device=g.device)
    transmission = torch.exp(-exponent)

    shifted = shift2d(
        g.expand(r * n, h, w).contiguous(), shifts.reshape(r * n, 2)
    ).view(r, n, h, w)
    obs = data.disk.unsqueeze(0) + transmission * shifted

    if add_noise:
        var = data.variance.unsqueeze(0)
        if calib.noise_model == "scaled":
            var = var * calib.variance_scale
        if calib.noise_model == "correlated":
            unit = _correlated_noise((r, n, h, w), calib.corr_sigma_px, gen, obs)
        else:
            unit = torch.randn((r, n, h, w), generator=gen).to(obs)
        noise = unit * torch.sqrt(var)
        if calib.noise_model == "outliers":
            hit = torch.rand((r, n, h, w), generator=gen).to(obs) < calib.outlier_frac
            sign = torch.where(torch.rand((r, n, h, w), generator=gen) < 0.5, -1.0, 1.0).to(obs)
            noise = noise + hit * sign * calib.outlier_sigma * torch.sqrt(var)
        obs = obs + noise

    truth = SyntheticTruth(tau=tau_arr, offsets_xy=offsets.cpu().numpy())
    return obs.contiguous(), truth
