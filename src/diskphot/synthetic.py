"""Synthetic epochs for injection--recovery calibration and roundtrips."""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from .config import CalibConfig, ProfileConfig, SolveConfig, Variant
from .data import EpochData
from .model import JointLogLikelihood
from .transforms import shift


@dataclasses.dataclass
class SyntheticTruth:
    tau: np.ndarray  #: injected tau (scalar in an array, or per-epoch)
    offsets_xy: np.ndarray  #: (R, N, 2) true residual offsets (epoch 0 = 0)


def empirical_template(
    data: EpochData,
    profile: ProfileConfig | None = None,
    solve: SolveConfig | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    """Exactly profiled null-model template at the fitted offsets.

    Returns ``(template, delta_hat)`` with the template in the epoch-0
    frame; injected truths are centred on the data's registration solution.
    """
    from .inference import fit_offsets  # local import to avoid a cycle

    profile = profile or ProfileConfig()
    model = JointLogLikelihood(data, Variant.NULL, solve or SolveConfig())
    res = fit_offsets(model, profile)
    theta = torch.as_tensor(res.theta[0], dtype=data.mask.dtype, device=data.mask.device)
    template = model.maps(theta)["template"]
    return template, res.theta[0].reshape(-1, 2)


def _noise(data: EpochData, n: int, calib: CalibConfig, gen: torch.Generator) -> torch.Tensor:
    """(R, N, H, W) noise draws under the requested stress model (CPU draws)."""
    sig = data.variance.sqrt()
    eps = torch.randn((n, *data.variance.shape), generator=gen, dtype=sig.dtype)
    if calib.noise_model == "correlated":
        from scipy.ndimage import gaussian_filter

        arr = eps.numpy()
        arr = gaussian_filter(arr, sigma=(0, 0, calib.corr_sigma_px, calib.corr_sigma_px))
        arr /= arr.std(axis=(-2, -1), keepdims=True).clip(min=1e-30)
        eps = torch.as_tensor(arr, dtype=sig.dtype)
    noise = sig * eps.to(sig.device)
    if calib.noise_model == "scaled":
        noise = noise * float(np.sqrt(calib.variance_scale))
    elif calib.noise_model == "outliers":
        hit = torch.rand(noise.shape, generator=gen) < calib.outlier_frac
        sgn = torch.sign(torch.randn(noise.shape, generator=gen, dtype=sig.dtype))
        noise = torch.where(hit.to(sig.device),
                            noise + calib.outlier_sigma * sig * sgn.to(sig.device), noise)
    return noise


def simulate_epochs(
    data: EpochData,
    tau: float | np.ndarray,
    template: torch.Tensor,
    n_realizations: int,
    variant: Variant,
    calib: CalibConfig,
    add_noise: bool = True,
    seed: int = 0,
) -> tuple[torch.Tensor, SyntheticTruth]:
    """Draw ``(R, N, H, W)`` disk-subtracted synthetic epochs.

    The template is shifted by nominal + true offsets, attenuated by the
    injected transmission, and (optionally) perturbed with noise from the
    per-pixel variance maps under the configured stress model.
    """
    dev = data.mask.device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = data.n_epochs
    hh, ww = data.mask.shape[-2:]

    if calib.half_pixel_offsets:
        off = torch.full((n_realizations, n, 2), 0.5, dtype=data.mask.dtype)
        off *= torch.sign(torch.randn(off.shape, generator=gen, dtype=off.dtype))
    else:
        off = calib.offset_sigma_px * torch.randn(
            (n_realizations, n, 2), generator=gen, dtype=data.mask.dtype
        )
    off[:, 0] = 0.0
    off = off.to(dev)

    taus = np.atleast_1d(np.asarray(tau, dtype=float))
    model = JointLogLikelihood(data, variant, SolveConfig())
    tpar = torch.as_tensor(
        np.broadcast_to(taus, (1, model.n_tau_params)).copy(),
        dtype=data.mask.dtype, device=dev,
    ) if model.n_tau_params else torch.zeros((1, 0), dtype=data.mask.dtype, device=dev)
    t = model.transmission(tpar)                       # (1, N, ...)

    shifts = (data.nominal_shift_xy.unsqueeze(0) + off).reshape(-1, 2)
    g = template.reshape(1, 1, hh, ww).expand(n_realizations, n, hh, ww)
    clean = t * shift(g.reshape(-1, hh, ww), shifts).reshape(n_realizations, n, hh, ww)
    obs = clean + _noise(data, n_realizations, calib, gen).to(dev) if add_noise else clean
    return obs, SyntheticTruth(tau=taus, offsets_xy=off.cpu().numpy())
