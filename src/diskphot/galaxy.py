"""Parametric galaxy image (Sersic bulge + exponential disk).

Used only to generate synthetic epochs for method validation and
injection--recovery; the inference itself is nonparametric (the galaxy
template is profiled out analytically in :mod:`diskphot.model`).  Default
parameters approximate the earlier parametric fit to the "Vermin" galaxy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["GalaxyParams", "galaxy_image", "sersic_bn"]


def sersic_bn(n: Tensor | float) -> Tensor:
    """Ciotti & Bertin (1999) series approximation of b_n (0.36 < n < 8)."""
    n = torch.as_tensor(n, dtype=torch.get_default_dtype())
    return (
        2.0 * n
        - 1.0 / 3.0
        + 4.0 / (405.0 * n)
        + 46.0 / (25515.0 * n**2)
        + 131.0 / (1148175.0 * n**3)
        - 2194697.0 / (30690717750.0 * n**4)
    )


@dataclass(slots=True)
class GalaxyParams:
    """Bulge/disk parameters in cutout pixels and count-rate units."""

    amplitude_bulge: float = 0.094  #: SB at the effective radius [counts/s/px]
    r_eff: float = 4.53  #: bulge effective (half-light) radius [px]
    sersic_n: float = 0.8
    ellip_bulge: float = 0.284
    theta_bulge_deg: float = 15.08
    amplitude_disk: float = 0.129  #: central disk SB [counts/s/px]
    r_disk: float = 5.0  #: exponential scale length [px]
    ellip_disk: float = 0.625
    theta_disk_deg: float = 26.74


def _elliptical_radius(
    x: Tensor, y: Tensor, dx: float, dy: float, theta_deg: float, ellip: float, scale: float
) -> Tensor:
    theta = torch.deg2rad(torch.as_tensor(theta_deg, dtype=x.dtype))
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    u = (x - dx) * cos_t + (y - dy) * sin_t
    v = -(x - dx) * sin_t + (y - dy) * cos_t
    return torch.sqrt((u / scale) ** 2 + (v / ((1.0 - ellip) * scale)) ** 2)


def galaxy_image(
    size: int,
    params: GalaxyParams | None = None,
    dx: float = 0.0,
    dy: float = 0.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Render a ``size x size`` galaxy image centred at ``(dx, dy)`` from centre."""
    p = params or GalaxyParams()
    c = (size - 1) / 2.0
    ys, xs = torch.meshgrid(
        torch.arange(size, device=device, dtype=dtype) - c,
        torch.arange(size, device=device, dtype=dtype) - c,
        indexing="ij",
    )
    z_b = _elliptical_radius(xs, ys, dx, dy, p.theta_bulge_deg, p.ellip_bulge, p.r_eff)
    bn = sersic_bn(torch.tensor(p.sersic_n, dtype=dtype)).to(device)
    bulge = p.amplitude_bulge * torch.exp(-bn * (z_b ** (1.0 / p.sersic_n) - 1.0))

    z_d = _elliptical_radius(xs, ys, dx, dy, p.theta_disk_deg, p.ellip_disk, p.r_disk)
    disk = p.amplitude_disk * torch.exp(-z_d)
    return bulge + disk
