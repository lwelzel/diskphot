"""Sub-pixel translation, delegated to torch's native grid sampler.

The interpolation kernel is torch's ``bicubic`` grid-sampler mode -- the
separable Keys (1981) cubic-convolution kernel with a = -3/4 -- evaluated by
one fused CUDA/CPU kernel per call instead of hand-rolled per-tap gathers.
For a pure translation with this symmetric kernel and zero padding the exact
matrix transpose of the shift operator is a negated shift,
``S_s^T = S_{-s}`` (entry-wise: ``(S_s)_{pq} = k(p - s - q)`` and ``k`` is
even), so the adjoint used by the normal equations is simply
``shift(x, -s)``.  Autograd through the sampler supplies exact derivatives
with respect to the shifts (used by the envelope gradient of the profiled
likelihood).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _grid(shifts_xy: Tensor, h: int, w: int) -> Tensor:
    """Normalized sampling grid realizing ``out[p] = in[p - s]``."""
    dt, dev = shifts_xy.dtype, shifts_xy.device
    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=dt, device=dev),
        torch.arange(w, dtype=dt, device=dev),
        indexing="ij",
    )
    gx = xs - shifts_xy[:, 0].view(-1, 1, 1)
    gy = ys - shifts_xy[:, 1].view(-1, 1, 1)
    return torch.stack(
        (2.0 * gx / max(w - 1, 1) - 1.0, 2.0 * gy / max(h - 1, 1) - 1.0), dim=-1
    )


def shift(images: Tensor, shifts_xy: Tensor) -> Tensor:
    """Translate each image by its (x, y) shift; content moves by ``+s``.

    ``images (B, H, W)``, ``shifts_xy (B, 2)`` -> ``(B, H, W)`` with zero
    padding at the exposed border.  The adjoint is ``shift(x, -s)``.
    """
    b, h, w = images.shape
    return F.grid_sample(
        images.unsqueeze(1),
        _grid(shifts_xy.to(images.dtype), h, w),
        mode="bicubic",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(1)


def translate(images: Tensor, shifts_xy: Tensor) -> Tensor:
    """Channelled variant: ``images (B, C, H, W)``, ``shifts_xy (B, 2)``."""
    b, c, h, w = images.shape
    out = shift(images.reshape(b * c, h, w),
                shifts_xy.repeat_interleave(c, dim=0))
    return out.reshape(b, c, h, w)
