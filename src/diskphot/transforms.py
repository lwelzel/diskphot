r"""Batched sub-pixel image translation with explicitly implemented kernels.

The shift operator is the workhorse of the joint forward model: the galaxy
template (never the data) is translated to each epoch position, and the
same operator's exact adjoint accumulates the normal equations of the
profiled template.  It is therefore implemented explicitly -- as a separable
4-tap convolution with per-sample fractional taps -- rather than through an
opaque resampling backend, so that three properties hold *by construction*
and are unit-tested:

1. **Exact adjoint.**  For a pure translation with a symmetric kernel and
   zero padding, the matrix transpose of :math:`S_{\delta}` is
   :math:`S_{-\delta}` exactly (entry-wise, including the frame boundary):
   :math:`(S_{\delta})_{pq} = k(p - \delta - q)` and
   :math:`(S_{\delta}^{\top})_{pq} = k(q - \delta - p) = k(p + \delta - q) =
   (S_{-\delta})_{pq}`.  The normal matrix
   :math:`H = \sum_i S_i^{\top} W_i S_i` assembled from these operators is
   therefore exactly symmetric positive semi-definite, as required by the
   conjugate-gradient solve in :mod:`diskphot.model`.

2. **Exact normal-matrix diagonal.**  Because the kernel is separable, so is
   its square, and ``squared=True`` evaluates the back-projection with the
   per-axis taps squared.  This yields
   :math:`\mathrm{diag}(H)_q = \sum_i \sum_p k^2(p - s_i - q)\, w_i(p)`
   exactly -- the Jacobi preconditioner of the template solve.  (The
   *plain*-tap back-projection of a weight field, by contrast, is not a
   valid weight: cubic kernels have negative lobes.)

3. **A defined kernel.**  ``"catmull-rom"`` (default) is the cubic
   convolution kernel of Keys (1981) with :math:`a = -1/2`, i.e. the
   Catmull--Rom interpolant: :math:`C^1`, exactly interpolating at integer
   shifts, and near-interpolating in between.  ``"keys"`` (:math:`a = -3/4`,
   the PyTorch ``bicubic`` convention) and ``"bilinear"`` are available for
   comparisons.  Bilinear resampling convolves the image with a tent kernel
   whose variance depends on the fractional shift (:math:`f(1-f)` px^2,
   maximal at half-pixel shifts); at the bulge S/N of this data that
   shift-dependent smoothing is distinguishable from a genuine sub-pixel
   misalignment and would bias fitted offsets toward integer values, which
   is why a near-interpolating cubic is used for the model warp.

Conventions match the previous pipeline: a shift ``(dx, dy)`` moves image
*content* by ``+dx`` along x (columns) and ``+dy`` along y (rows), i.e.
``out[y, x] = in[y - dy, x - dx]``, with zero padding.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["cubic_taps", "linear_taps", "shift2d", "translate", "KERNELS"]


def cubic_taps(f: Tensor, a: float = -0.5) -> Tensor:
    """Cubic-convolution taps (Keys 1981) for fractional shifts ``f`` in [0, 1).

    Returns taps of shape ``(..., 4)`` applying to input samples at offsets
    ``p - m - 2 + j`` (``j = 0..3``) for a total shift ``s = m + f``.  For
    ``f = 0`` the taps are exactly ``(0, 0, 1, 0)`` (integer shifts are exact
    permutations).  ``a = -1/2`` is Catmull--Rom; ``a = -3/4`` matches
    PyTorch's ``grid_sample(mode="bicubic")``.
    """

    def k(x: Tensor) -> Tensor:
        x = x.abs()
        near = (a + 2.0) * x**3 - (a + 3.0) * x**2 + 1.0
        far = a * (x**3 - 5.0 * x**2 + 8.0 * x - 4.0)
        return torch.where(x <= 1.0, near, torch.where(x < 2.0, far, torch.zeros_like(x)))

    return torch.stack([k(2.0 - f), k(1.0 - f), k(f), k(1.0 + f)], dim=-1)


def linear_taps(f: Tensor) -> Tensor:
    """Bilinear (tent) taps in the same 4-tap layout as :func:`cubic_taps`.

    The tap at slot ``j`` applies to input sample ``p - m - 2 + j``; the
    interpolation point ``p - m - f`` lies between samples ``p - m - 1``
    (slot 1, tent weight ``f``) and ``p - m`` (slot 2, weight ``1 - f``).
    """
    z = torch.zeros_like(f)
    return torch.stack([z, f, 1.0 - f, z], dim=-1)


KERNELS = {
    "catmull-rom": lambda f: cubic_taps(f, -0.5),
    "keys": lambda f: cubic_taps(f, -0.75),
    "bilinear": linear_taps,
}


def _shift_axis(x: Tensor, s: Tensor, kernel: str, squared: bool, axis: int) -> Tensor:
    """Shift ``x`` (B, H, W) along ``axis`` by per-sample ``s`` (B,)."""
    b = x.shape[0]
    n = x.shape[axis]
    m = torch.floor(s)
    taps = KERNELS[kernel](s - m)  # (B, 4)
    if squared:
        taps = taps * taps
    m = m.long()
    pos = torch.arange(n, device=x.device).view(1, -1)  # output positions p
    out = torch.zeros_like(x)
    for j in range(4):
        src = pos - (m.view(-1, 1) + 2 - j)  # reads in[p - m - 2 + j]
        inside = (src >= 0) & (src < n)
        src = src.clamp(0, n - 1)
        if axis == -1:
            g = torch.gather(x, -1, src.view(b, 1, n).expand(b, x.shape[-2], n))
            g = g * inside.view(b, 1, n)
        else:
            g = torch.gather(x, -2, src.view(b, n, 1).expand(b, n, x.shape[-1]))
            g = g * inside.view(b, n, 1)
        out = out + taps[:, j].view(-1, 1, 1) * g
    return out


def shift2d(
    images: Tensor,
    shifts_xy: Tensor,
    kernel: str = "catmull-rom",
    squared: bool = False,
) -> Tensor:
    """Translate a batch of images by per-sample sub-pixel shifts.

    Parameters
    ----------
    images
        Tensor of shape ``(B, H, W)``.
    shifts_xy
        Tensor of shape ``(B, 2)`` holding ``(dx, dy)`` in pixels; positive
        values move content toward larger x / y indices.  Zero padding.
    kernel
        ``"catmull-rom"`` (default, model warps), ``"keys"``, or
        ``"bilinear"`` (nonnegative; used for display alignment and for the
        template initial guess).
    squared
        Apply the *squared* per-axis taps.  ``shift2d(w, -s, squared=True)``
        is the exact diagonal contribution of ``S_s^T diag(w) S_s`` and is
        nonnegative for any kernel; it is used to build the Jacobi
        preconditioner of the template normal equations.

    Notes
    -----
    The exact adjoint of ``shift2d(., s, kernel)`` is
    ``shift2d(., -s, kernel)`` (symmetric kernels, zero padding); this
    identity is verified to machine precision in the test suite and by
    ``diskphot verify``.
    """
    if images.ndim != 3:
        raise ValueError(f"expected (B, H, W) images, got shape {tuple(images.shape)}")
    b = images.shape[0]
    if shifts_xy.shape != (b, 2):
        raise ValueError(f"expected ({b}, 2) shifts, got {tuple(shifts_xy.shape)}")
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}; choose from {sorted(KERNELS)}")
    s = shifts_xy.to(images.dtype)
    out = _shift_axis(images, s[:, 0], kernel, squared, axis=-1)
    return _shift_axis(out, s[:, 1], kernel, squared, axis=-2)


def translate(images: Tensor, shifts_xy: Tensor, mode: str = "catmull-rom") -> Tensor:
    """Channelled convenience wrapper: ``(B, C, H, W)`` in and out.

    ``mode`` accepts the kernel names of :func:`shift2d`; ``"bicubic"`` is
    accepted as an alias of ``"catmull-rom"`` for backward compatibility.
    """
    if images.ndim != 4:
        raise ValueError(f"expected (B, C, H, W) images, got shape {tuple(images.shape)}")
    if mode == "bicubic":
        mode = "catmull-rom"
    b, c, h, w = images.shape
    flat = images.reshape(b * c, h, w)
    s = shifts_xy.repeat_interleave(c, dim=0)
    return shift2d(flat, s, kernel=mode).reshape(b, c, h, w)
