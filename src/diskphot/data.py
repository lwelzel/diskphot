"""Data preparation on the native epoch pixel grids.

Loads the eight NUP-reduced epochs and the scattered-light disk model and
builds the :class:`EpochData` container consumed by the likelihood.  The
observed images are **never resampled**: each epoch is cropped at *integer*
pixel offsets around its galaxy position, so the per-pixel noise stays
uncorrelated by construction and validity masks stay binary.  The residual
sub-pixel geometry (the galaxy's fractional position in each native cutout)
is carried by the *model*: the loader stores the nominal per-epoch model
shift, and the fit refines it with the free offsets.

Pipeline
--------
1. Read the epoch count-rate images; co-align the smooth disk *model* with
   the star by the known ``(-540.45, -541.08)`` px shift (interpolating the
   noiseless model is unproblematic; the data are untouched).
2. Take approximate galaxy centres from ``galaxy_positions.csv`` (or
   estimate them) and refine each with a flux-weighted centroid of the
   disk-subtracted image.
3. Crop each epoch (and its disk-model view) at integer offsets so that the
   galaxy lies within half a pixel of the cutout centre; record the
   nominal sub-pixel model shifts ``s_i = q_i - q_0``.
4. Build the fixed per-epoch analysis weights: binary validity x an
   elliptical aperture centred on the nominal galaxy position, enlarged by
   the offset box half-width so it contains the galaxy for every allowed
   offset.  The same weights are used for the template profiling and the
   likelihood; they do not depend on fitted parameters.
5. Build per-pixel variance maps ``sigma^2 = sigma_bg^2 + max(r, 0)/(g t)``.
   The source rate ``r`` defaults to a *model-predicted* rate (one
   reweighting pass in :func:`diskphot.inference.prepare_data`) so that
   individual noise fluctuations do not set their own weights.
6. Normalize the disk model to its ring surface-brightness peak, so the
   fitted tau' is the physical optical depth at the SB peak.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

from . import constants
from .config import DataConfig
from .transforms import translate

log = logging.getLogger(__name__)

__all__ = [
    "EpochData",
    "estimate_star_position",
    "load_epoch_data",
    "load_full_frames",
    "make_ellipse_mask",
    "read_positions_csv",
    "ring_profile",
    "variance_map",
]


# --------------------------------------------------------------------------- #
# Container
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class EpochData:
    """All tensors and metadata needed by the likelihood, in one place.

    Image tensors have shape ``(N_ep, H, W)`` with ``H = W = cutout_size``;
    ``[y, x]`` indexing, origin lower-left.  Every epoch lives on its own
    native pixel grid (integer crop of the full frame); the sub-pixel
    geometry is carried by ``nominal_shift_xy``.
    """

    observations: torch.Tensor  #: native count-rate cutouts (integer crops)
    disk: torch.Tensor  #: disk-model cutouts (instrument units)
    disk_hat: torch.Tensor  #: disk cutouts / ring SB peak (dimensionless)
    variance: torch.Tensor  #: per-pixel variance of ``observations``
    valid: torch.Tensor  #: per-epoch binary validity (1 = usable pixel)
    mask: torch.Tensor  #: per-epoch elliptical analysis aperture (N, H, W)
    nominal_shift_xy: torch.Tensor  #: (N, 2) model shifts s_i = q_i - q_0
    galaxy_xy_cutout: torch.Tensor  #: (N, 2) galaxy positions q_i, cutout frame
    galaxy_xy_full: np.ndarray  #: (N, 2) galaxy positions, full frame
    crop_origin_xy: np.ndarray  #: (N, 2) integer crop origins, full frame
    star_xy_full: tuple[float, float]
    disk_peak_sb: float  #: ring SB peak [counts/s/px]
    disk_peak_radius_px: float
    sigma_bg: np.ndarray  #: per-epoch background RMS actually used
    exptimes_s: np.ndarray
    dates: tuple[str, ...] = constants.EPOCH_DATES
    config: DataConfig = field(default_factory=DataConfig)

    @property
    def n_epochs(self) -> int:
        return int(self.observations.shape[0])

    @property
    def galaxy_only(self) -> torch.Tensor:
        """Disk-subtracted cutouts (galaxy + noise)."""
        return self.observations - self.disk

    @property
    def weights(self) -> torch.Tensor:
        """Fixed per-pixel likelihood weights ``valid * mask / variance``.

        The same field enters the template normal equations and the residual
        chi-square; it does not depend on fitted parameters.
        """
        return self.valid * self.mask / self.variance

    def to(self, device: torch.device | str, dtype: torch.dtype = torch.float64) -> EpochData:
        def _cast(t: torch.Tensor) -> torch.Tensor:
            return t.to(device=device, dtype=dtype).contiguous()

        return replace(
            self,
            observations=_cast(self.observations),
            disk=_cast(self.disk),
            disk_hat=_cast(self.disk_hat),
            variance=_cast(self.variance),
            valid=_cast(self.valid),
            mask=_cast(self.mask),
            nominal_shift_xy=_cast(self.nominal_shift_xy),
            galaxy_xy_cutout=_cast(self.galaxy_xy_cutout),
        )

    def with_variance(self, variance: torch.Tensor) -> EpochData:
        """Copy of the container with a replacement variance map."""
        return replace(self, variance=variance.to(self.variance).contiguous())

    def with_observations(self, observations: torch.Tensor) -> EpochData:
        """Copy with replacement observations (synthetic data injection)."""
        return replace(self, observations=observations.to(self.observations).contiguous())

    # ------------------------------------------------------------- physics --
    def disk_hat_at_galaxy(self, radius_px: float = 2.0) -> np.ndarray:
        """Mean normalized disk SB within ``radius_px`` of each galaxy position.

        The per-epoch scale factor converting the fitted tau' into the
        physical optical depth along each sightline, ``tau_i = tau' * dhat_i``.
        """
        h, w = self.disk_hat.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=self.disk_hat.dtype, device=self.disk_hat.device),
            torch.arange(w, dtype=self.disk_hat.dtype, device=self.disk_hat.device),
            indexing="ij",
        )
        out = []
        for i in range(self.n_epochs):
            gx, gy = self.galaxy_xy_cutout[i]
            sel = (xx - gx) ** 2 + (yy - gy) ** 2 <= radius_px**2
            out.append(float(self.disk_hat[i][sel].mean()))
        return np.asarray(out)

    def galaxy_radii_arcsec(self) -> np.ndarray:
        """Stellocentric separation of the galaxy per epoch [arcsec]."""
        star = np.asarray(self.star_xy_full)
        d_px = np.hypot(*(self.galaxy_xy_full - star).T)
        return np.asarray(constants.pixels_to_arcsec(d_px))


# --------------------------------------------------------------------------- #
# Masks & noise
# --------------------------------------------------------------------------- #
def make_ellipse_mask(
    semi_major: float,
    semi_minor: float,
    angle_deg: float,
    size: int,
    center_xy: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Binary elliptical mask (float 0/1) in a ``size x size`` frame."""
    theta = math.radians(angle_deg)
    cx, cy = center_xy if center_xy is not None else ((size - 1) / 2.0,) * 2
    ys, xs = torch.meshgrid(
        torch.arange(size, dtype=torch.float64) - cy,
        torch.arange(size, dtype=torch.float64) - cx,
        indexing="ij",
    )
    u = xs * math.cos(theta) + ys * math.sin(theta)
    v = -xs * math.sin(theta) + ys * math.cos(theta)
    return ((u / semi_major) ** 2 + (v / semi_minor) ** 2 <= 1.0).to(torch.float64)


def variance_map(
    source_rate: torch.Tensor,
    sigma_bg: np.ndarray,
    exptimes_s: np.ndarray,
    gain: float,
) -> torch.Tensor:
    """Per-pixel variance ``sigma_bg^2 + max(source_rate, 0) / (g t)``.

    ``source_rate`` should be an *expected* count rate (model prediction or
    background-only zeros); using the raw noisy pixel value would let each
    fluctuation set its own weight.
    """
    rate = torch.clamp(source_rate, min=0.0)
    t = torch.as_tensor(exptimes_s, dtype=rate.dtype, device=rate.device).view(-1, 1, 1)
    bg = torch.as_tensor(sigma_bg, dtype=rate.dtype, device=rate.device).view(-1, 1, 1)
    return (bg**2 + rate / (gain * t)).contiguous()


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def _read_fits(path: Path) -> np.ndarray:
    with fits.open(path) as hdul:
        data = np.squeeze(hdul[0].data).astype(np.float64)
    if data.ndim != 2:
        raise ValueError(f"{path} is not a 2-D image (shape {data.shape})")
    return data


def _translate_model_np(image: np.ndarray, shift_xy: tuple[float, float]) -> np.ndarray:
    """Sub-pixel shift of the *noiseless disk model* (never applied to data)."""
    tensor = torch.from_numpy(np.ascontiguousarray(image)).view(1, 1, *image.shape)
    shift = torch.tensor([shift_xy], dtype=torch.float64)
    return translate(tensor, shift).view(*image.shape).numpy()


def _gaussian_blur(image: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    radius = max(1, int(3.0 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=image.dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = (kernel / kernel.sum()).to(image.device)
    img = image.view(1, 1, *image.shape[-2:])
    img = torch.nn.functional.conv2d(img, kernel.view(1, 1, 1, -1), padding=(0, radius))
    img = torch.nn.functional.conv2d(img, kernel.view(1, 1, -1, 1), padding=(radius, 0))
    return img.view(*image.shape)


def _sigma_clipped_rms(values: np.ndarray, nsigma: float = 3.0, iters: int = 6) -> float:
    vals = values[np.isfinite(values)].ravel()
    for _ in range(iters):
        med, std = np.median(vals), np.std(vals)
        keep = np.abs(vals - med) < nsigma * std
        if keep.all() or std == 0.0:
            break
        vals = vals[keep]
    return float(np.std(vals))


def read_positions_csv(path: Path, n_epochs: int) -> np.ndarray:
    """Read approximate full-frame galaxy positions from a CSV file.

    Requires columns ``epoch, x, y`` (extra columns ignored, ``#`` comments
    allowed).  Returns positions ordered by epoch, shape ``(n_epochs, 2)``.
    """
    from astropy.table import Table

    table = Table.read(path, format="ascii.csv", comment="#")
    for col in ("epoch", "x", "y"):
        if col not in table.colnames:
            raise ValueError(f"{path} must contain columns 'epoch', 'x', 'y'")
    order = np.argsort(np.asarray(table["epoch"], dtype=int))
    epochs = np.asarray(table["epoch"], dtype=int)[order]
    if not np.array_equal(epochs, np.arange(n_epochs)):
        raise ValueError(
            f"{path} must contain exactly one row per epoch 0..{n_epochs - 1}"
        )
    return np.stack(
        [np.asarray(table["x"], dtype=float)[order], np.asarray(table["y"], dtype=float)[order]],
        axis=1,
    )


# --------------------------------------------------------------------------- #
# Galaxy position estimation / refinement
# --------------------------------------------------------------------------- #
def refine_galaxy_positions(
    galaxy_region: torch.Tensor,
    initial_xy: np.ndarray | None = None,
    centroid_halfwidth: int = 3,
) -> torch.Tensor:
    """Per-epoch galaxy ``(x, y)`` in region coordinates.

    Starting from ``initial_xy`` (or the smoothed-image peak when not given),
    the position is refined by a flux-weighted centroid of the positive part
    of the disk-subtracted image in a small window.
    """
    n, h, w = galaxy_region.shape
    positions = torch.zeros(n, 2, dtype=torch.float64)
    for i in range(n):
        if initial_xy is not None:
            px, py = int(round(initial_xy[i, 0])), int(round(initial_xy[i, 1]))
            px, py = min(max(px, 0), w - 1), min(max(py, 0), h - 1)
        else:
            smooth = _gaussian_blur(torch.nan_to_num(galaxy_region[i]))
            py, px = divmod(int(torch.argmax(smooth)), w)
        y0, y1 = max(py - centroid_halfwidth, 0), min(py + centroid_halfwidth + 1, h)
        x0, x1 = max(px - centroid_halfwidth, 0), min(px + centroid_halfwidth + 1, w)
        window = torch.clamp(torch.nan_to_num(galaxy_region[i, y0:y1, x0:x1]), min=0.0)
        total = window.sum()
        ys, xs = torch.meshgrid(
            torch.arange(y0, y1, dtype=window.dtype),
            torch.arange(x0, x1, dtype=window.dtype),
            indexing="ij",
        )
        if total > 0:
            positions[i, 0] = (window * xs).sum() / total
            positions[i, 1] = (window * ys).sum() / total
        else:  # pragma: no cover - degenerate fallback
            positions[i] = torch.tensor([float(px), float(py)])
    return positions


# --------------------------------------------------------------------------- #
# Disk-model geometry
# --------------------------------------------------------------------------- #
def load_full_frames(cfg: DataConfig) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """Read the epoch frames and the co-aligned (shifted) disk model.

    Returns ``(frames, disk_full, obs_paths)`` where ``frames`` has shape
    ``(N_ep, H, W)`` and ``disk_full`` is the smooth disk model translated by
    the co-alignment shift into observation pixel coordinates.
    """
    data_dir = Path(cfg.data_dir)
    obs_paths = sorted((data_dir / cfg.obs_dirname).glob(cfg.obs_pattern))
    if len(obs_paths) != constants.N_EPOCHS:
        raise FileNotFoundError(
            f"expected {constants.N_EPOCHS} epoch files matching "
            f"{cfg.obs_pattern!r} in {data_dir / cfg.obs_dirname}, found {len(obs_paths)}"
        )
    
    frames_list = [_read_fits(p) for p in obs_paths]
    frames = np.zeros((len(frames_list), max(frame.shape[0] for frame in frames_list), max(frame.shape[1] for frame in frames_list)))
    for i, frame in enumerate(frames_list):
        frames[i, :frame.shape[0], :frame.shape[1]] = frame

    disk_path = data_dir / cfg.disk_model_path
    disk_full = _translate_model_np(np.nan_to_num(_read_fits(disk_path)), cfg.disk_model_shift_xy)
    return frames, disk_full, obs_paths


def estimate_star_position(
    disk_full: np.ndarray, footprint_hw: tuple[int, int]
) -> tuple[float, float]:
    """Star position as the flux-weighted centroid of the shifted disk model.

    The HD 107146 ring is close to face-on, so the centroid of the (positive)
    scattered-light model coincides with the projected star position to well
    below a pixel.  The centroid is taken over the intersection of the model
    array with the observation footprint.
    """
    fh = min(disk_full.shape[0], footprint_hw[0])
    fw = min(disk_full.shape[1], footprint_hw[1])
    flux = np.clip(np.nan_to_num(disk_full[:fh, :fw]), 0.0, None)
    total = flux.sum()
    if total <= 0:
        raise ValueError(
            "shifted disk model has no positive flux in the observation "
            "footprint; check DataConfig.disk_model_shift_xy or set star_xy"
        )
    ys, xs = np.mgrid[0:fh, 0:fw]
    return float((flux * xs).sum() / total), float((flux * ys).sum() / total)


def ring_profile(
    disk_full: np.ndarray,
    star_xy: tuple[float, float],
    r_min: float,
    footprint_hw: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthally averaged disk SB profile outside ``r_min``.

    Only strictly positive, finite model pixels enter the average, which
    excludes the zero padding introduced by the co-alignment shift and any
    empty background.  Returns ``(radii, profile)`` (bin centres, counts/s/px).
    """
    fh, fw = disk_full.shape
    if footprint_hw is not None:
        fh, fw = min(fh, footprint_hw[0]), min(fw, footprint_hw[1])
    img = disk_full[:fh, :fw]
    ys, xs = np.mgrid[0:fh, 0:fw]
    r = np.hypot(xs - star_xy[0], ys - star_xy[1])
    r_max = min(star_xy[0], star_xy[1], fw - star_xy[0], fh - star_xy[1])
    if r_max <= r_min + 2.0:
        raise ValueError(
            f"star position {tuple(np.round(star_xy, 2))} leaves no room for the "
            f"ring profile (r_max={r_max:.1f} <= r_min={r_min:.1f}); "
            "set DataConfig.star_xy"
        )
    good = np.isfinite(img) & (img > 0)
    bins = np.arange(r_min, r_max, 1.0)
    radii = 0.5 * (bins[:-1] + bins[1:])
    profile = np.full(radii.shape, np.nan)
    for k in range(len(radii)):
        sel = (r >= bins[k]) & (r < bins[k + 1]) & good
        if sel.any():
            profile[k] = float(np.mean(img[sel]))
    return radii, profile


def _ring_peak(
    disk_full: np.ndarray,
    star_xy: tuple[float, float],
    r_min: float,
    footprint_hw: tuple[int, int] | None = None,
) -> tuple[float, float]:
    """Peak of the azimuthally averaged disk SB profile outside ``r_min``."""
    radii, profile = ring_profile(disk_full, star_xy, r_min, footprint_hw)
    if not np.isfinite(profile).any():
        raise ValueError("could not locate the disk ring surface-brightness peak")
    k_peak = int(np.nanargmax(profile))
    peak_sb, peak_radius = float(profile[k_peak]), float(radii[k_peak])
    if peak_sb <= 0:
        raise ValueError("could not locate the disk ring surface-brightness peak")
    return peak_sb, peak_radius


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def load_epoch_data(cfg: DataConfig) -> EpochData:
    """Build the native-grid, noise-modelled :class:`EpochData` from FITS files."""
    frames, disk_full, _ = load_full_frames(cfg)
    log.info("loaded %d epochs, frame shape %s", len(frames), frames[0].shape)

    fh, fw = frames[0].shape
    footprint = (fh, fw)

    # Star position: flux centroid of the co-aligned disk model (the ring is
    # near face-on, so this is the projected star position), unless supplied.
    if cfg.star_xy is not None:
        star_xy = tuple(map(float, cfg.star_xy))
    else:
        star_xy = estimate_star_position(disk_full, footprint)
        log.info(
            "star position from disk-model flux centroid: (x, y) = (%.2f, %.2f) px",
            *star_xy,
        )
    if not (0 <= star_xy[0] < fw and 0 <= star_xy[1] < fh):
        raise ValueError(
            f"star position {tuple(np.round(star_xy, 2))} lies outside the "
            f"{fw}x{fh} frame; set DataConfig.star_xy explicitly"
        )

    peak_sb, peak_radius = _ring_peak(
        disk_full, star_xy, constants.DISK_MODEL_INNER_VALID_RADIUS_PX, footprint
    )
    log.info("ring SB peak %.4g counts/s/px at r = %.1f px", peak_sb, peak_radius)

    # -------------------------------------------------------- working crop --
    x0, y0 = cfg.region_origin_xy
    s = cfg.region_size
    if y0 + s > fh or x0 + s > fw:
        raise ValueError(
            f"analysis region {(x0, y0)}+{s} exceeds the {fw}x{fh} epoch frames"
        )
    if y0 + s > disk_full.shape[0] or x0 + s > disk_full.shape[1]:
        raise ValueError(
            f"analysis region {(x0, y0)}+{s} exceeds the "
            f"{disk_full.shape[1]}x{disk_full.shape[0]} disk-model frame"
        )
    region = np.stack([f[y0 : y0 + s, x0 : x0 + s] for f in frames])
    disk_region = disk_full[y0 : y0 + s, x0 : x0 + s]

    valid_region = np.isfinite(region).astype(np.float64)
    region = np.nan_to_num(region)

    # ------------------------------------------------------------ positions --
    if cfg.galaxy_positions_xy is not None:
        initial_full = np.asarray(cfg.galaxy_positions_xy, dtype=np.float64)
    else:
        initial_full = None
        if cfg.galaxy_positions_csv is not None:
            csv_path = Path(cfg.data_dir) / cfg.galaxy_positions_csv
            if csv_path.exists():
                initial_full = read_positions_csv(csv_path, len(frames))
                log.info("initial galaxy positions from %s", csv_path)
    initial_region = None if initial_full is None else initial_full - np.array([x0, y0])
    galaxy_region = torch.from_numpy(region - disk_region[None])
    pos_region = refine_galaxy_positions(galaxy_region, initial_region)
    pos_full = pos_region.numpy() + np.array([x0, y0])
    log.info("galaxy positions (full frame):\n%s", np.array_str(pos_full, precision=2))

    # -------------------------------------------------- integer-pixel crops --
    cut = cfg.cutout_size
    centre = (cut - 1) / 2.0
    origin = np.rint(pos_region.numpy() - centre).astype(int)  # (N, 2) x/y
    lo, hi = origin.min(axis=0), (origin + cut).max(axis=0)
    if lo.min() < 0 or hi[0] > s or hi[1] > s:
        raise ValueError(
            "cutouts exceed the working region; enlarge DataConfig.region_size "
            f"or move region_origin_xy (crop span x:[{lo[0]},{hi[0]}), "
            f"y:[{lo[1]},{hi[1]}), region {s})"
        )

    def _crop(stack: np.ndarray) -> torch.Tensor:
        out = np.stack(
            [
                stack[i, origin[i, 1] : origin[i, 1] + cut, origin[i, 0] : origin[i, 0] + cut]
                for i in range(len(frames))
            ]
        )
        return torch.from_numpy(np.ascontiguousarray(out))

    obs_cut = _crop(region)
    valid_cut = _crop(valid_region)
    disk_cut = _crop(np.broadcast_to(disk_region, region.shape).copy())

    for ep, px, py in cfg.mask_pixels:
        valid_cut[ep, py, px] = 0.0
        log.info("masking pixel (x=%d, y=%d) in EP%d", px, py, ep)

    # Sub-pixel geometry: galaxy position within each native cutout and the
    # nominal model shifts (epoch 0 is the template frame, so s_0 = 0).
    pos_cutout = pos_region - torch.from_numpy(origin.astype(np.float64))
    nominal = (pos_cutout - pos_cutout[0]).clone()

    # ------------------------------------------------- fixed analysis masks --
    a = cfg.mask_semi_major + cfg.mask_margin_px
    b = cfg.mask_semi_minor + cfg.mask_margin_px
    mask = torch.stack(
        [
            make_ellipse_mask(a, b, cfg.mask_angle_deg, cut, center_xy=tuple(pos_cutout[i].tolist()))
            for i in range(len(frames))
        ]
    )
    _check_geometry(cut, a, b, cfg.mask_angle_deg, nominal, cfg.mask_margin_px)

    # ---------------------------------------------------------------- noise --
    if cfg.sigma_bg is not None:
        sigma_bg = np.full(len(frames), float(cfg.sigma_bg))
    else:
        sigma_bg = np.array([_sigma_clipped_rms(f) for f in frames])
        bad = ~np.isfinite(sigma_bg) | (sigma_bg <= 0)
        sigma_bg[bad] = constants.SIGMA_BG_DEFAULT
    log.info("background RMS per epoch: %s", np.array_str(sigma_bg, precision=4))

    exptimes = np.asarray(cfg.exptimes_s, dtype=np.float64)
    if len(exptimes) != len(frames):
        raise ValueError("exptimes_s length does not match the number of epochs")
    if cfg.variance_source == "background":
        source_rate = torch.zeros_like(obs_cut)
    else:
        # Plug-in initialization; ``variance_source == "model"`` replaces this
        # with a model-predicted rate in inference.prepare_data (one pass).
        source_rate = obs_cut
    variance = variance_map(source_rate, sigma_bg, exptimes, cfg.gain)

    disk_hat = torch.clamp(disk_cut / peak_sb, min=0.0)

    return EpochData(
        observations=obs_cut,
        disk=disk_cut,
        disk_hat=disk_hat,
        variance=variance,
        valid=valid_cut,
        mask=mask,
        nominal_shift_xy=nominal,
        galaxy_xy_cutout=pos_cutout,
        galaxy_xy_full=pos_full,
        crop_origin_xy=origin + np.array([x0, y0]),
        star_xy_full=star_xy,
        disk_peak_sb=peak_sb,
        disk_peak_radius_px=peak_radius,
        sigma_bg=sigma_bg,
        exptimes_s=exptimes,
        config=cfg,
    )


def _check_geometry(
    cut: int,
    semi_major: float,
    semi_minor: float,
    angle_deg: float,
    nominal: torch.Tensor,
    margin: float,
) -> None:
    """Warn if the analysis aperture can reach the template's padded border.

    The model prediction inside the aperture uses template pixels within the
    kernel support (2 px) of the shifted aperture; if that support crosses
    the cutout edge the zero padding truncates the template.
    """
    theta = math.radians(angle_deg)
    ext_x = math.hypot(semi_major * math.cos(theta), semi_minor * math.sin(theta))
    ext_y = math.hypot(semi_major * math.sin(theta), semi_minor * math.cos(theta))
    shift_max = float(nominal.abs().max()) + margin
    half = (cut - 1) / 2.0
    need = max(ext_x, ext_y) + shift_max + 2.0 + 0.5  # kernel support + crop rounding
    if need > half:
        log.warning(
            "analysis aperture (+offsets +kernel) needs %.1f px but the cutout "
            "half-size is %.1f px; enlarge DataConfig.cutout_size",
            need,
            half,
        )
