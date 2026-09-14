#!/usr/bin/env python3
"""Publication remakes of G. Schneider's GS02 / GS03 / GS04 figures for the
HD 107146 transit-photometry paper, in the paper's shared style
(``paper_figures.py``: cmasher *bubblegum*, inward ticks, ``save`` -> pdf+png).

Figures
-------
gs02    HD 107146 debris-ring transit (ingress) over the background galaxy,
        chronological panels EP0-EP7 (2 x 4).  Per-epoch AQ (NUP) images,
        log10 stretch [-2, -0.7] dex counts/s/px, 300 x 300 px sub-arrays
        co-registered on the obscured star.  Replaces figures/GS02.eps.
gs0304  GS03 and GS04 combined, 1 row x 4 columns with a horizontal
        colorbar over the image panels (asinh stretch, +/- the GS02 vmax):
          - EP0+EP1 AQ image (fully sampled disk; exposure-weighted
            valid-pixel mean for display, while the profile samples a
            5-px-cross median resample pooling both epochs; registration
            verified against the galaxy reflex track, clean integer
            slips corrected losslessly),
          - fitted disk model with the major-axis cut and the on-sky
            galaxy track drawn (lines only),
          - representative-epoch AQ image minus the disk model
            (disk-light rejection; galaxy left standing),
          - surface-brightness cross-section along the disk major axis:
            EP0+EP1 (mean) and EP2-EP7 (median) images vs the model, each
            sampled as the median of a short constant-radius arc swept
            along the cut (rejects hot pixels and the radial PSF-residual
            streaks with no radial broadening; --arc-halfwidth 0 gives
            the plain per-pixel cut of the original GS03), with the
            r < 65 px PSF-residual zone, the stellocentric radii of the
            galaxy sight-lines, and the ACS/JWST sight-lines marked.
        Replaces figures/GS03.jpg + figures/GS04.eps.
diskalma
        All eight epochs combined into one galaxy-free disk image, with
        the ARKS ALMA band 6+7 (1.0 mm reference) continuum contours of the same disk
        overlaid.  Each epoch is masked with the pipeline galaxy ellipse
        (margin-enlarged) at its own measured position *before* the
        registration step, so the mask rides along with any integer-slip
        correction; the masked frames are then valid-pixel median (or
        exposure-weighted mean) combined.  The galaxy moves ~4.5 px/yr
        between epoch groups, so every masked zone is refilled by the
        epochs where the galaxy sat elsewhere, and the median also
        suppresses the halo wings beyond the mask plus the epoch-specific
        PSF-residual streaks and hot pixels.  Visualization figure only
        (uneven effective depth inside the galaxy corridor) -- not a
        science product.  The contour geometry is the vector-path
        extraction of the ARKS gallery-figure HD 107146 panel
        (hd107146_contours.json / .npz, star-centred (dRA, dDec) arcsec,
        levels [5, 18, 24] sigma default (30 also in the file),
        0.82" beam); see --alma.

The heavy lifting is reused: data access goes through the analysis module
(``self_sufficient_ultranest.py`` -- pass with ``--module``), style/track/
centering helpers come from ``paper_figures.py`` (same directory), and the
cross-section is sampled with ``scipy.ndimage.map_coordinates``.

Usage
-----
    python paper_figures_schneider.py --module self_sufficient_ultranest.py
    python paper_figures_schneider.py <log_dir> --figs gs0304 \
        --module path/to/self_sufficient_ultranest.py
    python paper_figures_schneider.py --figs diskalma \
        --alma literature_data/arks_alma/extracted

Passing any diskalma-specific option (``--alma``, ``--combine``,
``--gal-margin``, ``--alma-levels``) adds ``diskalma`` to ``--figs``
automatically.

``log_dir`` is optional: when given, the posterior-median registration
offsets refine the per-epoch galaxy positions (sub-pixel; invisible at
figure scale), exactly as in ``paper_figures.py`` figure 2.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import matplotlib.patheffects as pe                 # noqa: E402
from matplotlib import colors as mcolors            # noqa: E402
from matplotlib.colors import LogNorm               # noqa: E402
from matplotlib.patches import Ellipse              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import paper_figures as pf                      # noqa: E402  shared style
except ImportError as exc:                          # pragma: no cover
    sys.exit(f"paper_figures.py must sit next to this script ({exc})")

# --------------------------------------------------------------------------- #
# shared constants (GS02 caption / proposal bookkeeping)
# --------------------------------------------------------------------------- #
GO_IDS = (12228, 12228, 14714, 14714, 15221, 15221, 15501, 15501)
SUBARRAY = 300                  # px; GS02 sub-array extract size
LOG_RANGE_DEX = (-2.0, -0.7)    # GS02 log10 display stretch [counts/s/px]
LIN_WIDTH = 0.01                # gs0304 asinh norm: linear core half-width
EP_COMBINE = (0, 1)             # "fully sampled" pre-transit AQ combination
R_AGREE_PX = 65.0               # model/AQ agreement radius (GS03 caption)
APERTURE_AB_THETA = (9.0, 5.0, 35.0)   # galaxy ellipse of the pipeline masks
PROFILE_SIGMA_PX = 0.0          # optional pre-smoothing of the cut sampling (0 = off)
ARC_HALF_PX = 6.0               # tangential arc half-length of the robust cut median
STROKE = [pe.withStroke(linewidth=1.6, foreground="0.25")]

# diskalma: all-epoch stack + ARKS ALMA overlay ----------------------------- #
GAL_MARGIN_STACK = 10.0         # px; extra galaxy-ellipse enlargement for the
                                # per-epoch masks of the all-epoch stack (the
                                # pipeline's own margin is 1.5 px; the stack
                                # can afford to cut generously -- the other
                                # epochs refill everything)
GAL_ASPECT_STACK = 2.5          # stretch of the mask ellipse's (margined)
                                # major axis; catches the halo tail along the
                                # major axis without also fattening the mask
                                # across the galaxy track (which is nearly
                                # perpendicular, so the refill still works)
ALMA_FILES = ("hd107146_contours.json", "hd107146_contours.npz")
ALMA_SUBDIR = Path("literature_data") / "arks_alma" / "extracted"
ALMA_LABEL = "ALMA 1.0\u2009mm (B6+B7)"  # ARKS band 6+7 multi-frequency image, 300 GHz reference (Marino et al. 2026)
ALMA_LEVELS_DEFAULT = (5, 18, 24)  # sigma levels drawn by default (the file
                                # also carries 30; --alma-levels overrides)
ALMA_BEAM_FWHM_ARCSEC = 0.819   # circular; vector beam circle of the gallery
                                # panel: 13.65 x 13.67 pt on the 266.87-pt =
                                # 16" axes box (0.8184" x 0.8198")
DISKALMA_VMAX = 0.1             # counts/s/px; upper display stretch of the
                                # stack (-1.0 dex; GS02 uses -0.7)

CM_LOG = pf.CMAP_SEQ.copy()
CM_LOG.set_bad(CM_LOG(0.0))     # masked / non-positive -> zero color
CM_LIN = pf.CMAP_SEQ.copy()
CM_LIN.set_bad(CM_LIN(0.5))     # masked -> mid (zero of the +/- stretch)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def load_frames(ssu):
    """Per-epoch AQ (NUP) frames as float numpy arrays, NaN where invalid.

    The reductions flag unsampled / degraded pixels either as NaN or as
    exact zeros (masked-median combination); both count as invalid here.
    """
    frames = [pf.to_np(f).astype(float) for f in ssu.load_observations()]
    return [np.where(np.isfinite(f) & (f != 0.0), f, np.nan) for f in frames]


def epoch_positions(ssu, log_dir=None):
    """Per-epoch galaxy positions in full-frame pixels; the posterior-median
    registration offsets are applied when a run directory is given (same
    convention as paper_figures.fig_disk_track)."""
    pos = ssu.load_position_guess()
    q_full = np.stack([pos["x_est"].to_numpy(),
                       pos["y_est"].to_numpy()], axis=1)[: len(pf.EPOCH_DATES)]
    if log_dir:
        # layout-aware (v06: offsets + tau'; v07: + per-epoch noise scales)
        q_full = q_full + pf.median_offsets(pf.load_info(log_dir))
    return q_full


def sign_convention(q_full):
    """(sx, sy) mapping (RA, Dec) mas onto pixel x/y, self-calibrated
    against the measured epoch-to-epoch galaxy motion (as in
    paper_figures._plot_disk_track); falls back to N-up/E-left.

    Also returns the per-epoch residual vectors (measured - reflex-
    predicted, px, anchored at EP0): the galaxy is the one sharp feature
    with exactly known expected motion, so these residuals measure each
    frame's registration on the common grid -- a mis-indexed frame (e.g.
    a trimmed bottom row) shows up as a ~integer vector here."""
    from astropy.time import Time

    try:
        t_ep = Time(list(pf.EPOCH_DATES))
        d_ra, d_dec = pf.galaxy_track_mas(t_ep, pf.EPOCH_DATES[0])
        meas = (q_full - q_full[0]) * pf.PIXEL_SCALE_ARCSEC * 1000.0
        sx, sy = min(((a, b) for a in (1, -1) for b in (1, -1)),
                     key=lambda s: np.hypot(s[0] * d_ra - meas[:, 0],
                                            s[1] * d_dec - meas[:, 1]).mean())
        res = np.stack([meas[:, 0] - sx * d_ra, meas[:, 1] - sy * d_dec],
                       axis=1) / (pf.PIXEL_SCALE_ARCSEC * 1000.0)
        print(f"track sign convention (sx, sy) = ({sx}, {sy}); "
              f"mean |predicted - measured| = "
              f"{np.hypot(*res.T).mean():.2f} px")
        print("per-epoch (dx, dy) track residual [px] "
              "(frame-registration cross-check):")
        for i, r in enumerate(res):
            print(f"  EP{i}: ({r[0]:+.2f}, {r[1]:+.2f})")
        return sx, sy, res
    except Exception as exc:                                # noqa: BLE001
        warnings.warn(f"sign self-calibration failed ({exc}); "
                      "assuming N up, E left")
        return -1, 1, np.zeros((len(q_full), 2))


def crop(img, star, half):
    """Square integer crop of ``2*half`` px about ``star`` plus its arcsec
    extent (relative to the sub-pixel star position, origin='lower').
    Regions outside the frame are NaN-padded, so a window may extend past
    the frame edge (the AQ FOVs are not centered on the star)."""
    x0 = int(round(star[0])) - half
    y0 = int(round(star[1])) - half
    n = 2 * half
    cut = np.full((n, n), np.nan, dtype=float)
    ys = slice(max(y0, 0), min(y0 + n, img.shape[0]))
    xs = slice(max(x0, 0), min(x0 + n, img.shape[1]))
    cut[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = \
        img[ys, xs]
    s = pf.PIXEL_SCALE_ARCSEC
    ext = ((x0 - 0.5 - star[0]) * s, (x0 + n - 0.5 - star[0]) * s,
           (y0 - 0.5 - star[1]) * s, (y0 + n - 0.5 - star[1]) * s)
    return cut, ext


def on_common_canvas(arrays):
    """Origin-aligned NaN-padded copies on the joint (max_ny, max_nx) grid.

    The reduced AQ frames share one pixel coordinate system (integer index
    == full-frame coordinate, exactly as in load_native_data's crops) but
    not one extent: the row count differs between epochs, and the shifted
    disk model is a different shape again."""
    ny = max(a.shape[0] for a in arrays)
    nx = max(a.shape[1] for a in arrays)
    out = []
    for a in arrays:
        c = np.full((ny, nx), np.nan)
        c[: a.shape[0], : a.shape[1]] = a
        out.append(c)
    return out


def int_shift(img, dxy):
    """Lossless integer translation, NaN-padded: out[y, x] = img[y-dy, x-dx]
    (no resampling of the observations, per the pipeline's own rule)."""
    dx, dy = int(round(dxy[0])), int(round(dxy[1]))
    out = np.full_like(img, np.nan)
    out[max(dy, 0):img.shape[0] + min(dy, 0),
        max(dx, 0):img.shape[1] + min(dx, 0)] = \
        img[max(-dy, 0):img.shape[0] - max(dy, 0),
            max(-dx, 0):img.shape[1] - max(dx, 0)]
    return out


def mask_galaxies(img, q_full, idx, margin=3.0, aspect=1.0):
    """Copy of ``img`` with the pipeline's galaxy ellipse NaN-ed at the
    positions of the given epochs.  ``aspect`` stretches the margined
    major axis (minor unchanged), i.e. it multiplies the mask's axis
    ratio."""
    out = img.copy()
    yy, xx = np.mgrid[0:out.shape[0], 0:out.shape[1]]
    a, b, th = APERTURE_AB_THETA
    for i in idx:
        out[in_ellipse(xx, yy, q_full[i], (a + margin) * aspect,
                       b + margin, th, margin=0.0)] = np.nan
    return out


def verified_members(frames, idx, res_xy=None):
    """NaN-padded, common-grid copies of the chosen epochs, with the
    registration verified before use.

    ``res_xy`` (per-epoch galaxy-track residual vectors from
    sign_convention) verifies the members share one grid: residuals are
    EP0-anchored, so the pairwise difference is the relative registration
    of the two arrays.  Image cross-correlation is *not* used -- the disk
    is too smooth (~46 px ring) to constrain +/-1 px shifts and the
    epoch-specific PSF spikes then dominate; the galaxy is the one sharp
    feature whose expected motion is exactly known.  A clean integer slip
    (>= 1 px, within 0.3 px of an integer) is corrected by a lossless
    integer shift; sub-pixel scatter is normal pointing/measurement
    dispersion and is left untouched (no resampling of observations)."""
    imgs = on_common_canvas([frames[i] for i in idx])
    if res_xy is not None:
        for j, i in enumerate(idx[1:], 1):
            d = np.asarray(res_xy[i]) - np.asarray(res_xy[idx[0]])
            di = np.round(d).astype(int)
            msg = (f"EP{i} vs EP{idx[0]} registration offset from the "
                   f"galaxy positions (measured - reflex-predicted): "
                   f"({d[0]:+.2f}, {d[1]:+.2f}) px")
            if (np.abs(di) >= 1).any() and np.all(np.abs(d - di) < 0.3):
                warnings.warn(msg + f" -> clean integer slip; shifting "
                                    f"by ({-di[0]}, {-di[1]})")
                imgs[j] = int_shift(imgs[j], (-di[0], -di[1]))
            else:
                print(msg + ("  [co-registered]"
                             if np.all(np.abs(d) < 0.5)
                             else "  [sub-pixel / ambiguous -- not shifted]"))
    return imgs


def stack_mean(imgs, weights=None):
    """Valid-pixel (optionally weighted) mean; weights are renormalized
    per pixel over the epochs that actually cover it."""
    stack = np.stack(imgs)
    valid = np.isfinite(stack)
    w = (np.ones(len(imgs)) if weights is None
         else np.asarray(weights, float))
    w = w / w.sum()
    wm = np.where(valid, w[:, None, None], 0.0)
    wsum = wm.sum(0)
    with np.errstate(invalid="ignore"):
        out = np.nansum(np.where(valid, stack, 0.0) * wm, 0) \
            / np.where(wsum > 0, wsum, np.nan)
    return np.where(wsum > 0, out, np.nan)


def cross_median(imgs):
    """Per-pixel median over the single-pixel crosses (centre + the four
    direct neighbours) of *every* input image: with two epochs this pools
    2 x 5 = 10 samples per pixel, giving genuine outlier rejection that a
    straight two-frame per-pixel median cannot (the median of two values
    is their mean).  Radial mixing is limited to +/-1 px -- negligible
    against the 46 px ring FWHM."""
    layers = []
    for im in imgs:
        for dxy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            layers.append(im if dxy == (0, 0) else int_shift(im, dxy))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.stack(layers), axis=0)


def combine_epochs(frames, idx=EP_COMBINE, res_xy=None, method="mean",
                   weights=None):
    """Valid-pixel combination of the given epochs (``method``: "mean" or
    "median"; "mean" honours per-epoch ``weights``) -> fully sampled
    image.  Registration is verified first (see verified_members)."""
    imgs = verified_members(frames, idx, res_xy)
    if method == "median":
        stack = np.stack(imgs)
        n = np.isfinite(stack).sum(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = np.nanmedian(stack, axis=0)
        return np.where(n > 0, out, np.nan)
    return stack_mean(imgs, weights)


def nan_gaussian(img, sigma):
    """NaN-aware Gaussian smoothing (validity-weighted): sampling the
    result along the cut is the Gaussian-distance-weighted kernel moving
    along the line.  Pixels with < 35% kernel support stay NaN."""
    if sigma <= 0:
        return img
    from scipy.ndimage import gaussian_filter
    v = np.isfinite(img)
    num = gaussian_filter(np.where(v, img, 0.0), sigma, mode="constant")
    den = gaussian_filter(v.astype(float), sigma, mode="constant")
    out = np.full(img.shape, np.nan)
    ok = den > 0.35
    out[ok] = num[ok] / den[ok]
    return out


def minus_model(frame, disk):
    """frame - disk on their overlap (the shifted model is zero-padded
    beyond its coverage, so outside the overlap the frame is unchanged)."""
    out = frame.copy()
    h = min(frame.shape[0], disk.shape[0])
    w = min(frame.shape[1], disk.shape[1])
    out[:h, :w] -= np.nan_to_num(disk[:h, :w])
    return out


def in_ellipse(x, y, center, a, b, theta_deg, margin=1.5):
    th = np.deg2rad(theta_deg)
    u = (x - center[0]) * np.cos(th) + (y - center[1]) * np.sin(th)
    v = -(x - center[0]) * np.sin(th) + (y - center[1]) * np.cos(th)
    return (u / (a + margin)) ** 2 + (v / (b + margin)) ** 2 <= 1.0


def compass(ax, sx, sy, frac=0.105, color="white", pad_top=None,
            pad_right=0.045):
    """Thin pointy N/E arrows joined at a common base, top-right corner
    (same construction as paper_figures figure 2).  ``pad_top`` /
    ``pad_right``: extra fractional insets so the compass clears an
    in-panel header line or the right-spine tick marks."""
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    ln = frac * span
    bx = ax.get_xlim()[1] - pad_right * span - (ln if sx > 0 else 0.0)
    by = (ax.get_ylim()[1] - (pad_top or 0.045) * span
          - (ln if sy > 0 else 0.0))
    for lbl, (dx, dy) in (("E", (sx * ln, 0.0)), ("N", (0.0, sy * ln))):
        ax.annotate("", xy=(bx + dx, by + dy), xytext=(bx, by), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.7,
                                    mutation_scale=7, shrinkA=0, shrinkB=0,
                                    joinstyle="miter", capstyle="butt"))
        ax.text(bx + 1.32 * dx, by + 1.32 * dy, lbl, color=color,
                fontsize=7, ha="center", va="center", zorder=6,
                path_effects=STROKE)


def scale_bar(ax, au=100.0, color="white"):
    """Projected scale bar (lower right), au -> arcsec via the parallax."""
    length = au * pf.STAR_PARALLAX_MAS / 1000.0            # arcsec
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    x1 = ax.get_xlim()[1] - 0.045 * span
    y = ax.get_ylim()[0] + 0.055 * span
    ax.plot([x1 - length, x1], [y, y], color=color, lw=1.6,
            solid_capstyle="butt", zorder=6)
    ax.text(x1 - length / 2.0, y + 0.014 * span, f"{au:g} au", color=color,
            fontsize=6.5, ha="center", va="bottom", zorder=6,
            path_effects=STROKE)


def star_marker(ax, size=55):
    ax.scatter([0], [0], marker="*", s=size, color="white",
               edgecolor="black", lw=0.5, zorder=6)


def galaxy_track_arcsec(q_full, star, sx, sy, t_span=(2004.0, 2028.0)):
    """On-sky galaxy track (stellar reflex: proper motion + parallax) in
    arcsec offsets from the star, anchored at the epoch-0 galaxy position
    -- identical construction to paper_figures figure 2."""
    from astropy.time import Time

    s = pf.PIXEL_SCALE_ARCSEC
    tt = Time(np.arange(t_span[0], t_span[1] + 1e-9, 1.0 / 36.0),
              format="jyear")
    d_ra, d_dec = pf.galaxy_track_mas(tt, pf.EPOCH_DATES[0])
    return np.stack(
        [(q_full[0, 0] - star[0]) * s + sx * d_ra / 1000.0,
         (q_full[0, 1] - star[1]) * s + sy * d_dec / 1000.0], axis=-1)


# --------------------------------------------------------------------------- #
# GS02: transit epochs E0-E7, 2 x 4
# --------------------------------------------------------------------------- #
def fig_gs02(module_path, out_dir, log_dir=None):
    ssu = pf.import_module(module_path)
    frames = load_frames(ssu)
    disk = pf.to_np(ssu.load_disk_model()).astype(float)
    star = np.array(pf.find_star_center(disk))
    q_full = epoch_positions(ssu, log_dir)
    sx, sy, _ = sign_convention(q_full)
    print("star (full-frame px): %.2f, %.2f" % tuple(star))

    _caption_stats(frames, disk, star, q_full)

    n = len(frames)
    dates = [datetime.strptime(d, "%Y-%m-%d").strftime("%Y-%b-%d").upper()
             for d in pf.EPOCH_DATES[:n]]

    pf.apply_style()
    fig = plt.figure(figsize=(7.05, 3.46))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.07],
                          wspace=0.045, hspace=0.045)
    norm = LogNorm(vmin=10.0 ** LOG_RANGE_DEX[0],
                   vmax=10.0 ** LOG_RANGE_DEX[1])
    axes, im, share = [], None, None
    for i in range(n):
        r, c = divmod(i, 4)
        ax = fig.add_subplot(gs[r, c], sharex=share, sharey=share)
        share = share or ax
        img, ext = crop(frames[i], star, SUBARRAY // 2)
        im = ax.imshow(np.ma.masked_invalid(img), origin="lower",
                       cmap=CM_LOG, norm=norm, extent=ext,
                       interpolation="nearest")
        star_marker(ax, size=42)
        go = GO_IDS[i] if i < len(GO_IDS) else ""
        ax.text(0.035, 0.965, f"(EP{i})  {go}  {dates[i]}",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.0,
                color="white", zorder=6, path_effects=STROKE)
        ax.tick_params(labelsize=6.5, color="white")
        if r == 0:
            ax.tick_params(labelbottom=False)
        if c > 0:
            ax.tick_params(labelleft=False)
        axes.append(ax)
    axes[0].set_xlim(-SUBARRAY / 2 * pf.PIXEL_SCALE_ARCSEC,
                     SUBARRAY / 2 * pf.PIXEL_SCALE_ARCSEC)
    axes[0].set_ylim(*axes[0].get_xlim())
    axes[0].set_xticks((-5.0, 0.0, 5.0))
    axes[0].set_yticks((-5.0, 0.0, 5.0))
    axes[4].set_xlabel(r"$\Delta x$ [arcsec]", fontsize=6.0)
    axes[4].set_ylabel(r"$\Delta y$ [arcsec]", fontsize=6.0)
    compass(axes[3], sx, sy, pad_top=0.215, pad_right=0.105)  # top-right panel only
    scale_bar(axes[7], au=100.0)

    # pipeline galaxy-mask footprint (margin-enlarged, as applied) on the
    # lower-right panel
    ea, eb, eth = APERTURE_AB_THETA
    em = 1.5                                   # make_ellipse_mask margin [px]
    s = pf.PIXEL_SCALE_ARCSEC
    axes[n - 1].add_patch(Ellipse((q_full[n - 1] - star) * s,
                                  2.0 * (ea + em) * s, 2.0 * (eb + em) * s,
                                  angle=eth, fill=False, ec="white",
                                  lw=0.5, zorder=6))

    cax = fig.add_subplot(gs[:, -1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"counts s$^{-1}$ px$^{-1}$", fontsize=6.0)
    tks = (0.01, 0.02, 0.05, 0.1, 0.2)
    cb.set_ticks(tks)
    cb.set_ticklabels([f"{v:g}" for v in tks])
    cb.ax.tick_params(labelsize=6.5)
    cb.ax.minorticks_off()

    # pin the colorbar to the actual (aspect-shrunk) panel extents
    fig.draw_without_rendering()
    tr = axes[3].get_position()
    br = axes[7].get_position()
    cp = cax.get_position()
    cax.set_position([tr.x1 + 0.008, br.y0, cp.width, tr.y1 - br.y0])
    pf.save(fig, out_dir, "GS02_transit_epochs")


def _caption_stats(frames, disk, star, q_full):
    """Recompute the numbers quoted in the GS02 caption (Gmax, Dmax at the
    galaxy's stellocentric distance, background sigma_rms).  Each frame is
    evaluated on its own grid -- the epochs differ in shape."""
    a, b, th = APERTURE_AB_THETA
    grids = {}

    def grid(shape):
        if shape not in grids:
            yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
            grids[shape] = (xx, yy)
        return grids[shape]

    sig, gmax = [], []
    for f, q in zip(frames, q_full):
        xx, yy = grid(f.shape)
        r = np.hypot(xx - star[0], yy - star[1])
        gal_any = np.zeros(f.shape, bool)
        for qq in q_full:
            gal_any |= in_ellipse(xx, yy, qq, a, b, th, margin=4.0)
        bg = (r > min(190.0, 0.45 * min(f.shape))) & ~gal_any
        sig.append(np.nanstd(np.where(bg, f, np.nan)))
        gmax.append(np.nanmax(np.where(in_ellipse(xx, yy, q, a, b, th),
                                       f, np.nan)))
    r_gal = np.hypot(*(q_full - star).T)
    ang = np.linspace(0, 2 * np.pi, 721)
    from scipy.ndimage import map_coordinates
    ring = map_coordinates(np.nan_to_num(disk),
                           [star[1] + r_gal[0] * np.sin(ang),
                            star[0] + r_gal[0] * np.cos(ang)], order=1)
    print(f"caption stats: Gmax = {min(gmax):.2f}..{max(gmax):.2f} "
          f"counts/s/px (per epoch); Dmax(r = r_gal(E0) = {r_gal[0]:.0f} px)"
          f" = {ring.max():.3f} counts/s/px; "
          f"sigma_rms = {np.mean(sig):.1e} counts/s/px "
          f"(background r > 190 px, galaxy excluded)")


# --------------------------------------------------------------------------- #
# GS03+GS04 combined: AQ | model (+cut, +track) | rejection | cross-section
# --------------------------------------------------------------------------- #
def major_axis_dir(disk, star, pa_sky_deg=None, sx=-1, sy=1, r_fit=150.0):
    """Unit vector (pixel frame) of the disk major axis: principal axis of
    the flux-weighted second moments about the star, or a fixed sky PA
    (deg E of N) when given.  Returns (u, pa_sky_deg)."""
    if pa_sky_deg is None:
        ny, nx = disk.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        dx, dy = xx - star[0], yy - star[1]
        w = np.where(np.hypot(dx, dy) <= r_fit,
                     np.clip(np.nan_to_num(disk), 0.0, None), 0.0)
        mxx = (w * dx * dx).sum() / w.sum()
        myy = (w * dy * dy).sum() / w.sum()
        mxy = (w * dx * dy).sum() / w.sum()
        lam = 0.5 * (mxx + myy) + np.hypot(0.5 * (mxx - myy), mxy)
        u = np.array([mxy, lam - mxx])
        if not np.hypot(*u) > 0:
            u = np.array([1.0, 0.0])
    else:
        th = np.deg2rad(pa_sky_deg)
        u = np.array([sx * np.sin(th), sy * np.cos(th)])   # N = sy*y, E = sx*x
    u = u / np.hypot(*u)
    pa = np.rad2deg(np.arctan2(sx * u[0], sy * u[1])) % 180.0
    return u, pa


def cross_section(img, star, u, radii, half_arc=0.0):
    """Cut along +/-u through the star, robustified tangentially.

    For each radius the two semi-axes are sampled along short arcs of
    *constant stellocentric radius* (arc length +/-``half_arc`` px, 1 px
    spacing, bilinear) centred on the cut directions; each side
    contributes the median of its valid arc samples and the two sides are
    averaged.  The PSF-subtraction residual streaks are radial, so they
    cross the arc and are rejected by the median together with hot/CR
    pixels -- while the radial resolution stays that of a 1-px cut (no
    ring broadening; the arc deviates from the disk ellipse by << 0.1 px
    for the widths used here).  ``half_arc = 0`` is the plain per-pixel
    two-side cut of the original GS03.  Returns (mean-of-side-medians,
    [side+, side-])."""
    from scipy.ndimage import map_coordinates

    valid = np.isfinite(img)
    filled = np.where(valid, img, 0.0)
    th0 = np.arctan2(u[1], u[0])
    n_arc = max(int(round(half_arc)), 0)
    off = np.arange(-n_arc, n_arc + 1, dtype=float)          # arc-length px
    sides = []
    for base in (th0, th0 + np.pi):
        prof = np.full(len(radii), np.nan)
        th = base + off[None, :] / np.maximum(radii, 1.0)[:, None]
        x = star[0] + radii[:, None] * np.cos(th)
        y = star[1] + radii[:, None] * np.sin(th)
        v = map_coordinates(filled, [y.ravel(), x.ravel()], order=1,
                            cval=0.0).reshape(x.shape)
        f = map_coordinates(valid.astype(float), [y.ravel(), x.ravel()],
                            order=1, cval=0.0).reshape(x.shape)
        v = np.where(f > 0.999, v, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            prof = np.nanmedian(v, axis=1)
        sides.append(prof)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(sides), axis=0), sides


def _peak_and_hwhm(radii, prof):
    """Peak radius (parabolic sub-sample refinement) and the inner/outer
    half-maximum radii of a single-peaked profile."""
    k = int(np.nanargmax(prof))
    if 0 < k < len(prof) - 1:
        d = prof[k - 1] - 2 * prof[k] + prof[k + 1]
        off = 0.0 if d == 0 else 0.5 * (prof[k - 1] - prof[k + 1]) / d
    else:
        off = 0.0
    r_pk = radii[k] + off * (radii[1] - radii[0])
    half = prof[k] / 2.0
    lo = np.where(prof[:k] <= half)[0]
    hi = np.where(prof[k:] <= half)[0]
    r_in = np.interp(half, prof[lo[-1]:k + 1], radii[lo[-1]:k + 1]) \
        if len(lo) else np.nan
    r_out = np.interp(-half, -prof[k:k + hi[0] + 1],
                      radii[k:k + hi[0] + 1]) if len(hi) else np.nan
    return r_pk, float(prof[k]), r_in, r_out


def _diagnose_model_agreement(out_dir, disk, star, u_maj, radii, half_arc,
                              aq_img, l8_img, prof_aq, prof_l8, prof_md,
                              frames, idx_late, res_xy, q_full):
    """Where does the data-vs-model profile deviation live?

    Prints multiplicative data/model scale factors per cut side, per
    radial band, and along the minor axis, plus the peak locations, and
    cross-checks with the (frame - model) rejection residuals of the late
    epochs -- an estimate that is independent of the EP0+EP1 processing
    chain.  Also writes a two-panel diagnostic figure (per-side curves;
    data/model ratio)."""
    bands = ((65.0, 95.0), (95.0, 130.0), (130.0, 180.0))

    def band_scale(d, m, lo, hi):
        sel = (radii >= lo) & (radii <= hi) \
            & np.isfinite(d) & np.isfinite(m)
        return float((d[sel] * m[sel]).sum() / (m[sel] ** 2).sum()) \
            if sel.any() else np.nan

    def peak(p):
        sel = radii >= 43.0
        r_pk, v, _, _ = _peak_and_hwhm(radii[sel], p[sel])
        return r_pk, v

    _, d_sides = cross_section(aq_img, star, u_maj, radii, half_arc)
    _, m_sides = cross_section(np.nan_to_num(disk), star, u_maj, radii,
                               half_arc)
    n_min = np.array([-u_maj[1], u_maj[0]])
    d_minor, _ = cross_section(aq_img, star, n_min, radii, half_arc)
    m_minor, _ = cross_section(np.nan_to_num(disk), star, n_min, radii,
                               half_arc)

    print("---- model-agreement diagnostics " + "-" * 38)
    rows = (("AQ major +u ", d_sides[0], m_sides[0]),
            ("AQ major -u ", d_sides[1], m_sides[1]),
            ("AQ major avg", prof_aq, prof_md),
            ("med major avg", prof_l8, prof_md),
            ("AQ minor avg", d_minor, m_minor))
    for tag, d, m in rows:
        sc = [band_scale(d, m, lo, hi) for lo, hi in bands]
        rp_d, v_d = peak(d)
        rp_m, v_m = peak(m)
        print(f"  {tag}: data/model "
              + "  ".join(f"[{lo:.0f}-{hi:.0f}px] {s:5.3f}"
                          for (lo, hi), s in zip(bands, sc))
              + f"  | peak {v_d:.4f}@{rp_d:.1f} vs {v_m:.4f}@{rp_m:.1f}")

    # independent cross-check: late-epoch rejection residuals on the cut
    resid = []
    for i in idx_late:
        r_img = mask_galaxies(minus_model(frames[i], disk), q_full, [i])
        p, _ = cross_section(r_img, star, u_maj, radii, half_arc)
        resid.append(p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        resid = np.nanmean(np.stack(resid), axis=0)
    sel = (radii >= 70.0) & (radii <= 95.0)
    s_pk = band_scale(prof_l8, prof_md, 70.0, 95.0)
    print(f"  mean EP2-EP7 (frame - model) on the cut, 70-95 px: "
          f"{np.nanmean(resid[sel]):+.2e} counts/s/px; a model "
          f"{100.0 * (s_pk - 1.0):+.1f}% off there predicts "
          f"{np.nanmean(prof_md[sel]) * (s_pk - 1.0):+.2e}")

    pf.apply_style()
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    c0, c1 = pf.palette(2, 0.25, 0.80)
    sel = radii >= 43.0
    for c, ds, ms, lab in ((c0, d_sides[0], m_sides[0], "+u"),
                           (c1, d_sides[1], m_sides[1], "-u")):
        axl.plot(radii[sel], ds[sel], color=c, lw=0.8,
                 label=f"data {lab}")
        axl.plot(radii[sel], ms[sel], color=c, lw=0.9, ls="-.", alpha=0.7,
                 label=f"model {lab}")
    axl.set_xlim(43.0, 185.0)
    axl.set_xlabel("radius [px]", fontsize=7)
    axl.set_ylabel(r"counts s$^{-1}$ px$^{-1}$", fontsize=7)
    axl.legend(fontsize=6, loc="upper right")
    axl.set_title("major-axis cut, per side", fontsize=8)
    with np.errstate(divide="ignore", invalid="ignore"):
        for d, m, lab, ls in ((prof_aq, prof_md, "AQ major", "-"),
                              (prof_l8, prof_md, "median major",
                               (0, (1.4, 1.6))),
                              (d_minor, m_minor, "AQ minor",
                               (0, (3.0, 1.5)))):
            mm = np.where(m > 1e-4, m, np.nan)
            axr.plot(radii, d / mm, lw=0.8, ls=ls, label=lab)
    axr.axhline(1.0, color="0.4", lw=0.7)
    for lo, hi in bands:
        axr.axvspan(lo, hi, color="0.6", alpha=0.08, lw=0)
    axr.set_xlim(60.0, 185.0)
    axr.set_ylim(0.75, 1.35)
    axr.set_xlabel("radius [px]", fontsize=7)
    axr.set_ylabel("data / model", fontsize=7)
    axr.legend(fontsize=6, loc="upper right")
    axr.set_title("profile ratio", fontsize=8)
    fig.tight_layout()
    pf.save(fig, out_dir, "GS03_diagnostics")


def fig_gs0304(module_path, out_dir, log_dir=None, rep_epoch=2,
               pa_major=None, t_span=(2004.0, 2028.0),
               prof_sigma=PROFILE_SIGMA_PX, half_arc=ARC_HALF_PX,
               diagnose=False):
    ssu = pf.import_module(module_path)
    frames = load_frames(ssu)
    disk = pf.to_np(ssu.load_disk_model()).astype(float)
    star = np.array(pf.find_star_center(disk))
    q_full = epoch_positions(ssu, log_dir)
    sx, sy, res_xy = sign_convention(q_full)
    scale = pf.PIXEL_SCALE_ARCSEC
    print("star (full-frame px): %.2f, %.2f" % tuple(star))

    idx_late = tuple(range(2, len(frames)))
    texp = getattr(ssu, "EXPTIMES_S", None)
    w01 = (np.asarray([texp[i] for i in EP_COMBINE], float)
           if texp is not None and len(texp) > max(EP_COMBINE) else None)
    if w01 is not None:
        print("EP0+EP1 exposure weights: "
              + "/".join(f"{v:.3f}" for v in w01 / w01.sum())
              + " (" + "/".join(f"{v:g} s" for v in w01) + ")")
    imgs01 = verified_members(frames, EP_COMBINE, res_xy)
    aq01 = stack_mean(imgs01, w01)          # displayed AQ combination
    aq01_line = cross_median(imgs01)        # robust resample, profile only
    aq_l8 = combine_epochs(frames, idx_late, res_xy, method="median")
    rej = minus_model(frames[rep_epoch], disk)
    u_maj, pa_sky = major_axis_dir(disk, star, pa_major, sx, sy)
    print(f"major-axis cut: pixel direction ({u_maj[0]:+.3f}, "
          f"{u_maj[1]:+.3f}), sky PA = {pa_sky:.1f} deg E of N"
          + ("" if pa_major is None else " (fixed)"))

    # ---- cross-sections: galaxy masked, Gaussian-kernel sampled AQ ----
    from astropy.time import Time

    def _track_r_px(date):
        """Stellocentric radius [px] of the galaxy sight-line at ``date``
        (reflex track anchored at the EP0 position, as in figure 2)."""
        d_ra, d_dec = pf.galaxy_track_mas(Time([date]), pf.EPOCH_DATES[0])
        p = (q_full[0] - star) + np.array(
            [sx * float(d_ra[0]), sy * float(d_dec[0])]) / (scale * 1000.0)
        return float(np.hypot(*p))

    jwst_epochs = getattr(pf, "JWST_EPOCHS",
                          (("2023-06-15", "MIRI"),
                           ("2024-02-03", "MIRI+NIRCam")))
    r_acs = _track_r_px("2004-06-01")
    r_jw = [_track_r_px(d) for d, _ in jwst_epochs]
    print(f"sight-line radii: HST ACS 2004 at {r_acs:.1f} px "
          f"({r_acs * scale:.2f}\"); JWST at "
          + ", ".join(f"{r:.1f} px ({d})" for (d, _), r
                      in zip(jwst_epochs, r_jw)))
    r_gal = np.hypot(*(q_full - star).T)
    r_hi = max(180.0, r_gal.max() + 6.0, r_acs + 8.0, max(r_jw) + 8.0)
    radii = np.arange(1.0, r_hi + 0.5, 1.0)
    aq_line_img = nan_gaussian(mask_galaxies(aq01_line, q_full,
                                             EP_COMBINE), prof_sigma)
    l8_line_img = nan_gaussian(mask_galaxies(aq_l8, q_full, idx_late),
                               prof_sigma)
    prof_aq, _ = cross_section(aq_line_img, star, u_maj, radii, half_arc)
    prof_l8, _ = cross_section(l8_line_img, star, u_maj, radii, half_arc)
    prof_md, md_sides = cross_section(np.nan_to_num(disk), star, u_maj,
                                      radii, half_arc)
    r_pk, pk, r_in, r_out = _peak_and_hwhm(radii, prof_md)
    print("model peak by cut side (+u / -u): "
          + " / ".join(f"{np.nanmax(sd):.4f}" for sd in md_sides)
          + " counts/s/px (two-side average plotted; the original GS03 "
            "peak tells which convention it used)")
    print(f"model cross-section: peak {pk:.3f} counts/s/px at r = "
          f"{r_pk:.1f} px ({r_pk * scale:.2f}\"); HWHM-in {r_in:.1f} px "
          f"({r_in * scale:.2f}\"), HWHM-out {r_out:.1f} px "
          f"({r_out * scale:.2f}\"), FWHM {r_out - r_in:.1f} px "
          f"({(r_out - r_in) * scale:.2f}\")")
    lab01 = f"AQ image"
    lab_l8 = f"median image"
    for name, p, extra in ((lab01, prof_aq,
                            "5-px-cross median resample of both epochs; "),
                           (lab_l8, prof_l8, "")):
        roi = (radii >= R_AGREE_PX) & (radii <= 180.0) & np.isfinite(p)
        dev = p[roi] - prof_md[roi]
        note = (extra + f"tangential-median cut, +/-{half_arc:g} px arc"
                + (f", pre-smoothed sigma = {prof_sigma:g} px"
                   if prof_sigma > 0 else ""))
        print(f"{name} - model in r = {R_AGREE_PX:.0f}..180 px: mean "
              f"{dev.mean():+.1e}, rms {dev.std():.1e} counts/s/px "
              f"({note}; sigma_rms ~ "
              f"{getattr(ssu, 'SIGMA_BG', 3e-3):.0e})")
    print("galaxy sight-line radii EP0-EP7 [px]:", np.round(r_gal, 1))
    if diagnose:
        _diagnose_model_agreement(out_dir, disk, star, u_maj, radii,
                                  half_arc, aq_line_img, l8_line_img,
                                  prof_aq, prof_l8, prof_md, frames,
                                  idx_late, res_xy, q_full)

    # ---- layout: horizontal colorbar over three panels + cross-section ----
    pf.apply_style()
    fig = plt.figure(figsize=(7.35, 2.62))
    gs = fig.add_gridspec(2, 5, height_ratios=[0.050, 1.0],
                          width_ratios=[1, 1, 1, 0.42, 1.50],
                          wspace=0.055, hspace=0.04)
    half = int(min(min(min(f.shape) for f in frames) // 2 - 2,
                   max(SUBARRAY // 2, np.abs(q_full - star).max() + 26)))
    vmax = 10.0 ** LOG_RANGE_DEX[1]        # same upper stretch as GS02
    vmin = -vmax * 0.00
    try:                                   # asinh: linear core, compressed
        norm = mcolors.AsinhNorm(linear_width=LIN_WIDTH,
                                 vmin=vmin, vmax=vmax)
    except AttributeError:                 # matplotlib < 3.6
        norm = mcolors.SymLogNorm(LIN_WIDTH, vmin=vmin, vmax=vmax, base=10)
    axs_img, im0, share = [], None, None
    panels = ((aq01, f"AQ image (EP{EP_COMBINE[0]}+EP{EP_COMBINE[1]})"),
              (disk, "disk model"),
              (rej, f"EP{rep_epoch} $-$ disk model"))
    for k, (img, label) in enumerate(panels):
        ax = fig.add_subplot(gs[1, k], sharex=share, sharey=share)
        share = share or ax
        cut, ext = crop(img, star, half)
        im = ax.imshow(np.ma.masked_invalid(cut), origin="lower",
                       cmap=CM_LIN, norm=norm, extent=ext,
                       interpolation="nearest")
        im0 = im0 or im
        star_marker(ax, size=42)
        ax.text(0.035, 0.965, label, transform=ax.transAxes, ha="left",
                va="top", fontsize=6.6, color="white", zorder=7,
                path_effects=STROKE)
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel(r"$\Delta x$ [arcsec]", fontsize=5.6)
        if k == 0:
            ax.set_ylabel(r"$\Delta y$ [arcsec]", fontsize=5.6)
        else:
            ax.tick_params(labelleft=False)
        axs_img.append(ax)
    ax_a, ax_b, ax_c = axs_img
    lim = half * scale
    ax_a.set_xlim(-lim, lim)
    ax_a.set_ylim(-lim, lim)
    compass(ax_c, sx, sy, pad_top=0.155, pad_right=0.105)
    scale_bar(ax_c, au=100.0)

    # panel (b) overlays: major-axis cut + galaxy track (lines only)
    c_track = pf.palette(1, 0.04, 0.04)[0]
    ends = np.outer([-1.15, 1.15], u_maj) * lim
    ax_b.plot(ends[:, 0], ends[:, 1], color="white", ls=(0, (4, 2.4)),
              lw=0.9, alpha=0.95, zorder=4)
    track = galaxy_track_arcsec(q_full, star, sx, sy, t_span)
    ax_b.plot(track[:, 0], track[:, 1], color="white", lw=2.2, zorder=4)
    ax_b.plot(track[:, 0], track[:, 1], color=c_track, lw=1.0, zorder=5)

    cax = fig.add_subplot(gs[0, 0:3])
    cb = fig.colorbar(im0, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("top")
    # tks = (vmin, -0.05, -LIN_WIDTH, 0.0, LIN_WIDTH, 0.05, vmax)
    tks = (vmin, 0.0, LIN_WIDTH, 0.02, 0.05, 0.1, vmax)
    cb.set_ticks(tks)
    cb.set_ticklabels([f"{v:.2g}" for v in tks])
    cb.ax.tick_params(labelsize=5.6, pad=1.5)
    cb.ax.minorticks_off()
    cax.text(1.012, 0.45, r"counts s$^{-1}$ px$^{-1}$",
             transform=cax.transAxes, ha="left", va="center", fontsize=5.6)

    # ---- cross-section panel ----
    ax_d = fig.add_subplot(gs[1, 4])
    c_aq, c_l8, c_md = pf.palette(3, 0.20, 0.85)
    c_band = pf.palette(1, 0.55, 0.55)[0]
    show = radii >= 43.0
    gmax = np.nanmax([pk, np.nanmax(prof_aq[show]),
                      np.nanmax(prof_l8[show])])
    r_leg = radii[show][0] + 0.40 * (radii[-1] - radii[show][0])
    leg = radii >= r_leg                    # x-span under the legend box
    m_leg = np.nanmax([np.nanmax(np.where(leg, prof_aq, np.nan)),
                       np.nanmax(np.where(leg, prof_l8, np.nan)),
                       np.nanmax(np.where(leg, prof_md, np.nan))])
    y_top = 1.05 * gmax  # legend clears every curve
    ax_d.axvspan(radii[show][0], R_AGREE_PX, color="0.55", alpha=0.14,
                 lw=0, zorder=1)
    # ax_d.axvline(R_AGREE_PX, color="0.45", lw=0.7, ls=":", zorder=2)
    ax_d.axvspan(r_gal.min(), r_gal.max(), color=c_band, alpha=0.18,
                 lw=0, zorder=1)
    ax_d.plot(radii[show], prof_aq[show], color=c_aq, lw=0.7, ls="-",
              label=lab01, zorder=4)
    ax_d.plot(radii[show], prof_l8[show], color=c_l8, lw=0.7,
              ls=(0, (1.4, 1.6)), label=lab_l8, zorder=4)
    ax_d.plot(radii[show], prof_md[show], color=c_md, lw=0.7, ls="-.",
              label="disk model", zorder=5)
    # ACS 2004 + JWST sight-lines, figure-2 marker conventions, no legend
    c_acs = pf.palette(3, 0.2, 0.95)[2]
    c_jwst = "#FFC838"
    ax_d.axvline(r_acs, color=c_acs, lw=0.9, zorder=3)
    ax_d.scatter([r_acs], [0.5 * y_top], marker="s", s=26,
                 facecolor="none", edgecolor=c_acs, lw=1.1, zorder=6)
    for r_j in r_jw:
        ax_d.axvline(r_j, color=c_jwst, lw=0.9, zorder=3)
        ax_d.scatter([r_j], [0.5 * y_top], marker="h", s=14,
                     facecolor=c_jwst, edgecolor="black", lw=0.6, zorder=6)
    ax_d.annotate("galaxy\nsight-lines\n(EP0\u2013EP7)",
                  (0.5 * (r_gal.min() + min(r_gal.max(), radii[-1])),
                   0.030 * y_top),
                  ha="center", va="bottom", fontsize=5.6, color="0.35")
    ax_d.legend(loc="upper right", fontsize=5.8, framealpha=0.9,
                edgecolor="0.4", handlelength=2.3, borderaxespad=0.5)
    ax_d.set_ylim(0.0, y_top)
    ax_d.set_xlim(radii[show][0], radii[-1])
    ax_d.set_xlabel("radius [px]", fontsize=5.6)
    ax_d.set_ylabel(r"counts s$^{-1}$ px$^{-1}$", fontsize=5.6,
                    labelpad=2.0)
    ax_d.tick_params(labelsize=6.5, top=False)
    sec = ax_d.secondary_xaxis(
        "top", functions=(lambda r: r * scale, lambda x: x / scale))
    sec.set_xlabel("radius [arcsec]", fontsize=5.6, labelpad=2.5)
    sec.tick_params(labelsize=6.5, direction="in")

    # pin colorbar + cross-section panel to the actual (aspect-shrunk)
    # image-panel extents
    fig.draw_without_rendering()
    pa = ax_a.get_position()
    pc = ax_c.get_position()
    cax.set_position([pa.x0, pa.y1 + 0.030, pc.x1 - pa.x0, 0.030])
    pd = ax_d.get_position()
    ax_d.set_position([pd.x0, pa.y0, pd.x1 - pd.x0, pa.y1 - pa.y0])
    pf.save(fig, out_dir, "GS03_GS04_disk_model")


# --------------------------------------------------------------------------- #
# diskalma: all-epoch galaxy-free stack + ARKS ALMA band 6+7 (1.0 mm) contours
# --------------------------------------------------------------------------- #
def _resolve_alma(alma):
    """Path of the contour extraction: ``--alma`` may be the file, its
    directory, or omitted (then ``literature_data/arks_alma/extracted``
    is searched next to the script, one level up, and under the CWD);
    ``none`` switches the overlay off."""
    if alma is not None and str(alma).strip().lower() in ("none", "off"):
        return None
    here = Path(__file__).resolve().parent
    cands = ([Path(alma)] if alma else []) \
        + [base / ALMA_SUBDIR for base in (here, here.parent, Path.cwd())]
    cands = list(dict.fromkeys(c.resolve() for c in cands))
    tried = []
    for c in cands:
        probes = tuple(c / f for f in ALMA_FILES) if c.is_dir() else (c,)
        for p in probes:
            tried.append(p)
            if p.is_file():
                return p
    sys.exit("diskalma: ALMA contour extraction not found; tried:\n  "
             + "\n  ".join(str(t) for t in tried)
             + "\n(pass --alma FILE|DIR, or --alma none for no overlay)")


def load_alma_contours(path):
    """``[(sigma_level, [(N, 2) polyline arrays]), ...]``, sorted by level,
    in star-centred (dRA, dDec) arcsec (E and N positive) -- the ARKS
    gallery-figure vector paths as extracted.  Prefers the json
    (assembled polylines, proper line joins); the npz fallback carries
    the same geometry as raw 2-point segments."""
    path = Path(path)
    if path.suffix == ".json":
        d = json.loads(path.read_text())
        out = [(int(lay["level_sigma"]),
                [np.asarray(p, float) for p in lay["polylines_arcsec"]])
               for lay in d["layers"]]
    else:
        z = np.load(path)
        seg, lev = z["segments_arcsec"], z["level_sigma"]
        out = []
        for l in np.unique(lev):         # chain the consecutive 2-pt
            s = seg[lev == l]            # segments back into polylines
            polys, cur = [], [s[0][0], s[0][1]]
            for a, b in s[1:]:
                if np.allclose(a, cur[-1], atol=1e-9):
                    cur.append(b)
                else:
                    polys.append(np.asarray(cur))
                    cur = [a, b]
            polys.append(np.asarray(cur))
            out.append((int(l), polys))
    return sorted(out, key=lambda t: t[0])


def draw_alma_contours(ax, layers, sx, sy, levels=None, lw0=0.3,
                       lw_step=0.4, color="white"):
    """ARKS contour polylines on the pixel-frame arcsec axes: the same
    (sx, sy) sign mapping (RA, Dec) -> pixel (x, y) as the galaxy track;
    the frames are N-up, so no rotation.  Plain white lines whose width
    grows with the sigma level (lw0 + i * lw_step for the i-th drawn
    level, ascending), so the levels stay tellable apart without a
    legend per line.  Returns the sigma levels actually drawn."""
    from matplotlib.collections import LineCollection

    drawn = []
    for lev, polys in layers:
        if levels is not None and lev not in levels:
            continue
        segs = [np.column_stack([sx * np.asarray(p)[:, 0],
                                 sy * np.asarray(p)[:, 1]]) for p in polys]
        ax.add_collection(LineCollection(
            segs, colors=color, linewidths=lw0 * (1 + (len(drawn) + 0.25) ** 2.00) * lw_step,
            capstyle="round", joinstyle="round", alpha=0.95, zorder=5))
        drawn.append(lev)
    return drawn


def _beam_and_levels(ax, levels_drawn, beam=ALMA_BEAM_FWHM_ARCSEC):
    """ALMA beam (hatched, to scale) + level annotation, lower left --
    the gallery panel's own convention."""
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    x0 = ax.get_xlim()[0] + 0.055 * span
    y0 = ax.get_ylim()[0] + 0.055 * span
    y_txt = y0
    if beam and beam > 0.0:
        ax.add_patch(Ellipse((x0 + beam / 2.0, y0 + beam / 2.0), beam,
                             beam, facecolor="none", edgecolor="white",
                             lw=0.6, hatch="//////", zorder=6))
        y_txt = y0 + beam / 2.0
        x0 += beam + 0.030 * span
    lv = ",\u2009".join(f"{l:g}" for l in levels_drawn)
    ax.text(x0, y_txt, ALMA_LABEL + "\n" + rf"$[{lv}]\,\sigma$",
            color="white", fontsize=6.0, ha="left", va="center",
            linespacing=1.25, zorder=6, path_effects=STROKE)


def _stack_stats(frames, imgs, comb, nmap, star, q_full, margin, aspect,
                 how):
    """Caption numbers of the stack: coverage inside the union of the
    (margin-enlarged, aspect-stretched) galaxy zones, galaxy peak before
    vs after, and the background rms of the combined image."""
    a, b, th = APERTURE_AB_THETA
    yy, xx = np.mgrid[0:comb.shape[0], 0:comb.shape[1]]
    zone = np.zeros(comb.shape, bool)
    for q in q_full:
        zone |= in_ellipse(xx, yy, q, (a + margin) * aspect, b + margin,
                           th, margin=0.0)
    g_before = []
    for f, q in zip(frames, q_full):
        gy, gx = np.mgrid[0:f.shape[0], 0:f.shape[1]]
        g_before.append(np.nanmax(np.where(
            in_ellipse(gx, gy, q, a, b, th), f, np.nan)))
    k = nmap[zone]
    r = np.hypot(xx - star[0], yy - star[1])
    bg = (r > min(190.0, 0.45 * min(comb.shape))) & ~zone
    print(f"stack: {how} of {len(imgs)} epochs; per-epoch galaxy ellipse "
          f"(a, b, th) = {APERTURE_AB_THETA} + {margin:g} px margin, "
          f"major axis x {aspect:g}, masked first")
    print(f"galaxy-zone refill: masked pixels stay covered by "
          f"{int(k.min())}..{int(k.max())} of {len(imgs)} epochs "
          f"(mean {k.mean():.1f})")
    if k.min() == 0:
        print(f"WARNING: {int((k == 0).sum())} px inside the union zone "
              f"are masked in every epoch (NaN hole in the stack) -- "
              f"lower --gal-margin / --gal-aspect")
    print(f"galaxy-zone peak: {min(g_before):.2f}..{max(g_before):.2f} "
          f"counts/s/px per epoch -> "
          f"{np.nanmax(np.where(zone, comb, np.nan)):.3f} in the stack; "
          f"background sigma_rms = "
          f"{np.nanstd(np.where(bg, comb, np.nan)):.1e} counts/s/px")


def fig_disk_alma(module_path, out_dir, log_dir=None, alma=None,
                  combine="median", gal_margin=GAL_MARGIN_STACK,
                  gal_aspect=GAL_ASPECT_STACK, levels=None):
    """One galaxy-free disk image from all epochs, ALMA contours on top.

    Each epoch is masked with the pipeline galaxy ellipse (enlarged by
    ``gal_margin``, major axis stretched by ``gal_aspect``) at its own
    *measured* position on the raw frame, i.e. before verified_members,
    so a corrected integer slip moves the mask together with the data.
    The masked frames are then valid-pixel median combined (default):
    the galaxy sits somewhere else in every other epoch group, so the
    masked zones refill, and the median also rejects the halo wings
    beyond the mask, the epoch-specific radial PSF-residual streaks,
    and hot pixels.  ``combine='mean'`` gives the exposure-weighted
    mean instead (deeper, but keeps every residual).  Visualization
    only -- the effective depth varies across the galaxy corridor --
    not a science product."""
    layers, alma_path = None, _resolve_alma(alma)
    if alma_path is not None:
        layers = load_alma_contours(alma_path)
        print(f"ALMA contours: {alma_path} ("
              + ", ".join(str(l) for l, _ in layers) + " sigma; "
              + str(sum(len(p) for _, p in layers)) + " polylines)")

    ssu = pf.import_module(module_path)
    frames = load_frames(ssu)
    disk = pf.to_np(ssu.load_disk_model()).astype(float)
    star = np.array(pf.find_star_center(disk))
    q_full = epoch_positions(ssu, log_dir)
    sx, sy, res_xy = sign_convention(q_full)
    n = len(frames)
    print("star (full-frame px): %.2f, %.2f" % tuple(star))

    masked = [mask_galaxies(f, q_full, [i], margin=gal_margin,
                            aspect=gal_aspect)
              for i, f in enumerate(frames)]
    imgs = verified_members(masked, tuple(range(n)), res_xy)
    stack = np.stack(imgs)
    nmap = np.isfinite(stack).sum(0)
    if combine == "median":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            comb = np.where(nmap > 0, np.nanmedian(stack, axis=0), np.nan)
        how = "valid-pixel median"
    else:
        texp = getattr(ssu, "EXPTIMES_S", None)
        w = (np.asarray(texp[:n], float)
             if texp is not None and len(texp) >= n else None)
        comb = stack_mean(imgs, w)
        how = "valid-pixel mean" + ("" if w is None
                                    else " (exposure-weighted)")
    _stack_stats(frames, imgs, comb, nmap, star, q_full, gal_margin,
                 gal_aspect, how)

    # ---- single panel, GS02 stretch, side colorbar ----
    pf.apply_style()
    fig = plt.figure(figsize=(3.55, 3.42))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.055], wspace=0.035)
    ax = fig.add_subplot(gs[0, 0])
    half = int(min(min(min(f.shape) for f in frames) // 2 - 2,
                   SUBARRAY // 2))
    cut, ext = crop(comb, star, half)
    norm = LogNorm(vmin=10.0 ** LOG_RANGE_DEX[0], vmax=DISKALMA_VMAX)
    im = ax.imshow(np.ma.masked_invalid(cut), origin="lower", cmap=CM_LOG,
                   norm=norm, extent=ext, interpolation="nearest")
    lim = half * pf.PIXEL_SCALE_ARCSEC
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks((-5.0, 0.0, 5.0))
    ax.set_yticks((-5.0, 0.0, 5.0))
    ax.tick_params(labelsize=6.5, color="white")
    ax.set_xlabel(r"$\Delta x$ [arcsec]", fontsize=6.0)
    ax.set_ylabel(r"$\Delta y$ [arcsec]", fontsize=6.0)

    drawn = (draw_alma_contours(
        ax, layers, sx, sy,
        ALMA_LEVELS_DEFAULT if levels is None else levels)
        if layers else [])
    star_marker(ax, size=42)
    compass(ax, sx, sy, frac=0.085, pad_right=0.05, pad_top=0.075)
    scale_bar(ax, au=100.0)
    ax.text(0.035, 0.965, f"EP0\u2013EP{n - 1} {combine}, galaxy removed",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.6,
            color="white", zorder=7, path_effects=STROKE)
    if drawn:
        _beam_and_levels(ax, drawn)
        print(f"overlay: levels [" + ", ".join(map(str, drawn))
              + f"] sigma; beam {ALMA_BEAM_FWHM_ARCSEC:.2f}\" FWHM "
              f"(circular, hatched); (dRA, dDec) -> (sx, sy) = "
              f"({sx:+d}, {sy:+d}) pixel mapping")
    print(f"display: log10 stretch [{LOG_RANGE_DEX[0]:g}, "
          f"{np.log10(DISKALMA_VMAX):g}] dex counts/s/px "
          f"(1 count/s/px = 177 uJy/arcsec^2 = 18.04 Vmag/arcsec^2)")

    cax = fig.add_subplot(gs[0, 1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"counts s$^{-1}$ px$^{-1}$", fontsize=6.0)
    tks = tuple(v for v in (0.01, 0.02, 0.05, 0.1, 0.2)
                if v <= DISKALMA_VMAX * (1 + 1e-9))
    cb.set_ticks(tks)
    cb.set_ticklabels([f"{v:g}" for v in tks])
    cb.ax.tick_params(labelsize=6.5)
    cb.ax.minorticks_off()

    fig.draw_without_rendering()
    pa = ax.get_position()
    cp = cax.get_position()
    cax.set_position([pa.x1 + 0.010, pa.y0, cp.width, pa.y1 - pa.y0])
    pf.save(fig, out_dir, "disk_combined_ALMA")


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_dir", nargs="?", default=None,
                    help="optional run directory; the posterior-median "
                         "offsets then refine the epoch positions")
    ap.add_argument("--module", default="self_sufficient_ultranest.py",
                    help="path to self_sufficient_ultranest.py (data access)")
    ap.add_argument("--figs", nargs="+", default=["gs02", "gs0304"],
                    choices=["gs02", "gs0304", "diskalma"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--alma", default=None, metavar="FILE|DIR",
                    help="diskalma: ARKS contour extraction "
                         "(hd107146_contours.json/.npz or its directory; "
                         "default: search literature_data/arks_alma/"
                         "extracted next to the script, one level up, "
                         "and under the CWD); 'none' disables the "
                         "overlay")
    ap.add_argument("--combine", choices=("median", "mean"), default=None,
                    help="diskalma: all-epoch combination (default "
                         "median, which rejects the galaxy halo wings, "
                         "streaks and hot pixels; mean is "
                         "exposure-weighted)")
    ap.add_argument("--gal-margin", type=float, default=None,
                    help="diskalma: extra enlargement [px] of the "
                         "per-epoch galaxy mask ellipse (default "
                         f"{GAL_MARGIN_STACK:g})")
    ap.add_argument("--gal-aspect", type=float, default=None,
                    help="diskalma: stretch of the mask's margined "
                         "major axis, i.e. the factor on its axis "
                         f"ratio (default {GAL_ASPECT_STACK:g})")
    ap.add_argument("--alma-levels", nargs="+", type=int, default=None,
                    metavar="SIGMA",
                    help="diskalma: contour sigma levels to draw "
                         "(default: "
                         + " ".join(map(str, ALMA_LEVELS_DEFAULT))
                         + "; the file also carries 30)")
    ap.add_argument("--rep-epoch", type=int, default=2,
                    help="representative epoch of the rejection panel (c)")
    ap.add_argument("--pa-major", type=float, default=None,
                    help="fix the cut's sky PA [deg E of N] instead of the "
                         "flux-weighted principal axis of the model")
    ap.add_argument("--track-span", nargs=2, type=float,
                    default=(2004.0, 2028.0), metavar=("T0", "T1"),
                    help="jyear span of the galaxy track in the model panel")
    ap.add_argument("--arc-halfwidth", type=float, default=ARC_HALF_PX,
                    help="half-length [px] of the constant-radius arc "
                         "whose median forms each cut sample; rejects hot "
                         "pixels and the radial PSF-residual streaks with "
                         "no radial broadening (0 = plain per-pixel cut "
                         "as in the original GS03)")
    ap.add_argument("--diagnose", action="store_true",
                    help="print data/model agreement diagnostics (per "
                         "side, per band, minor axis, rejection "
                         "residuals) and save GS03_diagnostics")
    ap.add_argument("--profile-sigma", type=float,
                    default=PROFILE_SIGMA_PX,
                    help="Gaussian kernel sigma [px] of the AQ cut "
                         "sampling; default 0 = exact per-pixel bilinear "
                         "samples along the cut, as in the original GS03 "
                         "(a 2-D kernel mixes in the flanking radial "
                         "PSF-residual streaks and smears point artifacts)")
    args = ap.parse_args(argv)

    dka = (args.alma, args.combine, args.gal_margin, args.gal_aspect,
           args.alma_levels)
    if any(o is not None for o in dka) and "diskalma" not in args.figs:
        print("note: diskalma option(s) given; adding 'diskalma' to --figs")
        args.figs = list(args.figs) + ["diskalma"]
    combine = args.combine if args.combine is not None else "median"
    gal_margin = (args.gal_margin if args.gal_margin is not None
                  else GAL_MARGIN_STACK)
    gal_aspect = (args.gal_aspect if args.gal_aspect is not None
                  else GAL_ASPECT_STACK)

    module = Path(args.module)
    if not module.exists():
        alt = Path(__file__).resolve().parent / module.name
        if alt.exists():
            module = alt
        else:
            sys.exit(f"--module {args.module} not found")
    out = (Path(args.out_dir) if args.out_dir
           else (Path(args.log_dir) / "paper" / "figures" if args.log_dir
                 else Path("figures")))

    for f in args.figs:
        if f == "gs02":
            fig_gs02(str(module), out, log_dir=args.log_dir)
        elif f == "diskalma":
            fig_disk_alma(str(module), out, log_dir=args.log_dir,
                          alma=args.alma, combine=combine,
                          gal_margin=gal_margin, gal_aspect=gal_aspect,
                          levels=(tuple(args.alma_levels)
                                  if args.alma_levels else None))
        else:
            fig_gs0304(str(module), out, log_dir=args.log_dir,
                       rep_epoch=args.rep_epoch, pa_major=args.pa_major,
                       t_span=tuple(args.track_span),
                       prof_sigma=args.profile_sigma,
                       half_arc=args.arc_halfwidth,
                       diagnose=args.diagnose)
    return 0


if __name__ == "__main__":
    sys.exit(main())