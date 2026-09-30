#!/usr/bin/env python3
"""Publication figures for the HD 107146 transit-photometry paper.

All figures share one color scheme (``cmasher`` *bubblegum*; the diverging
residual maps use the visually matching *waterlily*).  Existing libraries do
the heavy lifting: UltraNest/corner for the posterior, astropy for the
parallax + proper-motion track, and the analysis module itself
(``self_sufficient_ultranest.py``) for everything that touches data.

Figures
-------
1  posterior      corner plot of the nested-sampling posterior (by default
                  the offsets and tau'; the per-epoch noise scales of a v07
                  run are excluded unless requested via --params, e.g.
                  ``--params noise tau`` or ``--params all``)
2  disk-track     disk model with epoch positions and the 2004-2028 on-sky
                  galaxy track (stellar reflex: proper motion + parallax)
3  residuals      per-epoch standardized residuals, 2 x 4 grid, shared colorbar
                  (standardized with the fitted noise scales for a v07 run)
4  spectrum       eigenvalue spectrum of the template normal matrix H
5  comparison     surface-density comparison figure (van Sluijs fig. 13
                  axes) built from the original extractions: Ardila 2004
                  fig. 3 omega*tau (converted with our grain assumptions;
                  albedo via --albedo), Ricci 2015 and Han 2026 (ARKS)
                  surface densities as published (mm-emission opacities),
                  the van Sluijs 2018 point (rescaled to our grain
                  density), and our per-epoch upper limits
6  noise-scales   per-epoch noise-scale posteriors f_i of a v07 run against
                  the square root of the nominal per-epoch MSR (from
                  ``<log_dir>/paper/paper_values.json`` when present)

Parameter layout: v07 runs sample (offsets, ln f_0..ln f_{N-1}, tau'), v06
runs (offsets, tau').  Every index into theta goes through
``paper_values.param_layout`` (the run's paramnames), never through
``[:-1]`` arithmetic, so both kinds of run directory work.

Usage
-----
    python paper_figures.py <log_dir> --figs 1 5 6 --lit-dir literature_data
    python paper_figures.py <log_dir> --figs 2 3 4 --module path/to/self_sufficient_ultranest.py

Figures 1 and 6 need only the run directory; figure 5 needs the literature
CSV directory and ``<log_dir>/paper/comparison_values.json`` (written by
``paper_values.py``, with ``--evaluate`` for the per-epoch points); figures
2-4 need ``--module`` and the data paths inside it to resolve.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
import cmasher as cmr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from paper_values import param_layout, offsets_from_theta  # noqa: E402
except ImportError as exc:                                     # pragma: no cover
    sys.exit(f"paper_values.py must sit next to this script ({exc})")

# --------------------------------------------------------------------------- #
# shared style
# --------------------------------------------------------------------------- #
CMAP_SEQ = plt.get_cmap("cmr.bubblegum")        # images, sequential data
CMAP_DIV = plt.get_cmap("cmr.viola")            # residual maps (diverging)

#: HD 107146 astrometry.  Parallax / proper motion follow constants.py of the
#: pipeline (Gaia DR3); RA/Dec (SIMBAD, ICRS J2000) enter only through the
#: parallax factors, which depend on them weakly.
STAR_RA_DEG = 184.77709
STAR_DEC_DEG = 16.54830
STAR_PM_MASYR = (-174.7, -149.0)      # (pmRA*, pmDec)
STAR_PARALLAX_MAS = 36.404
PIXEL_SCALE_ARCSEC = 0.05072

EPOCH_DATES = ("2011-02-22", "2011-05-03", "2017-03-26", "2017-07-28",
               "2018-03-16", "2018-07-29", "2018-12-20", "2019-05-28")


def palette(n, lo=0.12, hi=0.88):
    return cmr.take_cmap_colors("cmr.bubblegum", n, cmap_range=(lo, hi))


def apply_style():
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "savefig.dpi": 300, "pdf.fonttype": 42,
        "axes.prop_cycle": plt.cycler(color=palette(6)),
    })


def save(fig, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"wrote {out_dir}/{name}.pdf (+.png)")


def tex_label(name):
    """Raw pipeline names (dx_1, dy_3, lnf_2, tau) -> TeX.  Names that are
    already TeX (contain '$') or that do not match the raw pattern pass
    through unchanged, so runs whose paramnames were renamed to TeX keep
    working."""
    if "$" in name:
        return name
    if name == "tau":
        return r"$\tau^{\prime}$"
    try:
        kind, idx = name.split("_", 1)
        sym = {"dx": r"\delta x", "dy": r"\delta y", "lnf": r"\ln f"}[kind]
    except (ValueError, KeyError):
        return name
    if kind == "lnf":
        return rf"${sym}_{{{idx}}}$"
    return rf"${sym}_{{{idx}}}$ [px]"


PARAM_GROUPS = ("all", "offsets", "noise", "tau")


def _match_params(names, params):
    """Resolve --params entries against the run's paramnames.  Each entry
    is a group keyword (``all``, ``offsets``, ``noise``, ``tau``), an exact
    name, the raw old-style name (dx_2, lnf_0, tau) mapped through
    tex_label, or a unique substring.  Returns unique indices in the
    order first requested."""
    lay = param_layout(names)
    out = []

    def add(i):
        if i not in out:
            out.append(i)

    for p in params:
        if p in PARAM_GROUPS:
            group = {"all": list(range(len(names))),
                     "offsets": lay["i_off"], "noise": lay["i_lnf"],
                     "tau": [lay["i_tau"]]}[p]
            if not group:
                warnings.warn(f"--params {p}: this run has no such block")
            for i in group:
                add(i)
            continue
        if p in names:
            add(names.index(p))
            continue
        tex = tex_label(p)
        if tex in names:
            add(names.index(tex))
            continue
        hits = [i for i, n in enumerate(names) if p in n]
        if len(hits) == 1:
            add(hits[0])
            continue
        raise SystemExit(f"--params entry {p!r} does not match a parameter; "
                         f"available: {names} or a group in {PARAM_GROUPS}")
    return out


def to_np(x):
    """Torch tensor (any device) or array-like -> numpy array."""
    return (x.detach().cpu().numpy() if hasattr(x, "detach")
            else np.asarray(x))


def import_module(path):
    spec = importlib.util.spec_from_file_location("ssu", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# run-output access
# --------------------------------------------------------------------------- #
def load_info(log_dir):
    return json.loads((Path(log_dir) / "info" / "results.json").read_text())


def load_run(log_dir, max_samples=None, seed=0):
    """results-like dict with paramnames + weighted samples.

    Reads ``chains/weighted_post.txt`` (falls back to the equal-weighted
    file).  ``ultranest.read_file`` is deliberately NOT used: it re-walks
    ``results/points.hdf5`` with a quadratic scan and effectively hangs on
    runs with millions of likelihood calls.

    ``max_samples``: quick-look/test mode -- resample the posterior down to
    at most this many equal-weight points (weight-proportional draw with
    replacement, fixed seed) before plotting.
    """
    log_dir = Path(log_dir)
    info = load_info(log_dir)
    names = info["paramnames"]
    try:
        wp = np.loadtxt(log_dir / "chains" / "weighted_post.txt", skiprows=1)
        pts, wts = wp[:, 2:], wp[:, 0]
    except OSError:
        eq = np.loadtxt(log_dir / "chains" / "equal_weighted_post.txt",
                        skiprows=1)
        pts, wts = eq, np.full(len(eq), 1.0)
    wts = wts / wts.sum()
    if max_samples and len(pts) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pts), size=int(max_samples), replace=True,
                         p=wts)
        pts = pts[idx]
        wts = np.full(len(pts), 1.0 / len(pts))
        print(f"test mode: resampled posterior to {len(pts)} points")
    return {"paramnames": names,
            "weighted_samples": {"points": pts, "weights": wts},
            "samples": pts}, info


def median_theta(info):
    return np.asarray(info["posterior"]["median"], dtype=float)


def run_layout(info):
    """Parameter layout of a run from its paramnames (see paper_values)."""
    return param_layout(info["paramnames"])


def median_offsets(info):
    """(N, 2) posterior-median registration offsets, epoch 0 = gauge zero."""
    return offsets_from_theta(median_theta(info), run_layout(info))


# --------------------------------------------------------------------------- #
# 1) posterior corner plot
# --------------------------------------------------------------------------- #
def fig_posterior(log_dir, out_dir, params=None,
                  max_samples=None):
    from ultranest.plot import cornerplot

    results, info = load_run(log_dir, max_samples=max_samples)
    names = list(results["paramnames"])
    lay = param_layout(names)
    user_params = bool(params)
    if not params and lay["has_noise"]:
        # a 3N-1 dimensional corner is unreadable in print: the default
        # for a v07 run is the registration + science block (the same
        # content as a v06 corner, same file name), the noise scales get
        # their own figure (6) or ``--params noise tau``
        params = ["offsets", "tau"]
    if params:
        keep = _match_params(names, params)
        pts = np.asarray(results["weighted_samples"]["points"])[:, keep]
        results = {"paramnames": [names[i] for i in keep],
                   "weighted_samples": {"points": pts,
                                        "weights": results["weighted_samples"]["weights"]}}
        names = results["paramnames"]

    c_main, c_hi = palette(2, 0.25, 0.80)
    results["paramnames"] = [tex_label(n) for n in names]  # cornerplot passes
    fig = cornerplot(                                      # these as labels
        results,
        color=c_main,
        with_legend=False,
        contour_kwargs={"linestyles": ["-", "-.", ":", "--"],
                        "colors": [c_main, c_main, c_main, c_hi]},
        label_kwargs={"fontsize": 11},
        title_fmt=".3g",
        hist_kwargs={"color": c_main},
        max_n_ticks=3,
    )
    n = len(names)
    fig.set_size_inches(1.15 * n, 1.15 * n)
    import re
    save(fig, out_dir, "posterior_corner" if not user_params else
         "posterior_corner_" + "_".join(
             re.sub(r"[^A-Za-z0-9]+", "", p) or "p" for p in params))


# --------------------------------------------------------------------------- #
# 2) disk model + epoch positions + on-sky galaxy track
# --------------------------------------------------------------------------- #
def _parallax_factors(times, ra_deg, dec_deg):
    from astropy.coordinates import get_body_barycentric
    import astropy.units as u

    ra, dec = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    pos = get_body_barycentric("earth", times)
    x = pos.x.to_value(u.au)
    y = pos.y.to_value(u.au)
    z = pos.z.to_value(u.au)
    f_ra = x * np.sin(ra) - y * np.cos(ra)
    f_dec = (x * np.cos(ra) * np.sin(dec) + y * np.sin(ra) * np.sin(dec)
             - z * np.cos(dec))
    return f_ra, f_dec


def galaxy_track_mas(times, t_ref):
    """Galaxy position relative to the star [mas], zero at ``t_ref``.

    The galaxy is a fixed background source, so its apparent motion relative
    to HD 107146 is the REVERSED stellar motion: proper motion plus the
    annual parallax ellipse (parallax factors from the barycentric Earth
    position, ICRS).  Neglected as far sub-pixel over 2004-2028: the
    galaxy's own proper motion, orbital reflex, differential aberration.
    """
    from astropy.time import Time

    tt = times if hasattr(times, "jyear") else Time(times, format="jyear")
    t0 = Time(t_ref) if isinstance(t_ref, str) else t_ref
    f_ra, f_dec = _parallax_factors(tt, STAR_RA_DEG, STAR_DEC_DEG)
    f_ra0, f_dec0 = _parallax_factors(t0, STAR_RA_DEG, STAR_DEC_DEG)
    dt = tt.jyear - t0.jyear
    d_ra = -(STAR_PM_MASYR[0] * dt + STAR_PARALLAX_MAS * (f_ra - f_ra0))
    d_dec = -(STAR_PM_MASYR[1] * dt + STAR_PARALLAX_MAS * (f_dec - f_dec0))
    return np.asarray(d_ra), np.asarray(d_dec)


def find_star_center(model):
    """Center of 180-degree symmetry of the (azimuthally smooth) disk model:
    peak of the correlation with its own rotation (== self-convolution),
    with parabolic sub-pixel refinement.  Peak index / 2 is the center."""
    m = np.nan_to_num(np.asarray(model, dtype=float))
    m -= np.median(m)
    ny, nx = m.shape
    big = (2 * ny, 2 * nx)
    conv = np.fft.irfft2(np.fft.rfft2(m, big) ** 2, big)
    k0, k1 = np.unravel_index(np.argmax(conv), conv.shape)

    def sub(cm1, c0, cp1):
        d = cm1 - 2 * c0 + cp1
        return 0.0 if d == 0 else 0.5 * (cm1 - cp1) / d

    dy = sub(conv[k0 - 1, k1], conv[k0, k1], conv[k0 + 1, k1])
    dx = sub(conv[k0, k1 - 1], conv[k0, k1], conv[k0, k1 + 1])
    return ((k1 + dx) / 2.0, (k0 + dy) / 2.0)  # (x, y)


def _smooth_upsample(img, factor=8, sigma=0.7):
    """Small-Gaussian smooth + separable bilinear upsample (numpy-only) so
    contours of a coarse pixel grid are smooth curves, not staircases.
    Returns the upsampled image and its sample coordinates in units of the
    ORIGINAL pixel grid."""
    k = max(3, int(3 * sigma)) * 2 + 1
    xk = np.arange(k) - k // 2
    ker = np.exp(-0.5 * (xk / sigma) ** 2)
    ker /= ker.sum()
    sm = np.apply_along_axis(lambda r: np.convolve(r, ker, "same"), 1, img)
    sm = np.apply_along_axis(lambda c: np.convolve(c, ker, "same"), 0, sm)
    n0, n1 = sm.shape
    xo = (np.arange(n1 * factor) + 0.5) / factor - 0.5
    yo = (np.arange(n0 * factor) + 0.5) / factor - 0.5
    tmp = np.stack([np.interp(xo, np.arange(n1), row) for row in sm])
    up = np.stack([np.interp(yo, np.arange(n0), col) for col in tmp.T]).T
    return up, xo, yo


#: JWST observations of the disk to annotate on the track.
JWST_EPOCHS = (("2023-06-15", "MIRI"),
               ("2024-02-03", "MIRI+NIRCam"))


CORNER_ALIGN = {"ul": ("left", "top"), "ur": ("right", "top"),
                "ll": ("left", "bottom"), "lr": ("right", "bottom")}


def _tick_annotation(ax, p, n_dir, label, corner="ll", gap=0.10,
                     length=0.26, color="white", fontsize=7, lw=0.9):
    """Connector exactly along ``n_dir`` (the SAME direction for every
    annotation -> all connectors parallel), starting a small gap off the
    point and ending at the named CORNER of the text box."""
    import matplotlib.patheffects as pe

    n = np.asarray(n_dir, dtype=float)
    n = n / np.hypot(*n)
    a, e = p + gap * n, p + (gap + length) * n
    ax.plot([a[0], e[0]], [a[1], e[1]], color=color, lw=lw, zorder=6,
            solid_capstyle="butt")
    ha, va = CORNER_ALIGN[corner]
    o = e + 0.015 * n
    ax.text(o[0], o[1], label, color=color, fontsize=fontsize, ha=ha, va=va,
            zorder=6,
            path_effects=[pe.withStroke(linewidth=1.6, foreground="0.25")])


def _plot_disk_track(disk, q_full, out_dir, t_span=(2004.0, 2028.0),
                     template=None, template_center=None):
    """Pure-plotting core (numpy in, figure out) so the layout is testable
    without torch/data."""
    import matplotlib.patheffects as pe
    from astropy.time import Time
    from matplotlib.colors import AsinhNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    star = np.array(find_star_center(disk))
    scale = PIXEL_SCALE_ARCSEC
    t_ep = Time(list(EPOCH_DATES))
    r_ep = np.hypot(*(q_full - star).T) * scale
    print("star (full-frame px): %.2f, %.2f" % tuple(star))
    print("epoch radii [arcsec]:", np.round(r_ep, 3))
    print("epoch radii [au]     :", np.round(r_ep * 1000.0 /
                                             STAR_PARALLAX_MAS, 1))

    # sign self-calibration of the sky -> pixel convention (frames are
    # North-up; the East-West x direction is resolved against the data)
    d_ra, d_dec = galaxy_track_mas(t_ep, EPOCH_DATES[0])
    meas = (q_full - q_full[0]) * scale * 1000.0            # mas
    sx, sy = min(((a, b) for a in (1, -1) for b in (1, -1)),
                 key=lambda s: np.hypot(s[0] * d_ra - meas[:, 0],
                                        s[1] * d_dec - meas[:, 1]).mean())
    rms = np.hypot(sx * d_ra - meas[:, 0], sy * d_dec - meas[:, 1]).mean()
    print(f"track sign convention (sx, sy) = ({sx}, {sy}); "
          f"mean |predicted - measured| = {rms / scale / 1000.0:.2f} px")

    to_xy = lambda dra, ddec: np.stack(
        [(q_full[0, 0] - star[0]) * scale + sx * np.atleast_1d(dra) / 1000.0,
         (q_full[0, 1] - star[1]) * scale + sy * np.atleast_1d(ddec) / 1000.0],
        axis=-1)
    tt = Time(np.arange(t_span[0], t_span[1] + 1e-9, 1.0 / 36.0),
              format="jyear")
    track = to_xy(*galaxy_track_mas(tt, EPOCH_DATES[0]))
    ep_xy = (q_full - star) * scale

    def track_point(t_like):
        tq = Time(t_like) if isinstance(t_like, str) else t_like
        return to_xy(*galaxy_track_mas(tq, EPOCH_DATES[0]))[0]

    # one common connector direction: exactly normal to the MEAN motion,
    # so every annotation line is parallel
    u = track[-1] - track[0]
    u = u / np.hypot(*u)
    if u[0] < 0:
        u = -u                                # canonical reading direction
    n_up = np.array([-u[1], u[0]])
    if n_up[1] < 0:
        n_up = -n_up
    n_dn = -n_up
    ang = float(np.degrees(np.arctan2(u[1], u[0])))
    p_bar = track.mean(axis=0)
    stroke = [pe.withStroke(linewidth=1.6, foreground="0.25")]

    apply_style()
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ny, nx = disk.shape
    xg = (np.arange(nx) - star[0]) * scale
    yg = (np.arange(ny) - star[1]) * scale
    dhat = disk / np.nanmax(disk)
    cm = CMAP_SEQ.copy()
    cm.set_bad(cm(0.0))                       # NaN pixels == zero pixels
    im = ax.imshow(np.ma.masked_invalid(dhat), origin="lower", cmap=cm,
                   extent=(xg[0], xg[-1], yg[0], yg[-1]),
                   norm=AsinhNorm(linear_width=0.05, vmin=0.0, vmax=1.0),
                   interpolation="bilinear")
    cs = ax.contour(xg, yg, np.ma.masked_invalid(dhat),
                    levels=[0.1, 0.5, 0.9], colors="white",
                    linewidths=0.5, alpha=0.7)
    cax = make_axes_locatable(ax).append_axes("right", size="4.5%",
                                              pad=0.08)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\hat{D}$")
    tks = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    cb.set_ticks(tks)
    cb.set_ticklabels([f"{v:g}" for v in tks])

    c_track = palette(1, 0.04, 0.04)[0]
    ax.plot(track[:, 0], track[:, 1], color="white", lw=2.6, zorder=3)
    ax.plot(track[:, 0], track[:, 1], color=c_track, lw=1.2, zorder=4,
            label="galaxy track (2004\u20132028)")
    _, c_ep, c_acs = palette(3, 0.2, 0.95)
    ax.scatter(ep_xy[:, 0], ep_xy[:, 1], marker="o", s=14,
               facecolor=c_ep, edgecolor="black", lw=0.6, zorder=5,
               label="HST STIS (EP0\u2013EP7)")
    ax.scatter(*track_point("2004-06-01"), marker="s", s=26,
               facecolor="none", edgecolor=c_acs, lw=1.1, zorder=5,
               label="HST ACS 2004")

    for yr in (2004, 2008, 2012, 2016, 2020, 2024, 2028):
        _tick_annotation(ax, track_point(Time(float(yr), format="jyear")),
                         n_up, str(yr), corner="ll")

    ax.scatter([0], [0], marker="*", s=90, color="white",
               edgecolor="black", lw=0.5, zorder=6)

    pad = 1.3
    ax.set_xlim(track[:, 0].min() - pad, track[:, 0].max() + pad)
    ax.set_ylim(track[:, 1].min() - pad, track[:, 1].max() + pad)

    # fitted-template isophotes drawn AT the 2004 ACS position, marking
    # (must run AFTER the limits: clabel's inline gap is sized in
    # screen units, so at the pre-zoom autoscale it would swallow
    # the whole ovals)
    # where (and how large) the galaxy was then; line width decreases
    # toward lower flux levels
    if template is not None and template_center is not None:
        up, xo, yo = _smooth_upsample(np.asarray(template, dtype=float))
        lv = float(np.nanmax(up))
        if lv > 0:
            p04 = track_point("2004-06-01")
            gxg = (xo - template_center[0]) * scale + p04[0]
            gyg = (yo - template_center[1]) * scale + p04[1]
            cs_g = ax.contour(gxg, gyg, up, levels=[0.1 * lv, 0.5 * lv],
                              colors="white", linewidths=[0.45, 0.8],
                              alpha=0.95, zorder=6)
            man_g = []
            for segs in cs_g.allsegs:
                if any(len(s) for s in segs):
                    allv = np.concatenate([s for s in segs if len(s)])
                    man_g.append(tuple(allv[np.argmin(allv[:, 1])]))
            if man_g:
                labs_g = ax.clabel(cs_g, fmt=lambda v: f"{v / lv:g}",
                                   colors="white", fontsize=5.5,
                                   inline=True, inline_spacing=1,
                                   manual=man_g)
                for lab in labs_g:
                    lab.set_zorder(7)
                    lab.set_color("white")
                    lab.set_alpha(1.0)
                    lab.set_path_effects(stroke)
            cols = np.any(up >= 0.1 * lv, axis=0)
            x_left = (float(np.min(xo[cols] - template_center[0])) * scale
                      + p04[0]) if cols.any() else p04[0] - 0.5
            ax.text(x_left - 0.08, p04[1], r"template $\hat{G}$",
                    color="white", fontsize=6.5, ha="right", va="center",
                    zorder=6, path_effects=stroke)


    # epoch / JWST rail: connectors normal to the mean-motion line, all
    # ENDING at one common distance from it (different lengths); labels
    # rotated parallel to the track.  Multi-line labels are drawn as
    # stacked single lines, so the TOP of every label sits exactly at
    # d_end + text_gap -- no bounding-box measurement involved.
    fig.draw_without_rendering()
    per_px = ((ax.get_xlim()[1] - ax.get_xlim()[0])
              / ax.get_window_extent().width)
    lh = 7.0 * fig.dpi / 72.0 * 1.25 * per_px       # line height [data]
    text_gap = 0.075

    def rail_annotation(q, label, ha, d_end, gap=0.10):
        d = float((q - p_bar) @ n_dn)
        e = q + (d_end - d) * n_dn
        a = q + gap * n_dn
        ax.plot([a[0], e[0]], [a[1], e[1]], color="white", lw=0.9,
                zorder=6, solid_capstyle="butt")
        for k, line in enumerate(label.split("\n")):
            o = e + (text_gap + k * lh) * n_dn
            ax.text(o[0], o[1], line, color="white", fontsize=7, ha=ha,
                    va="top", rotation=ang, rotation_mode="anchor",
                    zorder=6, path_effects=stroke)

    c_jwst = "#FFC838"
    jw = [track_point(date) for date, _ in JWST_EPOCHS]
    rail_pts = [ep_xy[i] for i in (0, 1, 2, 7)] + jw
    d_end = max(float((q - p_bar) @ n_dn) for q in rail_pts) + 0.32
    for i, ha in ((0, "left"), (1, "right"), (2, "left"), (7, "left")):
        rail_annotation(ep_xy[i], f"EP{i}", ha, d_end)
    for pj, label, ha in ((jw[0], "MIRI", "left"),
                          (jw[1], "NIRCam\n+ MIRI", "right")):
        ax.scatter(*pj, marker="h", s=14, facecolor=c_jwst,
                   edgecolor="black", lw=0.6, zorder=6)
        rail_annotation(pj, label, ha, d_end, gap=0.12)
    ax.scatter([], [], marker="h", s=14, facecolor=c_jwst,
               edgecolor="black", lw=0.6, label="JWST")

    # contour labels: placed at the TOP of each contour inside the view,
    # above the other plot elements
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    excl = [(xhi - 1.55, xhi, yhi - 1.95, yhi)]    # compass corner
    inside_excl = lambda pt: any(a <= pt[0] <= b and c <= pt[1] <= d
                                 for a, b, c, d in excl)
    # deterministic: pool the in-view vertices of ALL segments of a level
    # and take the argmax of (x + y) outside the exclusions -- the label
    # lands on the top-right-most point, and for a level broken into
    # disjoint arcs that maximum lies on the top-right-most arc
    man = []
    for segs in cs.allsegs:
        verts = [seg[((seg[:, 0] > xlo + 0.2) & (seg[:, 0] < xhi - 0.2)
                      & (seg[:, 1] > ylo + 0.25) & (seg[:, 1] < yhi - 0.25))]
                 for seg in segs]
        verts = [v for v in verts if len(v)]
        if not verts:
            continue
        allv = np.concatenate(verts)
        for v in allv[np.argsort(allv[:, 0] + allv[:, 1])[::-1]]:
            if not inside_excl(v):
                man.append(tuple(v))
                break
    if man:
        labs = ax.clabel(cs, fmt=lambda v: f"{v:g}", colors="white",
                         fontsize=6, inline=True, inline_spacing=2,
                         manual=man)
        for lab in labs:
            lab.set_zorder(7)
            lab.set_color("white")
            lab.set_alpha(1.0)
            lab.set_path_effects(stroke)

    # compass (top right): thin pointy arrows joined at a common base
    bx = ax.get_xlim()[1] - 0.50 - (0.42 if sx > 0 else 0.0)
    by = ax.get_ylim()[1] - 0.50 - (0.42 if sy > 0 else 0.0)
    for lbl, (dx, dy) in (("E", (sx * 0.42, 0.0)), ("N", (0.0, sy * 0.42))):
        ax.annotate("", xy=(bx + dx, by + dy), xytext=(bx, by), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color="white",
                                    lw=0.7, mutation_scale=7, shrinkA=0,
                                    shrinkB=0, joinstyle="miter",
                                    capstyle="butt"))
        ax.text(bx + 1.30 * dx, by + 1.30 * dy, lbl, color="white",
                fontsize=7.5, ha="center", va="center", zorder=6)

    # projected 20 au scale bar (lower right)
    len_bar = 20.0 * STAR_PARALLAX_MAS / 1000.0         # arcsec
    x1s = ax.get_xlim()[1] - 0.45
    ysb = ax.get_ylim()[0] + 0.45
    ax.plot([x1s - len_bar, x1s], [ysb, ysb], color="white", lw=1.6,
            solid_capstyle="butt", zorder=6)
    ax.text(x1s - len_bar / 2.0, ysb + 0.09, "20 au", color="white",
            fontsize=7, ha="center", va="bottom", zorder=6,
            path_effects=stroke)

    ax.set_xlabel(r"$\Delta x$ [arcsec]")
    ax.set_ylabel(r"$\Delta y$ [arcsec]")
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)

    save(fig, out_dir, "disk_galaxy_track")


def fig_disk_track(module_path, log_dir, out_dir):
    import torch

    ssu = import_module(module_path)
    disk = to_np(ssu.load_disk_model())
    pos = ssu.load_position_guess()
    q_full = np.stack([pos["x_est"].to_numpy(),
                       pos["y_est"].to_numpy()], axis=1)[: len(EPOCH_DATES)]
    info = load_info(log_dir)
    med = median_theta(info)
    q_full = q_full + median_offsets(info)

    template = center = None
    try:
        obs, disks, masks, var, nominal, q_cut = ssu.load_native_data()
        ll = ssu.LogLikelihood(obs, disks, masks, var, nominal,
                               noise_scale=run_layout(info)["has_noise"])
        theta = torch.as_tensor(med, dtype=torch.float64).view(1, -1)
        g = to_np(ll.template_mean(theta)[0])   # f64 solve, fitted scales
        g = np.where(to_np((ll.w > 0).any(0)).astype(bool), g, 0.0)
        hwt = g.shape[-1]                     # despike isolated hot pixels
        padded = np.pad(g, 1, mode="edge")
        nb = np.stack([padded[dy:dy + hwt, dx:dx + hwt]
                       for dy in (0, 1, 2) for dx in (0, 1, 2)
                       if not (dy == 1 and dx == 1)])
        nmed = np.median(nb, axis=0)
        g = np.where(g > 4.0 * np.maximum(nmed, 1e-12), nmed, g)
        template, center = g, np.asarray(q_cut[0], dtype=float)
    except Exception as exc:                          # noqa: BLE001
        warnings.warn(f"template extent contours skipped: {exc}")
    _plot_disk_track(disk, q_full, out_dir, template=template,
                     template_center=center)


# --------------------------------------------------------------------------- #
# 3) per-epoch standardized residual grid (+ template panel)
# --------------------------------------------------------------------------- #
def _plot_residuals(obs, model, chi, g, q_cut, offsets, out_dir,
                    support=None, scaled=False):
    """3-row x N-epoch grid: RAW observation cutouts (disk included), the
    FULL forward model t S ghat + D, and the standardized residual (which
    is identical with or without the disk, since it cancels in y - m);
    template at the left spanning all rows.  Template/observation/model
    share one colorbar; the residual row keeps its own.  Colorbars live
    in a dedicated gridspec column so the panels stay aligned.  ``scaled``
    only changes the residual colorbar label to the v07 normalization
    (y - m) sqrt(w) / f."""
    n, hw = chi.shape[0], chi.shape[-1]
    centers = np.asarray(q_cut) + np.asarray(offsets)
    lim = hw / 2.0
    vres = max(3.0, np.percentile(np.abs(chi), 99.9))
    stack = np.concatenate([obs.ravel(), model.ravel(),
                            np.asarray(g).ravel()])
    vmax = np.nanpercentile(stack, 99.7)
    vmin = 0.0

    # template diagnostics: common absolute-position error of the initial
    # guesses shows up as an offset of the template from the origin.
    # Isolated hot pixels (value > 4x their 8-neighbour median) are
    # replaced by that median for the DIAGNOSTIC only, so neither the
    # threshold normalization nor the centroid is biased by them.
    xr = np.arange(hw) - q_cut[0][0]
    yr = np.arange(hw) - q_cut[0][1]
    gs_ = np.where(support, g, 0.0) if support is not None else np.asarray(g)
    padded = np.pad(gs_, 1, mode="edge")
    nb = np.stack([padded[dy:dy + hw, dx:dx + hw]
                   for dy in (0, 1, 2) for dx in (0, 1, 2)
                   if not (dy == 1 and dx == 1)])
    med = np.median(nb, axis=0)
    spikes = gs_ > 4.0 * np.maximum(med, 1e-12)
    gs_c = np.where(spikes, med, gs_)
    if spikes.any():
        ks = np.argwhere(spikes)
        print(f"despiked {spikes.sum()} isolated hot template pixel(s) at "
              + ", ".join(f"({xr[c]:+.1f}, {yr[r]:+.1f})" for r, c in ks))
    core = gs_c * (gs_c > 0.25 * gs_c.max())
    cx_t = float((core * xr[None, :]).sum() / core.sum())
    cy_t = float((core * yr[:, None]).sum() / core.sum())
    k = np.unravel_index(np.argmax(gs_c), gs_c.shape)
    print(f"template centroid offset from origin: "
          f"({cx_t:+.2f}, {cy_t:+.2f}) px; peak at "
          f"({xr[k[1]]:+.1f}, {yr[k[0]]:+.1f}) px "
          f"-- a COMMON error of the initial position guesses (gauge; "
          f"apply it to galaxy_positions.csv to recenter masks/sightlines)")

    apply_style()
    fig = plt.figure(figsize=(0.98 * (n + 3.3) + 1.1, 3.35))
    gs = fig.add_gridspec(3, n + 2,
                          width_ratios=[3.25] + [1.0] * n + [0.10],
                          wspace=0.08, hspace=0.08)
    ax_t = fig.add_subplot(gs[:, 0])
    axes, share = [[None] * n for _ in range(3)], None
    for r in range(3):
        for c in range(n):
            ax = fig.add_subplot(gs[r, 1 + c], sharex=share, sharey=share)
            share = share or ax
            axes[r][c] = ax
    cax_img = fig.add_subplot(gs[0:2, -1])
    cax_res = fig.add_subplot(gs[2, -1])

    cm_t = CMAP_SEQ.copy()
    cm_t.set_bad("white")     # unconstrained / NaN template pixels -> white
    cm_zero = CMAP_SEQ.copy()
    cm_zero.set_bad(CMAP_SEQ(0.0))   # observation row: NaN -> zero color
    str_zero = CMAP_SEQ(0.0)
    g_show = np.ma.masked_invalid(np.where(support, g, np.nan)) \
        if support is not None else np.ma.masked_invalid(g)
    ax_t.set_facecolor("white")
    im_img = ax_t.imshow(g_show, origin="lower", cmap=cm_t,
                         vmin=vmin, vmax=vmax,
                         extent=(xr[0] - 0.5, xr[-1] + 0.5,
                                 yr[0] - 0.5, yr[-1] + 0.5),
                         interpolation="nearest")
    ax_t.plot(0, 0, "+", color="white", ms=8, mew=1.1)
    ax_t.add_patch(Ellipse((0.0, 0.0), 2 * (9 + 1.5), 2 * (5 + 1.5),
                           angle=35, fill=False, ls="--", lw=0.9,
                           ec="white"))
    ax_t.text(0.04, 0.965, r"template $\hat{G}$",
              transform=ax_t.transAxes, ha="left", va="top", fontsize=9,
              color="black")
    ax_t.set_xlabel(r"$\Delta x$ [px]")
    ax_t.set_ylabel(r"$\Delta y$ [px]")
    ax_t.tick_params(labelsize=7)

    row_data = (obs, model, chi)
    im_res = None
    for r in range(3):
        for c in range(n):
            ax = axes[r][c]
            cx, cy = centers[c]
            ext = (-0.5 - cx, hw - 0.5 - cx, -0.5 - cy, hw - 0.5 - cy)
            if r < 2:
                img = np.asarray(row_data[r][c], dtype=float)
                ax.set_facecolor(str_zero)
                ax.imshow(np.ma.masked_invalid(img), origin="lower",
                          cmap=cm_zero, vmin=vmin, vmax=vmax, extent=ext,
                          interpolation="nearest")
                ax.plot(0, 0, "+", color="white", ms=6, mew=0.9)
                dxy = np.asarray(q_cut[c]) - centers[c]
                ax.add_patch(Ellipse(tuple(dxy), 2 * (9 + 1.5),
                                     2 * (5 + 1.5), angle=35, fill=False,
                                     ls="--", lw=0.7, ec="white"))
            else:
                ax.set_facecolor(CMAP_DIV(0.5))
                im_res = ax.imshow(chi[c], origin="lower", cmap=CMAP_DIV,
                                   vmin=-vres, vmax=vres, extent=ext,
                                   interpolation="nearest")
                ax.plot(0, 0, "+", color="black", ms=6, mew=0.9)
                dxy = np.asarray(q_cut[c]) - centers[c]
                ax.add_patch(Ellipse(tuple(dxy), 2 * (9 + 1.5),
                                     2 * (5 + 1.5), angle=35, fill=False,
                                     ls="--", lw=0.7, ec="black"))
            if r == 0:
                ax.set_title(f"EP{c}", fontsize=8, pad=2)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xticks([-10, 0, 10])
            ax.set_yticks([-10, 0, 10])
            ax.tick_params(labelsize=6.5)
            if r < 2:
                ax.tick_params(labelbottom=False)
            if c > 0:
                ax.tick_params(labelleft=False)
    ax_t.set_xlim(-lim, lim)
    ax_t.set_ylim(-lim, lim)
    ax_t.set_xticks([-10, -5, 0, 5, 10])
    ax_t.set_yticks([-10, -5, 0, 5, 10])

    cb = fig.colorbar(im_img, cax=cax_img)
    cb.set_label(r"$y{+}D,\;m{+}D,\;\hat{G}$ [counts s$^{-1}$ px$^{-1}$]",
                 fontsize=8)
    cb.ax.tick_params(labelsize=6.5)
    cbr = fig.colorbar(im_res, cax=cax_res)
    cbr.set_label(r"$\left(y - m\right)\sqrt{w}\,/\,f$" if scaled
                  else r"$\left(y - m\right)\sqrt{w}$", fontsize=8)
    cbr.ax.tick_params(labelsize=6.5)

    # ---- post-layout alignment: pin the colorbars and the template to
    # the ACTUAL extents of the (aspect-shrunk) panel grid ----
    fig.draw_without_rendering()
    tl = axes[0][0].get_position()
    ml = axes[1][0].get_position()
    bl = axes[2][0].get_position()
    tr = axes[0][-1].get_position()
    br = axes[2][-1].get_position()
    cp = cax_img.get_position()
    cax_img.set_position([tr.x1 + 0.010, ml.y0, cp.width, tl.y1 - ml.y0])
    cax_res.set_position([br.x1 + 0.010, bl.y0, cp.width, bl.y1 - bl.y0])
    fw, fh = fig.get_size_inches()
    side = (tl.y1 - bl.y0) * fh / fw           # square box, figure fraction
    ax_t.set_position([tl.x0 - 0.030 - side, bl.y0, side, tl.y1 - bl.y0])
    save(fig, out_dir, "residuals_grid")


def fig_residuals(module_path, log_dir, out_dir):
    import torch

    ssu = import_module(module_path)
    obs, disks, masks, var, nominal, q_cut = ssu.load_native_data()
    info = load_info(log_dir)
    lay = run_layout(info)
    ll = ssu.LogLikelihood(obs, disks, masks, var, nominal,
                           noise_scale=lay["has_noise"])
    med = median_theta(info)
    theta = torch.as_tensor(med, dtype=torch.float32).view(1, -1)
    # standardized residuals in the likelihood's units: (y - m) sqrt(w) / f
    # with the fitted per-epoch noise scales (f = 1 for a v06 run)
    residuals, chi = ll.error_map(theta, reference=True)
    ghat = ll.template_mean(torch.as_tensor(med, dtype=torch.float64
                                            ).view(1, -1))
    if lay["has_noise"]:
        f_med = np.exp(med[lay["i_lnf"]])
        print("noise scales at the posterior median: "
              + ", ".join(f"EP{i} {v:.2f}" for i, v in enumerate(f_med)))
    y = to_np(ll.y)                       # disk-subtracted (likelihood) data
    disks_np = to_np(disks)
    obs_raw = to_np(obs)                  # raw cutouts, disk included
    # full forward model t S ghat + D.  The galaxy part is DISPLAYED only
    # where the likelihood evaluates (w > 0): outside, the ridge's weakly
    # constrained boundary modes carry noise-amplified values (~sigma/c
    # for small Keys taps c) that cancel exactly in data space and never
    # enter chi^2 -- cosmetic, but distracting.  Inside the aperture the
    # shown model is bit-identical to the likelihood's.
    masks_np = to_np(ll.w > 0).astype(float)
    model_full = disks_np + (y - to_np(residuals[0])) * masks_np
    offsets = median_offsets(info)
    support = to_np(ll.w > 0).astype(bool).any(0)
    _plot_residuals(obs_raw, model_full, to_np(chi[0]), to_np(ghat[0]),
                    np.asarray(q_cut), offsets, out_dir, support=support,
                    scaled=lay["has_noise"])


# --------------------------------------------------------------------------- #
# 4) eigenvalue spectrum of the template normal matrix
# --------------------------------------------------------------------------- #
def _plot_spectrum(eig, lam, out_dir, n_weighted=None):
    """Eigenvalue spectrum of H at the nominal registration.  Only the
    informative modes are drawn; the exact null space (eigenvalue 0) is
    indicated by a downward arrow instead of a clipped flat line."""
    eig = np.asarray(eig, dtype=float)
    rel = eig / eig[0]
    lam_rel = lam / eig[0]
    n_above = int((eig > lam).sum())
    p_tot = len(eig)
    print(f"modes above lambda: {n_above} of {p_tot}"
          + (f"; weighted pixels / epoch: {n_weighted}" if n_weighted else "")
          + f"; single-aperture area pi(a+m)(b+m) = "
            f"{np.pi * 10.5 * 6.5:.0f} px")

    c_line, c_lam = palette(2, 0.25, 0.85)
    apply_style()
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    y_lo = lam_rel * 1e-2
    j = np.arange(1, n_above + 1)
    ax.semilogy(j, rel[:n_above], color=c_line, lw=1.5, drawstyle="steps-mid",
                label=r"eigenvalues of $H$ ($\sigma_{j} > \lambda$)")
    ax.axhspan(lam_rel / 10, lam_rel * 10, color=c_lam, alpha=0.12)
    ax.axhspan(lam_rel / 3, lam_rel * 3, color=c_lam, alpha=0.18)
    ax.axhline(lam_rel, color=c_lam, lw=1.4, ls="--",
               label=r"$\lambda$ (bands: $\times/\div\,3$ and $10$)")
    ax.axvline(n_above + 0.5, color="0.4", lw=0.7, ls=":")
    # exact null space: arrow from the end of the plateau to the axis floor
    ax.annotate("", xy=(n_above + 0.5, y_lo * 2.0),
                xytext=(n_above + 0.5, rel[n_above - 1] * 0.45),
                arrowprops=dict(arrowstyle="-|>", color=c_line, lw=1.3,
                                shrinkA=0, shrinkB=0))
    ax.text(n_above / 2.0, np.sqrt(lam_rel * 30.0),
            rf"${n_above}$ modes above $\lambda$" + "\n(aperture pixels)",
            ha="center", va="center", fontsize=7.5, color="0.25")
    ax.text(n_above + 14, y_lo * 1.0e4,
            rf"$j > {n_above}$: exactly null" + "\n(template pixels outside\nthe aperture)",
            ha="left", va="center", fontsize=7.5, color="0.25")
    ax.set_xlabel(r"mode index $j$")
    ax.set_ylabel(r"$\sigma_{j} / \sigma_{\max}$")
    ax.set_xlim(1, p_tot)
    ax.set_ylim(y_lo, 4.0)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)
    save(fig, out_dir, "template_spectrum")


def fig_spectrum(module_path, out_dir):
    import torch

    ssu = import_module(module_path)
    obs, disks, masks, var, nominal, _ = ssu.load_native_data()
    ll = ssu.LogLikelihood(obs, disks, masks, var, nominal)
    with torch.no_grad():
        s_ref = ll.nominal.double().unsqueeze(0)
        t = torch.ones(1, *ll.dhat.shape, dtype=torch.float64)
        planes = ll._stencil_planes(s_ref, t, torch.float64)
        h = ll._dense_from_planes(planes)[0]
        eig = to_np(torch.linalg.eigvalsh(h))[::-1].copy()
    _plot_spectrum(eig, float(ll.lam), out_dir,
                   n_weighted=int(to_np((ll.w[0] > 0).sum())))


# --------------------------------------------------------------------------- #
# 5) surface-density comparison (van Sluijs fig. 13 axes)
# --------------------------------------------------------------------------- #
def _read_csv(path):
    header, lines = {}, []
    for line in Path(path).read_text().splitlines():
        if line.startswith("# "):
            if ": " in line:
                k, v = line[2:].split(": ", 1)
                header[k] = v
        elif line.strip():
            lines.append(line)
    names = lines[0].split(",")
    raw = {n: [] for n in names}
    for row in lines[1:]:
        for n, v in zip(names, row.split(",")):
            raw[n].append(v)
    out = {}
    for n, col in raw.items():
        try:
            out[n] = np.asarray([float(v) if v else np.nan for v in col])
        except ValueError:                   # non-numeric column (style)
            out[n] = np.asarray(col, dtype=object)
    return header, out


def _extent(arrays, side):
    """(x, y) of the left- or rightmost finite vertex across curve pieces."""
    x = np.concatenate([a["x"] for a in arrays])
    y = np.concatenate([a["y"] for a in arrays])
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    i = np.argmax(x) if side == "right" else np.argmin(x)
    return float(x[i]), float(y[i])


def _line_verts(a):
    """Per-segment line polylines (obstacles for _dodge); segment-aware so
    gaps between segments stay label-free space."""
    seg = a.get("segment", np.zeros_like(a["x"]))
    out = []
    for s in np.unique(seg):
        m = (seg == s) & np.isfinite(a["x"]) & np.isfinite(a["y"])
        if m.sum():
            out.append(np.column_stack([a["x"][m], a["y"][m]]))
    return out


def _clamp_h(fig, ax, texts, pad=2.0):
    """Shift offset-point annotations horizontally so they lie inside the
    axes (fixed labels are not routed through _dodge, which clamps only
    the entries it moves)."""
    ren = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(ren)
    for txt in texts:
        bb = txt.get_window_extent(ren)
        dxp = 0.0
        if bb.x0 < axbb.x0 + pad:
            dxp = (axbb.x0 + pad - bb.x0) * 72.0 / fig.dpi
        elif bb.x1 > axbb.x1 - pad:
            dxp = (axbb.x1 - pad - bb.x1) * 72.0 / fig.dpi
        if dxp:
            txt.xyann = (txt.xyann[0] + dxp, txt.xyann[1])


def _dodge(fig, ax, entries, polylines, avoid=(), pad=2.0, step=1.5,
           max_steps=40):
    """Nudge annotations vertically (offset points) until each clears the
    plotted lines and the labels placed before it.  ``entries`` is a list
    of ``(annotation, direction)`` with direction +1 (up) / -1 (down) --
    the side the label was asked to sit on.  Runs after the axis limits are
    final, so label positions adapt to whatever CSVs were loaded."""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    pts = []
    for line in polylines:                     # densify in pixel space so no
        q = ax.transData.transform(line)       # label straddles a segment
        pts.append(q)
        for (x0, y0), (x1, y1) in zip(q[:-1], q[1:]):
            n = int(np.hypot(x1 - x0, y1 - y0) // 4)
            if n:
                t = np.linspace(0, 1, n + 2)[1:-1, None]
                pts.append((1 - t) * (x0, y0) + t * (x1, y1))
    pix = np.vstack(pts) if pts else np.empty((0, 2))
    placed = [t.get_window_extent(ren).padded(pad) for t in avoid]
    ax_bb = ax.get_window_extent(ren)
    px2pt = 72.0 / fig.dpi
    for txt, d in entries:
        bb = txt.get_window_extent(ren)      # keep the text inside the axes
        dx, dy = txt.xyann                   # before clearing it vertically
        if bb.x0 < ax_bb.x0 + 2:
            dx += (ax_bb.x0 + 2 - bb.x0) * px2pt
        elif bb.x1 > ax_bb.x1 - 2:
            dx -= (bb.x1 - ax_bb.x1 + 2) * px2pt
        if bb.y0 < ax_bb.y0 + 2:
            dy += (ax_bb.y0 + 2 - bb.y0) * px2pt
        elif bb.y1 > ax_bb.y1 - 2:
            dy -= (bb.y1 - ax_bb.y1 + 2) * px2pt
        txt.xyann = (dx, dy)
        for _ in range(max_steps):
            bb = txt.get_window_extent(ren).padded(pad)
            hit = ((pix[:, 0] >= bb.x0) & (pix[:, 0] <= bb.x1) &
                   (pix[:, 1] >= bb.y0) & (pix[:, 1] <= bb.y1)).any()
            if not hit and not any(bb.overlaps(p) for p in placed):
                break
            dx, dy = txt.xyann
            txt.xyann = (dx, dy + d * step)
        bb = txt.get_window_extent(ren)      # never leave the axes, even if
        dx, dy = txt.xyann                   # the dodge ran out of room
        if bb.y1 > ax_bb.y1 - 2:
            dy -= (bb.y1 - ax_bb.y1 + 2) * px2pt
        elif bb.y0 < ax_bb.y0 + 2:
            dy += (ax_bb.y0 + 2 - bb.y0) * px2pt
        txt.xyann = (dx, dy)
        placed.append(txt.get_window_extent(ren).padded(pad))


def _upper_limits(ax, r, y, color, arrow=0.30, bar_pts=6.0):
    """95% upper limits: horizontal bar at the limit + sharp downward arrow."""
    ax.plot(r, y, ls="none", marker="_", ms=bar_pts, mew=1.5, color=color,
            zorder=6)
    for xi, yi in zip(np.atleast_1d(r), np.atleast_1d(y)):
        ax.annotate("", xy=(xi, yi - arrow), xytext=(xi, yi), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", mutation_scale=7,
                                    color=color, lw=1.1,
                                    shrinkA=0, shrinkB=0))


#: grain density behind the van Sluijs et al. (2018) attenuation point
#: (their fig. 13 assumptions box); the point is shifted by
#: log10(rho_ours / VS_RHO_GCC) so every optical-depth-based dataset in the
#: figure shares the grain density of comparison_values.json.  The ALMA
#: surface densities (Ricci et al. 2015; Han et al. 2026) are plotted as
#: published: they rest on mm-emission mass opacities (e.g. 1.9 cm^2/g at
#: 0.88 mm for ARKS), which the geometric-optics closure of our mu-m-grain
#: probes cannot reproduce, and a bare density swap is not a defined
#: operation on such kappa prescriptions -- those curves describe the
#: mm-grain population and are tagged as emission in the figure.
VS_RHO_GCC = 1.2

_LS = {"solid": "-", "dashed": "--", "dotted": ":"}


def _lit_file(lit, sub, name):
    """Resolve an extracted literature CSV.  ``lit`` may be the
    literature_data root (<lit>/<sub>/extracted/<name>), a flat directory
    holding all CSVs, or -- for backward compatibility -- a path inside
    one extraction (e.g. .../fig13/extracted), whose siblings are then
    searched."""
    lit = Path(lit)
    for c in (lit / sub / "extracted" / name, lit / name,
              lit.parent.parent / sub / "extracted" / name):
        if c.exists():
            return c
    sys.exit(f"figure 5: {name} not found; looked for "
             f"{sub}/extracted/{name} under {lit} (and as a flat file, "
             "and next to a fig13/extracted-style --lit-dir).  Point "
             "--lit-dir at the literature_data root.")


def _tolog(v, scale):
    """log10 of positive values (times ``scale``); measured nonpositive
    values map to -30 (unbounded below on the log axis: drawn to the
    frame bottom); NaN stays NaN."""
    out = np.full(np.shape(v), np.nan)
    pos = np.asarray(v) > 0
    out[pos] = np.log10(np.asarray(v)[pos] * scale)
    out[np.isfinite(v) & ~pos] = -30.0
    return out


def _sigma_curve(ax, path, color, scale, band_alpha=0.25, lw=1.1,
                 ls=None, gap_break=False):
    """Linear surface-density extraction -> log10(Sigma x 1e4 g cm^-2).

    ``scale`` maps the file's y unit onto 1e4 x Sigma [g cm^-2] (the
    ricci16 files already carry that quantity -> 1.0; the ARKS profile is
    in ug cm^-2 -> 1e-2).  ``ls`` fixes one line style for the whole
    curve; with ``ls=None`` a ``style`` column, if present, switches the
    style along the curve as in the source figure.  ``gap_break``: treat
    the extraction's ``dotted`` run (the digitized floor of an empty
    density gap) as a true break -- no line, no band there; instead each
    edge of the gap gets a short vertical tail capped by a downward
    triangle, marking that the profile drops off the axis (Sigma -> 0).
    Nonpositive values and band edges run to the frame bottom.
    """
    _, a = _read_csv(path)
    x, y = a["x"], _tolog(a["y"], scale)
    sty = a.get("style")
    if sty is None:
        sty = np.full(len(x), "solid", dtype=object)
    else:
        last = "solid"                       # forward-fill blank styles
        sty = np.asarray([(last := s) if s else last for s in sty],
                         dtype=object)
    in_gap = (sty == "dotted") if gap_break else np.zeros(len(x), bool)
    if in_gap.any():
        # absorb the blank jump rows flanking the gap so the uncertainty
        # band does not plunge across the break as a thin whisker
        g = np.flatnonzero(in_gap)
        lo_i, hi_i = g[0], g[-1]
        while lo_i > 0 and not np.isfinite(y[lo_i - 1]):
            lo_i -= 1
        while hi_i < len(y) - 1 and not np.isfinite(y[hi_i + 1]):
            hi_i += 1
        in_gap[lo_i:hi_i + 1] = True
    if "y_lower" in a:
        lo, hi = _tolog(a["y_lower"], scale), _tolog(a["y_upper"], scale)
        good = np.isfinite(lo) & np.isfinite(hi) & ~in_gap
        if band_alpha and good.any():
            ax.fill_between(x, np.where(good, lo, 0.0),
                            np.where(good, hi, 0.0), where=good,
                            color=color, alpha=band_alpha, lw=0)
    # contiguous curve pieces: blank rows at the near-vertical jumps are
    # bridged, the gap run is a real break
    idx = np.flatnonzero(np.isfinite(y) & ~in_gap)
    segs, cur = [], [idx[0]]
    for k0, k1 in zip(idx[:-1], idx[1:]):
        if in_gap[k0 + 1:k1].any():
            segs.append(cur); cur = [k1]
        else:
            cur.append(k1)
    segs.append(cur)
    for seg in segs:
        seg = np.asarray(seg)
        if ls is not None:
            ax.plot(x[seg], y[seg], color=color, lw=lw, ls=ls)
            continue
        cuts = np.flatnonzero(sty[seg][1:] != sty[seg][:-1]) + 1
        for k, sub in enumerate(np.split(np.arange(len(seg)), cuts)):
            if k:                            # keep the polyline connected
                sub = np.r_[sub[0] - 1, sub]
            ax.plot(x[seg[sub]], y[seg[sub]], color=color, lw=lw,
                    ls=_LS.get(sty[seg[sub[-1]]], "-"))
    if gap_break and len(segs) > 1:          # tails + caps at the gap edges
        for k in range(len(segs) - 1):
            for j in (segs[k][-1], segs[k + 1][0]):
                ax.plot([x[j], x[j]], [y[j], y[j] - 0.25], color=color,
                        lw=lw, ls="-", solid_capstyle="butt")
                ax.plot([x[j]], [y[j] - 0.29], "v", ms=3.5, color=color)
    keep = np.concatenate(segs)
    return {"x": x[keep], "y": y[keep],
            "segment": np.concatenate([np.full(len(s), i)
                                       for i, s in enumerate(segs)])}


def _ardila_sigma(ax, path, color, conv, band_alpha=0.25):
    """Ardila et al. (2004) fig. 3 profile -> log10(Sigma x 1e4 g cm^-2).

    The extraction carries absolute, erratum-corrected omega*tau with
    per-point error-bar ends in ``y_lower``/``y_upper``; ``conv`` is the
    additive log10 offset of the scattering conversion built in
    fig_comparison.  Lower edges consistent
    with zero are unbounded below in the log and are drawn to the bottom of
    the frame.
    """
    hdr, a = _read_csv(path)
    # Unit guard: the extractor (extract_f3.py) writes absolute, erratum-
    # corrected tau*omega (peak a few 1e-4) and says so in the header.
    # Any other file -- the old "x 1e3" convention, or the as-printed axis
    # values (x 1e-6) -- would silently land the curve 3 dex off, which
    # is exactly the mistake that produced the original factor-20 problem.
    header_text = " ".join(ln for ln in Path(path).read_text().splitlines()
                           if ln.startswith("#"))
    ymax = float(np.nanmax(a["y"]))
    if "ABSOLUTE" not in header_text.upper() or not (1e-5 < ymax < 1e-2):
        raise SystemExit(
            f"{path}: expected absolute erratum-corrected tau*omega "
            f"(peak ~5e-4, header marked ABSOLUTE); found peak {ymax:.3g}. "
            "Re-run extract_f3.py (the version with the erratum pre-factor) "
            "and point --lit-dir at its 'extracted' output.")
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.log10(a["y"]) + conv
        lo = np.log10(a["y_lower"]) + conv
        hi = np.log10(a["y_upper"]) + conv
    lo = np.where(a["y_lower"] > 0, lo, -30.0)     # consistent with zero
    good = np.isfinite(lo) & np.isfinite(hi)
    if band_alpha and good.any():
        ax.fill_between(a["x"][good], lo[good], hi[good], color=color,
                        alpha=band_alpha, lw=0)
    ax.plot(a["x"], y, color=color, lw=1.1)
    return {"x": a["x"], "y": y}


def fig_comparison(log_dir, lit_dir, out_dir, albedo=0.5,
                   scatter_a_cm=None):
    # every curve from its original extraction; the fig. 13 compilation
    # supplies only the van Sluijs measurement itself
    f_ard = [_lit_file(lit_dir, "fig3", f"tau_omega_{b}.csv")
             for b in ("F606W", "F814W")]
    f_ricci = {m: _lit_file(lit_dir, "ricci16", f"surface_density_{m}.csv")
               for m in ("double_power_law", "single_power_law",
                         "single_power_law_gap")}
    f_han = _lit_file(lit_dir, "arks_radial_profile", "HD107146.csv")
    f_att = _lit_file(lit_dir, "fig13", "attenuation_errorbar.csv")

    candidates = [Path(log_dir) / "extra" / "comparison_values.json",
                  Path(out_dir) / "comparison_values.json",
                  Path(out_dir).parent / "comparison_values.json",
                  Path(log_dir) / "comparison_values.json"]
    comp_path = next((c for c in candidates if c.exists()), None)
    if comp_path is None:
        sys.exit(
            "figure 5 needs comparison_values.json (searched: "
            + ", ".join(str(c) for c in candidates) + ").\n"
            "Produce it with paper_values.py, then re-run this figure:\n"
            f"  python paper_values.py {log_dir} --evaluate "
            "--module <self_sufficient_ultranest.py>\n"
            f"  python paper_figures.py {log_dir} --figs 5 "
            f"--lit-dir {lit_dir}\n"
            "(--evaluate needs the data; without it the json is still "
            "written but holds only the peak limit, no per-epoch points)")
    print(f"figure 5: using {comp_path}")
    comp = json.loads(comp_path.read_text())
    asm = comp["assumptions"]

    # Ardila et al. (2004) measure the scattering product omega*tau, where
    # tau is the GEOMETRIC optical depth of the paper's eq. (sb_relation)
    # and omega the albedo relative to the geometric cross section.  With an
    # assumed albedo, tau_geo = (omega tau) / omega, and the mass column is
    # Sigma = tau_geo 4 rho a / 3 = (omega tau) 4 rho a / (3 omega): no
    # Q_ext here -- Q_ext = 2 enters only the ATTENUATION columns, whose
    # tau' is an extinction optical depth (Sigma = tau' / kappa_ext with
    # kappa_ext = 3 Q_ext / (4 rho a)).  An earlier version divided the
    # scattering columns by Q_ext as well, which put them a factor 2 too
    # low for a given omega.  The extraction
    # (extract_f3.py) stores ABSOLUTE omega*tau, already corrected for the
    # erratum (Ardila et al. 2005: the 2004 values are 1600/4pi = 127.3
    # too small) -- NOT the printed axis value, whose unit is 1e-6, and
    # not "x 1e3" as an earlier version of this code assumed (that put the
    # curve a factor 7.9 too high).  The axis is log10(Sigma x 1e4), so the
    # conversion is one additive log offset.  --scatter-a-cm substitutes a
    # different grain size for the scattering curves only (e.g. a size-
    # distribution effective radius, as in the van Sluijs 2018 rendering).
    a_sc = scatter_a_cm if scatter_a_cm else asm["grain_a_cm"]
    conv = np.log10(1.0e4 * 4.0 * asm["grain_rho_gcc"] * a_sc
                    / (3.0 * albedo))
    dy_vs = np.log10(asm["grain_rho_gcc"] / VS_RHO_GCC)   # see VS_RHO_GCC

    c_v, c_i, c_han, c_dpl, c_spl, c_gap = palette(6, 0.18, 0.88)
    c_us = palette(1, 0.02, 0.02)[0]      # ours: darkest end of the scheme
    c_vs = "0.35"                          # external point: neutral grey
    c_mech = "0.45"                        # mechanism tags: muted

    sl = comp.get("sightlines")
    if sl:
        sl = [s for s in sl if s["dhat"] >= 0.01]   # drop off-disk epochs
    if not sl:
        warnings.warn("no per-epoch sightlines in comparison_values.json "
                      "(run paper_values.py --evaluate); plotting the peak "
                      "limit only")
        r = np.array([comp["peak"].get("r_au", 100.0)])
        yy = np.array([comp["peak"]["log_sigma_ul95"]])
    else:
        r = np.array([s["r_au"] for s in sl])
        yy = np.array([s["log_sigma_ul95"] for s in sl])

    apply_style()
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=1.5, foreground="white")]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ricci = {m: _sigma_curve(ax, f, c, scale=1.0, ls=s, gap_break=g)
             for (m, f), c, s, g in zip(
                 f_ricci.items(), (c_dpl, c_spl, c_gap),
                 ("-", "--", "-."), (False, False, True))}
    han = _sigma_curve(ax, f_han, c_han, scale=1e-2, lw=1.3)
    av = _ardila_sigma(ax, f_ard[0], c_v, conv)
    ai = _ardila_sigma(ax, f_ard[1], c_i, conv)
    ymin = min(yy.min() - 0.60,
               np.nanmin(av["y"]) - 0.12, np.nanmin(ai["y"]) - 0.12)

    # direct labels at the curve extents replace the legend
    def label(xy, text, color, dx=0, dy=0, ha="left", va="center", fs=6.5,
              **kw):
        return ax.annotate(text, xy, textcoords="offset points",
                           xytext=(dx, dy), ha=ha, va=va, fontsize=fs,
                           color=color, zorder=5, **kw)

    x_pk = float(han["x"][np.nanargmax(han["y"])])
    y_pk = float(np.nanmax(han["y"]))
    ren = fig.canvas.get_renderer()
    to_pt = 72.0 / fig.dpi

    # the mm-emission rail (left curve starts, bottom to top: single PL,
    # single PL + gap, double PL, mechanism header) and the Han label are
    # placed deterministically so the layout is stable against font
    # metrics; the dodged labels below keep clear of everything in
    # ``fixed``
    fixed = [
        label(_extent([ricci["single_power_law"]], "left"),
              "single PL", c_spl, dx=-4, ha="right", va="bottom"),
        label(_extent([ricci["single_power_law_gap"]], "left"),
              "single PL + gap", c_gap, dx=-4, dy=2, ha="right",
              va="bottom"),
    ]
    dpl = label(_extent([ricci["double_power_law"]], "left"),
                "Ricci et al. 2015\ndouble PL", c_dpl,
                dx=-4, ha="right", va="bottom")
    h_dpl = dpl.get_window_extent(ren).height * to_pt
    fixed += [
        dpl,
        label(_extent([ricci["double_power_law"]], "left"),
              "emission (mm)", c_mech, dx=-4, dy=h_dpl + 3, ha="right",
              va="bottom", fs=6, fontstyle="italic"),
        label((x_pk, y_pk), "Han et al. 2026\n(ARKS)", c_han,
              dx=5, dy=4, ha="left", va="bottom"),
    ]

    # Ardila pair labelled as one block: filter labels immediately above
    # and below the scattering mechanism tag
    xa_i, ya_i = _extent([ai], "left")
    xa_v, ya_v = _extent([av], "left")
    xs, ys = min(xa_i, xa_v), 0.5 * (ya_i + ya_v)
    fixed += [
        label((xs, ys), "scattering ($\\mu$m)", c_mech, dx=-4,
              ha="right", va="center", fs=6, fontstyle="italic"),
        label((xs, ys), "Ardila et al. 2004\n(F814W)", c_i, dx=-4, dy=7,
              ha="right", va="bottom"),
        label((xs, ys), "Ardila et al. 2004\n(F606W)", c_v, dx=-4,
              dy=-7, ha="right", va="top"),
    ]
    obstacles = [pl for a in [av, ai, han] + list(ricci.values())
                 for pl in _line_verts(a)]

    # van Sluijs attenuation constraint (one-sided: lower side unbounded),
    # shifted onto our grain density (dy_vs)
    _, att = _read_csv(f_att)
    x, y = att["x"][0], att["y"][0] + dy_vs
    y_cap = att["y_upper"][0] + dy_vs
    ax.hlines(y, att["x_lower"][0], att["x_upper"][0], color=c_vs, lw=1.2)
    ax.vlines(x, -30, y_cap, color=c_vs, lw=1.2)
    ax.hlines(y_cap, x - 3, x + 3, color=c_vs, lw=0.9)
    ax.plot([x], [y], "s", ms=5, color=c_vs)
    fixed.append(label((att["x_lower"][0], y),
                       "van Sluijs et al. 2018\n(attenuation)",
                       c_vs, dx=-4, ha="right", va="center",
                       path_effects=halo))

    # our per-epoch 95% upper limits
    _upper_limits(ax, r, yy, c_us)
    if sl:
        # vertical epoch tags: EP0 right of / EP1 left of its bar,
        # EP2 + EP7 rising from just above theirs
        ep_style = {"EP0": dict(dx=5, ha="center", va="top"),
                    "EP1": dict(dx=-5, ha="center", va="bottom"),
                    "EP2": dict(dy=5, ha="left", va="center"),
                    "EP7": dict(dy=5, ha="left", va="center")}
        for s in sl:
            name = str(s["epoch"])
            name = name if name.startswith("EP") else f"EP{name}"
            st = ep_style.get(name)
            if st:
                fixed.append(
                    label((s["r_au"], s["log_sigma_ul95"]), name, c_us,
                          fs=6, rotation=90, rotation_mode="anchor",
                          path_effects=halo, **st))
        # the label sits just below the arrow tips of the dense epoch
        # cluster; EP0/EP1-style outliers lie beyond the widest radial
        # gap and are excluded so they cannot drag it down or sideways
        order = np.argsort(r)
        rs, ws = r[order], yy[order]
        gaps = np.diff(rs)
        if len(rs) > 2 and gaps.max() > 15.0:
            k = int(np.argmax(gaps)) + 1
            cl = slice(0, k) if k >= len(rs) - k else slice(k, len(rs))
        else:
            cl = slice(0, len(rs))
        x_lab = 0.5 * (rs[cl].min() + rs[cl].max())
        y_lab = float(ws[cl].min()) - 0.40
    else:
        x_lab, y_lab = float(r[0]), float(yy[0]) - 0.40
    tw = label((x_lab, y_lab), "this work", c_us, ha="center", va="top")
    h_tw = tw.get_window_extent(ren).height * to_pt
    fixed += [tw,
              label((x_lab, y_lab), "attenuation ($\\mu$m)", c_mech,
                    dy=-(h_tw + 2), ha="center", va="top", fs=6,
                    fontstyle="italic")]
    dodge = []

    box = [rf"$Q_{{\rm ext}} = {asm['q_ext']:g}$",
           rf"$\rho = {asm['grain_rho_gcc']:g}\;\mathrm{{g\,cm^{{-3}}}}$",
           rf"$\bar{{a}} = {asm['grain_a_cm'] * 1e4:g}\;\mu\mathrm{{m}}$",
           rf"$\omega = {albedo:g}$"]
    if a_sc != asm["grain_a_cm"]:
        box.insert(3, rf"$a_{{\rm sc}} = {a_sc * 1e4:g}\;\mu\mathrm{{m}}$")
    fixed.append(ax.text(0.985, 0.975, "\n".join(box),
                         transform=ax.transAxes, ha="right", va="top",
                         fontsize=7))   # labels must also clear the box

    tops = ([np.nanmax(av["y"]), np.nanmax(ai["y"]), y_cap, yy.max(), y_pk]
            + [np.nanmax(q["y"]) for q in ricci.values()])
    ax.set_xlim(0, 200)
    ax.set_ylim(ymin, max(tops) + 0.48)   # top strip: Han label sits above its peak label
    ax.set_xlabel("Radius [au]")
    ax.set_ylabel(r"$\log_{10}\Sigma\;[\times 10^{4}\,\mathrm{g\,cm^{-2}}]$")
    _clamp_h(fig, ax, fixed)
    _dodge(fig, ax, dodge, obstacles, avoid=fixed)
    save(fig, out_dir, "sigma_comparison")


# --------------------------------------------------------------------------- #
# 6) per-epoch noise-scale posteriors (v07 runs)
# --------------------------------------------------------------------------- #
def _wquantiles(x, w, qs):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    cw /= cw[-1]
    return np.array([np.interp(q, cw, x[o]) for q in qs])


def fig_noise_scales(log_dir, out_dir, max_samples=None):
    """Posterior of the per-epoch noise scales f_i: median, 68 % (thick)
    and 95 % (thin) intervals per epoch, the nominal noise model f = 1,
    and -- when ``paper_values.py --evaluate`` has been run -- the square
    root of the nominal per-epoch MSR at the posterior median (the value
    a single-scale correction of the fixed noise model would imply; it
    exceeds f_i by the template degrees of freedom absorbed per epoch)."""
    results, info = load_run(log_dir, max_samples=max_samples)
    lay = param_layout(results["paramnames"])
    if not lay["has_noise"]:
        warnings.warn("figure 6: this run has no per-epoch noise scales "
                      "(v06 layout); nothing to plot")
        return
    pts = np.asarray(results["weighted_samples"]["points"])
    wts = np.asarray(results["weighted_samples"]["weights"], dtype=float)
    f = np.exp(pts[:, lay["i_lnf"]])
    n = f.shape[1]
    q = np.stack([_wquantiles(f[:, i], wts, (0.025, 0.16, 0.5, 0.84, 0.975))
                  for i in range(n)])                       # (N, 5)

    msr = None
    candidates = [Path(log_dir) / "paper" / "paper_values.json",
                  Path(out_dir).parent / "paper_values.json",
                  Path(log_dir) / "paper_values.json"]
    pv = next((c for c in candidates if c.exists()), None)
    if pv is not None:
        try:
            raw = json.loads(pv.read_text())
            msr = raw.get("extras", {}).get("msr_per_epoch")
            if msr is not None:
                msr = np.asarray(msr, dtype=float)
                print(f"figure 6: sqrt(nominal MSR_i) from {pv}")
        except Exception as exc:                      # noqa: BLE001
            warnings.warn(f"figure 6: could not read {pv}: {exc}")

    c_post, c_ref = palette(2, 0.25, 0.85)
    apply_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    x = np.arange(n)
    ax.axhline(1.0, color="0.55", lw=0.9, ls="--", zorder=1,
               label="nominal noise model ($f = 1$)")
    ax.errorbar(x, q[:, 2], yerr=[q[:, 2] - q[:, 0], q[:, 4] - q[:, 2]],
                fmt="none", ecolor=c_post, elinewidth=0.8, capsize=0,
                alpha=0.55, zorder=2)
    ax.errorbar(x, q[:, 2], yerr=[q[:, 2] - q[:, 1], q[:, 3] - q[:, 2]],
                fmt="o", color=c_post, ecolor=c_post, elinewidth=2.0,
                ms=4.5, capsize=0, zorder=3,
                label=r"posterior $f_{i}$ (68 %, 95 %)")
    if msr is not None and msr.shape[0] == n:
        ax.plot(x, np.sqrt(msr), "D", mfc="none", mec=c_ref, ms=5.5, mew=1.1,
                zorder=4, label=r"$\sqrt{\mathrm{MSR}_{i}}$ (nominal weights)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"EP{i}" for i in range(n)], fontsize=8)
    ax.set_ylabel(r"noise scale $f_{i}$")
    lo = min(0.9, q[:, 0].min() - 0.1)
    hi = q[:, 4].max() + 0.25
    if msr is not None and msr.shape[0] == n:
        hi = max(hi, np.sqrt(msr).max() + 0.25)
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.6, n - 0.4)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    gm = float(np.exp(np.average(np.log(q[:, 2]))))
    ax.text(0.975, 0.05,
            rf"geometric mean $\langle f\rangle = {gm:.2f}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            color="0.35")
    save(fig, out_dir, "noise_scales")


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log_dir")
    ap.add_argument("--figs", nargs="+", default=["1", "2", "3", "4", "5", "6"])
    ap.add_argument("--module", default=None,
                    help="path to self_sufficient_ultranest.py (figs 2-4)")
    ap.add_argument("--lit-dir", default=None,
                    help="literature_data root with fig3/, ricci16/, "
                         "arks_radial_profile/ and fig13/ extractions "
                         "(a flat directory of the CSVs, or the old "
                         "fig13/extracted path, also works; fig 5)")
    ap.add_argument("--albedo", type=float, default=0.5,
                    help="albedo omega for the Ardila scattering "
                         "conversion (default 0.5)")
    ap.add_argument("--scatter-a-cm", type=float, default=None,
                    help="grain size [cm] for the scattering conversion "
                         "only (default: grain_a_cm from "
                         "comparison_values.json)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--params", nargs="+", default=None,
                    help="subset of parameters for fig 1, given as group "
                         "keywords (all, offsets, noise, tau), the run's "
                         "own names, or raw ones (e.g. tau dx_2 lnf_0); "
                         "default: all parameters of a v06 run, offsets + "
                         "tau of a v07 run")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="test mode: resample the posterior to at most this "
                         "many points before plotting (figs 1, 6)")
    args = ap.parse_args(argv)

    out = Path(args.out_dir) if args.out_dir else Path(args.log_dir) / "paper" / "figures"
    for f in args.figs:
        if f == "1":
            fig_posterior(args.log_dir, out, params=args.params,
                          max_samples=args.max_samples)
        elif f in ("2", "3", "4"):
            if not args.module:
                sys.exit(f"figure {f} needs --module")
            {"2": lambda: fig_disk_track(args.module, args.log_dir, out),
             "3": lambda: fig_residuals(args.module, args.log_dir, out),
             "4": lambda: fig_spectrum(args.module, out)}[f]()
        elif f == "5":
            if not args.lit_dir:
                sys.exit("figure 5 needs --lit-dir")
            fig_comparison(args.log_dir, args.lit_dir, out,
                           albedo=args.albedo,
                           scatter_a_cm=args.scatter_a_cm)
        elif f == "6":
            fig_noise_scales(args.log_dir, out, max_samples=args.max_samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())