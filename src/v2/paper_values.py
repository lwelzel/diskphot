#!/usr/bin/env python3
"""Compute the numbers reported in the paper from the inference-run outputs
and write them as LaTeX ``\\newcommand`` macros (plus ready-made tables).

Inputs (all inside the UltraNest ``log_dir`` of the production run):
    info/results.json               UltraNest run summary (logZ, posterior
                                    moments, ML point, ESS, ncall, ...)
    chains/equal_weighted_post.txt  equal-weighted posterior samples
    sensitivity.json                single-run prior-sensitivity table
                                    (written by self_sufficient_ultranest)
    MODEL_CARD.txt                  priors of the run (lam, sigma_g, P,
                                    offset bound, tau' ceiling, noise-scale
                                    range, theta layout)
    verify.json                     verify() audit numbers (v07 runs; older
                                    runs fall back to DEFAULT_EXTRAS)

Outputs (default: <log_dir>/paper/):
    paper_values.tex        one \\newcommand per reported value
    table_offsets.tex       per-epoch registration-offset table
    table_sensitivity.tex   prior-sensitivity table
    table_noise.tex         per-epoch noise-scale table (v07 runs)
    table_sightlines.tex    per-epoch sightline limits (with --evaluate)
    comparison_values.json  inputs of the surface-density comparison figure
    paper_values.json       raw numbers, for the record

Parameter layout.  v07 runs sample theta = (offsets, ln f_0..ln f_{N-1},
tau'), v06 runs theta = (offsets, tau'); tau' is last in both.  The layout
is recovered from the run's ``paramnames`` (``param_layout``), so both
kinds of run directory are handled, and every index into theta goes
through it -- never through ``[:-1]`` / ``[-1]`` arithmetic.

Every macro is ALWAYS defined; values that could not be computed are set to
``\\textbf{??}`` and listed in a warning so a missing input can never break
the LaTeX build silently.  Values that come from the verification audit
rather than the run directory (volume-term slope, mode counts, f32 check)
are taken from ``<log_dir>/verify.json`` when present, else from
``--extras extras.json`` when given; the built-in defaults are the values
recorded from the v06 production run's ``verify()`` and are clearly marked
as such below.

Optional ``--evaluate --module <self_sufficient_ultranest.py>`` re-imports
the likelihood (requires the data paths of the module to resolve) and
computes the goodness-of-fit numbers (chi^2 and the mean squared
standardized residual at the posterior median, under the NOMINAL noise
model and under the fitted per-epoch noise scales) and the per-epoch
sightline limits.

Usage:
    python paper_values.py <log_dir> [--out-dir DIR] [--extras extras.json]
                           [--evaluate --module path/to/script.py]
"""

import argparse
import json
import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# defaults for audit-derived quantities: recorded from the v06 production
# run's verify() / sensitivity post-processing (2026-08, 512 live points,
# FIXED nominal weights).  A v07 run (per-epoch noise scales) writes
# verify.json and the f32 check into sensitivity.json, which take precedence
# automatically; --extras overrides everything.
# --------------------------------------------------------------------------- #
DEFAULT_EXTRAS = {
    "volume_slope": -401.3,        # verify(): dlogdet_dtau_fd  [v06]
    "volume_slope_lnf": None,      # verify(): dlogdet_dlnf_fd  [v07 only]
    "n_modes_above_lam": 315,      # verify(): n_modes_above_lam  [v06]
    "h_cond_unridged": 6.0e12,     # verify(): H_cond_unridged  [v06]
    "f32_max_dlnl": 3.03e-3,       # sensitivity block: max|f32 - f64| at
                                   # the posterior points  [v06]
    "noise_scale_identity": None,  # verify(): dlnl_noise_scale_identity
    "lam_rel_init": 1.0e-5,        # empirical-Bayes scale (ctor default)
    "tau_hi": 1.0,                 # flat tau' prior ceiling (MODEL_CARD wins)
    "offset_halfwidth": 1.0,       # offset prior half-width [px] (MODEL_CARD
                                   # wins: the script's OFFSET_HALFWIDTH is
                                   # 1.5 -- check which one the run used)
    "noise_scale_range": None,     # log-uniform f prior bounds (MODEL_CARD)
    "n_live": 521,                 # NOTE: the v16 run script sets
                                   # min_num_live_points=521; confirm which
                                   # value the production run used and pass
                                   # it via --extras if it differs
    "msr": None,                   # mean squared standardized residual,
                                   # NOMINAL noise model
    "msr_scaled": None,            # ... under the fitted noise scales
    "chi2_med": None,              # chi^2 at the posterior median (nominal)
    "chi2_scaled_med": None,       # chi^2 at the posterior median (scaled)
    "n_weighted_pix": None,        # number of weighted pixels
    # geometry + grain assumptions for the tau -> Sigma comparison figure
    # (constants.py of the pipeline).  NOTE: van Sluijs fig. 13 adopted
    # rho = 1.2 g cm^-3 (their omega = 0.5 enters scattering, not
    # attenuation); override via --extras to match their conversion exactly.
    "pixel_scale_arcsec": 0.05072,  # STIS 50CCD plate scale
    "parallax_mas": 36.404,         # Gaia DR3
    "q_ext": 2.0,                   # geometric-optics extinction efficiency
    "grain_rho_gcc": 1.5,           # solid grain density [g cm^-3]
    "grain_a_cm": 3.7e-4,           # effective mean grain radius (3.7 um)
}

# verify.json key -> extras key
VERIFY_KEYS = {
    "dlogdet_dtau_fd": "volume_slope",
    "dlogdet_dlnf_fd": "volume_slope_lnf",
    "n_modes_above_lam": "n_modes_above_lam",
    "H_cond_unridged": "h_cond_unridged",
    "dlnl_noise_scale_identity": "noise_scale_identity",
}

PCT16, PCT84 = 0.158655, 0.841345   # Gaussian-equivalent one-sigma quantiles


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def _round_sig(x, n=2):
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


def _decimals_for(err, n=2):
    """Number of decimals that displays ``err`` with n significant digits."""
    if err == 0 or not math.isfinite(err):
        return n
    return max(0, -int(math.floor(math.log10(abs(err)))) + (n - 1))


def fmt_plain(x, sig=2):
    """Plain decimal at ``sig`` significant digits (no exponent)."""
    if x is None or not math.isfinite(x):
        return r"\textbf{??}"
    xr = _round_sig(x, sig)
    d = max(0, -int(math.floor(math.log10(abs(xr)))) + (sig - 1)) if xr != 0 else sig
    return f"{xr:.{d}f}"


def fmt_pow_bare(x, sig=2):
    """Scientific notation ``a\\times 10^{b}`` without the ensuremath wrap."""
    if x is None or not math.isfinite(x):
        return r"\textbf{??}"
    if x == 0:
        return "0"
    b = int(math.floor(math.log10(abs(x))))
    a = x / 10.0 ** b
    a = _round_sig(a, sig)
    if abs(a) >= 10:
        a /= 10.0
        b += 1
    return rf"{a:.{sig - 1}f}\times 10^{{{b}}}"


def fmt_pow(x, sig=2):
    """Scientific notation ``a\\times10^{b}`` at ``sig`` significant digits."""
    core = fmt_pow_bare(x, sig)
    if core.startswith("\\textbf") or core == "0":
        return core
    return rf"\ensuremath{{{core}}}"


def fmt_asym(med, p16, p84, sig=2):
    """``(a^{+e}_{-e})\\times10^{b}`` with a common power of ten."""
    if any(v is None or not math.isfinite(v) for v in (med, p16, p84)):
        return r"\textbf{??}"
    b = int(math.floor(math.log10(abs(med)))) if med != 0 else 0
    s = 10.0 ** b
    m, ep, em = med / s, (p84 - med) / s, (med - p16) / s
    d = _decimals_for(min(abs(ep), abs(em)), sig)
    core = f"{m:.{d}f}^{{+{ep:.{d}f}}}_{{-{em:.{d}f}}}"
    if b == 0:
        return rf"\ensuremath{{{core}}}"
    return rf"\ensuremath{{\left({core}\right)\times 10^{{{b}}}}}"


def fmt_pm(val, err, sig=2):
    if any(v is None or not math.isfinite(v) for v in (val, err)):
        return r"\textbf{??}"
    d = _decimals_for(err, sig)
    return rf"\ensuremath{{{val:.{d}f} \pm {err:.{d}f}}}"


def fmt_signed_pct(x, dec=1):
    if x is None or not math.isfinite(x):
        return r"\textbf{??}"
    return rf"\ensuremath{{{x:+.{dec}f}\,\%}}"


# --------------------------------------------------------------------------- #
# input parsing
# --------------------------------------------------------------------------- #
def load_results_json(log_dir):
    p = Path(log_dir) / "info" / "results.json"
    if not p.exists():
        warnings.warn(f"missing {p}")
        return {}
    return json.loads(p.read_text())


def load_equal_samples(log_dir):
    p = Path(log_dir) / "chains" / "equal_weighted_post.txt"
    if not p.exists():
        warnings.warn(f"missing {p}")
        return None, None
    with open(p) as fh:
        header = fh.readline().split()
    data = np.loadtxt(p, skiprows=1)
    return header, np.atleast_2d(data)


def load_sensitivity(log_dir):
    p = Path(log_dir) / "sensitivity.json"
    if not p.exists():
        warnings.warn(f"missing {p}")
        return []
    return json.loads(p.read_text())


def load_verify(log_dir):
    """verify() audit numbers written by a v07 run (empty for older runs)."""
    p = Path(log_dir) / "verify.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


_RE_NOISE = re.compile(r"\\ln\s*f|^ln_?f[_\d]|^f_?sigma", re.I)
_RE_TAU = re.compile(r"\\tau|^tau$", re.I)


def param_layout(paramnames):
    """Index bookkeeping for the sampled parameters from their names.

    Returns dict with ``i_off`` (offset indices, dx/dy interleaved),
    ``i_lnf`` (noise-scale indices; empty for a v06 run), ``i_tau``,
    ``n_epochs`` and ``has_noise``.  Names are matched by pattern (TeX
    ``$\\ln f_3$`` or raw ``lnf_3``; ``$\\tau$`` or ``tau``), everything
    else counts as an offset; the epoch count is cross-checked between
    the two blocks.
    """
    names = list(paramnames or [])
    if not names:
        return None
    i_lnf = [i for i, n in enumerate(names) if _RE_NOISE.search(n)]
    i_tau = [i for i, n in enumerate(names) if _RE_TAU.search(n)]
    if len(i_tau) != 1:
        raise ValueError(f"cannot identify tau' among {names}")
    i_tau = i_tau[0]
    i_off = [i for i in range(len(names)) if i not in i_lnf and i != i_tau]
    if len(i_off) % 2:
        raise ValueError(f"odd number of offset parameters in {names}")
    n_epochs = len(i_off) // 2 + 1
    if i_lnf and len(i_lnf) != n_epochs:
        raise ValueError(f"{len(i_lnf)} noise scales but {n_epochs} epochs "
                         f"implied by the offsets: {names}")
    return {"i_off": i_off, "i_lnf": i_lnf, "i_tau": i_tau,
            "n_epochs": n_epochs, "has_noise": bool(i_lnf),
            "ndim": len(names)}


def offsets_from_theta(theta, layout):
    """(N, 2) per-epoch offsets (epoch 0 = gauge zero) from a theta vector."""
    off = np.asarray(theta, dtype=float)[layout["i_off"]].reshape(-1, 2)
    return np.concatenate([[[0.0, 0.0]], off], axis=0)


def load_model_card(log_dir):
    p = Path(log_dir) / "MODEL_CARD.txt"
    out = {}
    if not p.exists():
        warnings.warn(f"missing {p}")
        return out
    txt = p.read_text()
    for key, pat in (("lam", r"lam\s*=\s*([0-9.eE+\-]+)"),
                     ("sigma_g", r"sigma_g\s*=\s*([0-9.eE+\-]+)"),
                     ("P", r"P\s*=\s*(\d+)"),
                     ("offset_halfwidth", r"uniform on \+/-\s*([0-9.]+)\s*px"),
                     ("tau_hi", r"flat on \[0,\s*([0-9.eE+\-]+)\]"),
                     ("n_epochs", r"N\s*=\s*(\d+)\s*epochs")):
        m = re.search(pat, txt)
        if m:
            out[key] = float(m.group(1))
    m = re.search(r"log-uniform\s+f in \[([0-9.eE+\-]+),\s*([0-9.eE+\-]+)\]",
                  txt)
    if m:
        out["noise_scale_range"] = [float(m.group(1)), float(m.group(2))]
    m = re.search(r"n_i\s*=\s*([0-9,\s]+);", txt)
    if m:
        out["n_weighted_per_epoch"] = [int(v) for v in
                                       m.group(1).replace(" ", "").split(",")
                                       if v]
    out["has_noise"] = "noise scales f_i" in txt
    return out


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def reflected_kde(x, grid, bw=None, weights=None):
    """Gaussian KDE on tau >= 0 with reflection at the boundary."""
    x = np.asarray(x, dtype=float)
    w = (np.full(x.size, 1.0 / x.size) if weights is None
         else np.asarray(weights, dtype=float) / np.sum(weights))
    if bw is None:
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        bw = 0.9 * min(x.std(), iqr / 1.34) * x.size ** -0.2
        bw = max(bw, 1e-12)
    pts = np.concatenate([x, -x])
    ww = np.concatenate([w, w])
    d = (grid[:, None] - pts[None, :]) / bw
    return (np.exp(-0.5 * d * d) * ww[None, :]).sum(1) / (
        bw * math.sqrt(2 * math.pi))


def _wquantile(x, q, weights):
    i = np.argsort(x)
    cw = np.cumsum(np.asarray(weights, dtype=float)[i])
    cw /= cw[-1]
    return np.interp(q, cw, np.asarray(x)[i])


def _sigma_equiv(cred):
    """Two-sided Gaussian sigma equivalent of a credibility level."""
    lo, hi = 0.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if math.erf(mid / math.sqrt(2.0)) < cred:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _zero_exclusion(grid, dens):
    """Credibility of the smallest HPD region that excludes tau=0:
    posterior mass where the density exceeds the density at zero."""
    _tz = getattr(np, "trapezoid", None) or np.trapz
    norm = _tz(dens, grid)
    mask = dens >= dens[0]
    cred = float(_tz(np.where(mask, dens, 0.0), grid) / norm)
    return min(cred, 1.0 - 1e-12)


def tau_statistics(tau, tau_hi, tau_soft=1e-2):
    q = np.quantile(tau, [0.01, 0.05, 0.10, PCT16, 0.5, PCT84,
                          0.95, 0.99, 0.9973])
    grid = np.linspace(0.0, max(tau.max(), 5 * tau.std()), 2048)
    dens = reflected_kde(tau, grid)
    imode = int(np.argmax(dens))
    out = {
        "ll99": float(q[0]), "ll95": float(q[1]), "ll90": float(q[2]),
        "p16": float(q[3]), "median": float(q[4]), "p84": float(q[5]),
        "ul95": float(q[6]), "ul99": float(q[7]), "ul997": float(q[8]),
        "mean": float(tau.mean()), "std": float(tau.std()),
        "mode": float(grid[imode]),
        "dens0": float(dens[0]), "dens_mode": float(dens[imode]),
    }
    out["dens0_rel"] = out["dens0"] / out["dens_mode"]
    # Savage-Dickey at tau'=0 against the flat prior density 1/tau_hi;
    # positive lnB_sd favors the point null.  B01 scales linearly with
    # the prior width: even odds at tau_hi = 1/dens0.
    out["lnB_sd"] = (math.log(out["dens0"] * tau_hi)
                     if out["dens0"] > 0 else float("nan"))
    out["tau_hi_even_odds"] = (1.0 / out["dens0"]
                               if out["dens0"] > 0 else float("inf"))
    # HPD exclusion of tau=0 and its Gaussian sigma equivalent
    out["zero_excl_cred"] = _zero_exclusion(grid, dens)
    out["zero_excl_sigma"] = _sigma_equiv(out["zero_excl_cred"])
    # prior-shape robustness: reweight the flat-prior samples to the
    # soft-log prior pi(tau) ~ 1/(tau + tau_soft) and recompute the tail
    w = 1.0 / (tau + tau_soft)
    dens_s = reflected_kde(tau, grid, weights=w)
    i_s = int(np.argmax(dens_s))
    out["soft"] = {
        "tau_soft": tau_soft,
        "median": float(_wquantile(tau, 0.5, w)),
        "ll95": float(_wquantile(tau, 0.05, w)),
        "ul95": float(_wquantile(tau, 0.95, w)),
        "dens0_rel": float(dens_s[0] / dens_s[i_s]),
        "zero_excl_cred": _zero_exclusion(grid, dens_s),
    }
    out["soft"]["zero_excl_sigma"] = _sigma_equiv(
        out["soft"]["zero_excl_cred"])
    return out


# --------------------------------------------------------------------------- #
# optional goodness of fit via the likelihood module
# --------------------------------------------------------------------------- #
def _find_star_center(model):
    """Center of 180-deg symmetry of the disk model (peak of the correlation
    with its own rotation == self-convolution), sub-pixel refined."""
    m = np.nan_to_num(np.asarray(model, dtype=float))
    m -= np.median(m)
    ny, nx = m.shape
    conv = np.fft.irfft2(np.fft.rfft2(m, (2 * ny, 2 * nx)) ** 2,
                         (2 * ny, 2 * nx))
    k0, k1 = np.unravel_index(np.argmax(conv), conv.shape)

    def sub(a, b, c):
        d = a - 2 * b + c
        return 0.0 if d == 0 else 0.5 * (a - c) / d

    dy = sub(conv[k0 - 1, k1], conv[k0, k1], conv[k0 + 1, k1])
    dx = sub(conv[k0, k1 - 1], conv[k0, k1], conv[k0, k1 + 1])
    return ((k1 + dx) / 2.0, (k0 + dy) / 2.0)  # (x, y)


def evaluate_run(module_path, theta_med, tau_samples, extras, layout):
    """Goodness of fit + per-epoch sightline values via the analysis module
    (requires the data paths inside the module to resolve).

    All chi^2 / MSR numbers come in two flavours: under the NOMINAL noise
    model (the fixed weights of \\autoref{eq:perpixel_noise}; ``msr``) and
    under the fitted per-epoch noise scales (the likelihood's units;
    ``msr_scaled``, ~1 by construction at the posterior median up to the
    template degrees of freedom).  For a v06 run (no noise scales) the two
    coincide.  The template is the one fitted WITH the scales in both
    cases.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("ssu", module_path)
    ssu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ssu)
    import torch

    obs, disks, masks, var, nominal, q_cut = ssu.load_native_data()
    ll = ssu.LogLikelihood(obs, disks, masks, var, nominal,
                           noise_scale=layout["has_noise"])
    if ll.ndim != layout["ndim"]:
        raise ValueError(f"module expects {ll.ndim} parameters, run has "
                         f"{layout['ndim']}")
    theta = torch.as_tensor(np.asarray(theta_med, dtype=float),
                            dtype=torch.float64).view(1, -1)
    parts = ll.reference_parts(theta)
    n_ep = ll.n_weighted.detach().cpu().numpy()          # weighted px / epoch
    nw = int(n_ep.sum())
    chi2_s = float(parts["chi2"][0])
    chi2_n = float(parts["chi2_nominal"][0])
    chi2_ep_s = parts["chi2_epoch"][0].detach().cpu().numpy()
    chi2_ep_n = parts["chi2_nominal_epoch"][0].detach().cpu().numpy()
    f_med = parts["f"][0].detach().cpu().numpy()
    out = {"chi2_med": chi2_n, "chi2_scaled_med": chi2_s,
           "n_weighted_pix": nw, "n_weighted_per_epoch": n_ep.tolist(),
           "msr": chi2_n / nw, "msr_scaled": chi2_s / nw,
           "msr_per_epoch": (chi2_ep_n / n_ep).tolist(),
           "msr_scaled_per_epoch": (chi2_ep_s / n_ep).tolist(),
           "f_at_median": f_med.tolist()}
    print("per-epoch MSR at the posterior median -- nominal noise model:",
          ", ".join(f"{v:.2f}" for v in out["msr_per_epoch"]))
    print("                                      -- fitted noise scales:",
          ", ".join(f"{v:.2f}" for v in out["msr_scaled_per_epoch"]))
    print("noise scales f_i at the posterior median:",
          ", ".join(f"{v:.2f}" for v in f_med),
          "(sqrt of nominal per-epoch MSR:",
          ", ".join(f"{math.sqrt(v):.2f}" for v in out["msr_per_epoch"]) + ")")

    # per-epoch chi^2 penalty of forcing tau'=0 (template re-profiled in
    # both hypotheses; noise scales held at their medians, so the numbers
    # are in the likelihood's SCALED units).  A real common-tau signal
    # distributes the penalty across epochs increasing with (differential)
    # dhat; a single dominating epoch points at a systematic instead.
    th32 = torch.as_tensor(np.asarray(theta_med, dtype=float),
                           dtype=torch.float32).view(1, -1)
    th0 = th32.clone()
    th0[0, layout["i_tau"]] = 0.0
    _, chi_hat = ll.error_map(th32, reference=True)
    _, chi_0 = ll.error_map(th0, reference=True)
    per_hat = (chi_hat[0].double() ** 2).sum(dim=(1, 2))
    per_0 = (chi_0[0].double() ** 2).sum(dim=(1, 2))
    dchi2 = (per_0 - per_hat).detach().cpu().numpy()
    out["dchi2_tau0_total"] = float(dchi2.sum())

    # noise diagnostics: per-epoch background pedestal measured OUTSIDE
    # the aperture (a differential pedestal across epochs projects onto
    # tau'), and MSR in model-flux terciles (rising MSR with flux means
    # the shot-noise term is underestimated; flat >1 means the floor or
    # correlated speckle is).  The tercile split is reported under both
    # noise models: a per-epoch SCALAR f_i cannot absorb a flux-dependent
    # misestimate, so a residual trend in the scaled terciles is the
    # signal that the shot-noise term itself is off.
    y_np = ll.y.detach().cpu().numpy()
    w_np = ll.w.detach().cpu().numpy()
    bg_mean, bg_std = [], []
    for i in range(y_np.shape[0]):
        sel0 = (w_np[i] == 0) & np.isfinite(y_np[i])
        bg_mean.append(float(y_np[i][sel0].mean()))
        bg_std.append(float(y_np[i][sel0].std()))
    out["bg_mean_per_epoch"] = bg_mean
    out["bg_std_per_epoch"] = bg_std
    print("per-epoch background pedestal (outside aperture, disk-"
          "subtracted) [counts/s/px]:")
    print("  mean:", ", ".join(f"{v:+.2e}" for v in bg_mean))
    print("  std :", ", ".join(f"{v: .2e}" for v in bg_std))
    print(f"  epoch-to-epoch pedestal spread: "
          f"{np.std(bg_mean):.2e} (differential -> projects onto tau')")

    try:
        res_map, chi_map_s = ll.error_map(th32, reference=True, scaled=True)
        _, chi_map_n = ll.error_map(th32, reference=True, scaled=False)
        m_np = (ll.y - res_map[0]).detach().cpu().numpy()
        sel = w_np > 0
        flux = m_np[sel]
        edges = np.quantile(flux, [0.0, 1 / 3, 2 / 3, 1.0])
        for tag, cm in (("msr_flux_terciles", chi_map_n),
                        ("msr_scaled_flux_terciles", chi_map_s)):
            chis = (cm[0].detach().cpu().numpy() ** 2)[sel]
            out[tag] = [float(chis[(flux >= edges[k])
                                   & (flux <= edges[k + 1])].mean())
                        for k in range(3)]
        print("MSR in model-flux terciles (low, mid, high) -- nominal:",
              ", ".join(f"{v:.2f}" for v in out["msr_flux_terciles"]),
              "; scaled:",
              ", ".join(f"{v:.2f}" for v in out["msr_scaled_flux_terciles"]))
    except Exception as exc:                          # noqa: BLE001
        warnings.warn(f"flux-binned MSR skipped: {exc}")

    offsets = offsets_from_theta(theta_med, layout)
    centers = np.asarray(q_cut) + offsets
    dhat = ll.dhat.detach().cpu().numpy()
    hw = dhat.shape[-1]
    yy, xx = np.mgrid[0:hw, 0:hw]

    pos = ssu.load_position_guess()
    q_full = np.stack([pos["x_est"].to_numpy(), pos["y_est"].to_numpy()],
                      axis=1)[: dhat.shape[0]] + offsets
    star = _find_star_center(
        ssu.load_disk_model().detach().cpu().numpy())
    scale = extras["pixel_scale_arcsec"]
    dist_pc = 1000.0 / extras["parallax_mas"]
    kappa = 3.0 * extras["q_ext"] / (4.0 * extras["grain_rho_gcc"]
                                     * extras["grain_a_cm"])
    tq = np.quantile(np.asarray(tau_samples, dtype=float), [0.5, PCT84, 0.95])
    sight = []
    for i in range(dhat.shape[0]):
        sel = np.hypot(xx - centers[i, 0], yy - centers[i, 1]) <= 2.0
        d_i = float(dhat[i][sel].mean())
        r_px = float(np.hypot(*(q_full[i] - np.asarray(star))))
        tau95 = float(tq[2] * d_i)
        sigma = tau95 / kappa
        sight.append({
            "epoch": f"EP{i}", "dhat": d_i, "r_px": r_px,
            "r_arcsec": r_px * scale, "r_au": r_px * scale * dist_pc,
            "tau_med": float(tq[0] * d_i), "tau_p84": float(tq[1] * d_i),
            "tau_ul95": tau95, "sigma_ul95_cgs": sigma,
            "log_sigma_ul95": float(np.log10(sigma * 1.0e4)),
            "dchi2_tau0": float(dchi2[i]),
            "f_med": float(f_med[i]),
            "msr_nominal": float(chi2_ep_n[i] / n_ep[i]),
            "msr_scaled": float(chi2_ep_s[i] / n_ep[i]),
            "n_weighted": int(n_ep[i]),
        })
    out["sightlines"] = sight
    out["star_xy_fullframe"] = [float(star[0]), float(star[1])]
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
VARIANT_SUFFIX = [
    ("run prior", "Base"), ("lam x 0.1", "LamTenth"),
    ("lam x 0.333", "LamThird"), ("lam x 3", "LamThree"),
    ("lam x 10", "LamTen"),
]   # the soft-log tau prior variant of sensitivity.json is not reported in the paper


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log_dir")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--extras", default=None)
    ap.add_argument("--prefix", default="Vg")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--module", default=None)
    args = ap.parse_args(argv)

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir) if args.out_dir else log_dir / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    extras = dict(DEFAULT_EXTRAS)
    user_extras = {}
    if args.extras:
        user_extras = json.loads(Path(args.extras).read_text())
        extras.update(user_extras)

    res = load_results_json(log_dir)
    header, samples = load_equal_samples(log_dir)
    sens = load_sensitivity(log_dir)
    card = load_model_card(log_dir)
    audit = load_verify(log_dir)

    # audit-derived numbers: verify.json (v07 runs) and the f32 check
    # stored in the run-prior row of sensitivity.json take precedence over
    # the v06 defaults; an explicit --extras file overrides both.
    for k_src, k_dst in VERIFY_KEYS.items():
        if k_src in audit and k_dst not in user_extras:
            extras[k_dst] = audit[k_src]
    for row in sens:
        if row.get("variant", "").startswith("run prior") \
                and "f32_max_abs_dlnl" in row \
                and "f32_max_dlnl" not in user_extras:
            extras["f32_max_dlnl"] = row["f32_max_abs_dlnl"]
    # priors as actually used by the run (MODEL_CARD) beat the defaults
    for k in ("offset_halfwidth", "tau_hi", "noise_scale_range"):
        if k in card and k not in user_extras:
            extras[k] = card[k]

    P = args.prefix
    M = {}                                    # macro name (no prefix) -> tex
    raw = {"extras": extras}
    missing = []

    def put(name, tex, value=None):
        M[name] = tex
        if value is not None:
            raw[name] = value
        if r"\textbf{??}" in str(tex):
            missing.append(name)

    # ---- geometry / model size / parameter layout -------------------------
    paramnames = res.get("paramnames") or (header if header else [])
    layout = param_layout(paramnames) if paramnames else None
    ndim = layout["ndim"] if layout else None
    neps = layout["n_epochs"] if layout else None
    has_noise = bool(layout and layout["has_noise"])
    if layout:
        raw["layout"] = layout
        print(f"parameter layout: {ndim} sampled parameters, {neps} epochs, "
              + ("per-epoch noise scales" if has_noise
                 else "fixed nominal weights (no noise scales)"))
    put("NDim", str(ndim) if ndim else r"\textbf{??}", ndim)
    put("NEpochs", str(neps) if neps else r"\textbf{??}", neps)
    put("NNoiseScales", str(neps if has_noise else 0) if neps else
        r"\textbf{??}", neps if has_noise else 0)
    npix = int(card.get("P", 0)) or None
    put("NTemplatePix", str(npix) if npix else r"\textbf{??}", npix)

    # ---- template prior --------------------------------------------------
    put("Lam", fmt_plain(card.get("lam"), 3), card.get("lam"))
    put("SigmaG", fmt_plain(card.get("sigma_g"), 2), card.get("sigma_g"))
    put("LamRel", fmt_pow(extras["lam_rel_init"], 1), extras["lam_rel_init"])
    put("TauHi", fmt_plain(extras["tau_hi"], 1), extras["tau_hi"])
    put("OffsetBound", fmt_plain(extras["offset_halfwidth"], 2),
        extras["offset_halfwidth"])
    nsr = extras.get("noise_scale_range")
    put("NoiseScaleLo", fmt_plain(nsr[0], 2) if nsr else r"\textbf{??}",
        nsr[0] if nsr else None)
    put("NoiseScaleHi", fmt_plain(nsr[1], 2) if nsr else r"\textbf{??}",
        nsr[1] if nsr else None)

    # ---- run statistics --------------------------------------------------
    logz = res.get("logz")
    logzerr = res.get("logzerr") or res.get("logzerr_bs")
    put("LogZ", fmt_pm(logz, logzerr) if logz is not None else r"\textbf{??}",
        logz)
    put("LogZErr", fmt_plain(logzerr, 2) if logzerr else r"\textbf{??}",
        logzerr)
    dlz = res.get("logzerr_bs") or logzerr
    put("Dlogz", fmt_plain(dlz, 2) if dlz else r"\textbf{??}", dlz)
    ess = res.get("ess")
    put("ESS", f"{ess:.0f}" if ess else r"\textbf{??}", ess)
    ncall = res.get("ncall")
    put("NCalls", fmt_pow(ncall, 2) if ncall else r"\textbf{??}", ncall)
    niter = res.get("niter")
    put("NIter", fmt_pow(niter, 2) if niter else r"\textbf{??}", niter)
    put("NLive", str(extras["n_live"]), extras["n_live"])

    # ---- audit-derived (extras) -----------------------------------------
    put("VolumeSlope", fmt_plain(extras["volume_slope"], 3),
        extras["volume_slope"])
    put("VolumeTilt", fmt_plain(abs(extras["volume_slope"]) / 2.0, 3),
        abs(extras["volume_slope"]) / 2.0)
    vsl = extras.get("volume_slope_lnf")
    put("VolumeSlopeLnF", fmt_plain(vsl, 3) if vsl is not None
        else r"\textbf{??}", vsl)
    put("NModes", str(int(extras["n_modes_above_lam"])),
        extras["n_modes_above_lam"])
    put("HCond", fmt_plain(extras["h_cond_unridged"], 2),
        extras["h_cond_unridged"])
    put("FmaxDlnL", fmt_pow(extras["f32_max_dlnl"], 2),
        extras["f32_max_dlnl"])
    nsi = extras.get("noise_scale_identity")
    put("NoiseScaleIdentity", fmt_pow(nsi, 1) if nsi is not None
        else r"\textbf{??}", nsi)

    # ---- goodness of fit + sightlines (optional module evaluation) -------
    sightlines = None
    if args.evaluate and args.module and res.get("posterior") \
            and samples is not None and layout:
        try:
            ev = evaluate_run(args.module, res["posterior"]["median"],
                              samples[:, layout["i_tau"]], extras, layout)
            sightlines = ev.pop("sightlines", None)
            raw["star_xy_fullframe"] = ev.pop("star_xy_fullframe", None)
            extras.update(ev)
        except Exception as exc:                       # noqa: BLE001
            warnings.warn(f"--evaluate failed: {exc}")
    put("MSR", fmt_plain(extras["msr"], 3) if extras["msr"] else
        r"\textbf{??}", extras["msr"])
    put("MSRSqrt", fmt_plain(math.sqrt(extras["msr"]), 2) if extras["msr"]
        else r"\textbf{??}",
        math.sqrt(extras["msr"]) if extras["msr"] else None)
    put("MSRScaled", fmt_plain(extras["msr_scaled"], 3)
        if extras.get("msr_scaled") else r"\textbf{??}",
        extras.get("msr_scaled"))
    put("NWeightedPix", f"{int(extras['n_weighted_pix'])}" if
        extras["n_weighted_pix"] else r"\textbf{??}",
        extras["n_weighted_pix"])
    put("ChiSqMed", fmt_plain(extras["chi2_med"], 4) if extras["chi2_med"]
        else r"\textbf{??}", extras["chi2_med"])
    put("ChiSqScaledMed", fmt_plain(extras["chi2_scaled_med"], 4)
        if extras.get("chi2_scaled_med") else r"\textbf{??}",
        extras.get("chi2_scaled_med"))
    for tag, key in (("MSRTercileLow", 0), ("MSRTercileMid", 1),
                     ("MSRTercileHigh", 2)):
        terc = extras.get("msr_flux_terciles")
        put(tag, fmt_plain(terc[key], 2) if terc else r"\textbf{??}",
            terc[key] if terc else None)
        terc_s = extras.get("msr_scaled_flux_terciles")
        put(tag + "Scaled", fmt_plain(terc_s[key], 2) if terc_s
            else r"\textbf{??}", terc_s[key] if terc_s else None)

    # ---- tau posterior ---------------------------------------------------
    if samples is not None and layout:
        tau = samples[:, layout["i_tau"]]
        ts = tau_statistics(tau, extras["tau_hi"])
        raw["tau"] = ts
        put("TauMed", fmt_asym(ts["median"], ts["p16"], ts["p84"]),
            ts["median"])
        put("TauMedVal", fmt_plain(ts["median"], 2), ts["median"])
        put("TauPSixteen", fmt_pow(ts["p16"], 2), ts["p16"])
        put("TauPEightyFour", fmt_pow(ts["p84"], 2), ts["p84"])
        put("TauULNinetyFive", fmt_pow(ts["ul95"], 2), ts["ul95"])
        put("TauULNinetyFiveVal", fmt_plain(ts["ul95"], 2), ts["ul95"])
        put("TauULThreeSig", fmt_pow(ts["ul997"], 2), ts["ul997"])
        put("TauMean", fmt_pow(ts["mean"], 2), ts["mean"])
        put("TauStd", fmt_pow(ts["std"], 2), ts["std"])
        put("TauMAP", fmt_pow(ts["mode"], 2), ts["mode"])
        put("TauDensZeroRel", fmt_plain(ts["dens0_rel"], 2), ts["dens0_rel"])
        put("TauSavageDickey", fmt_plain(ts["lnB_sd"], 2), ts["lnB_sd"])
    else:
        for k in ("TauMed", "TauMedVal", "TauPSixteen", "TauPEightyFour",
                  "TauULNinetyFive", "TauULNinetyFiveVal", "TauULThreeSig",
                  "TauMean", "TauStd", "TauMAP", "TauDensZeroRel",
                  "TauSavageDickey"):
            put(k, r"\textbf{??}")

    ml = res.get("maximum_likelihood", {}).get("point")
    put("TauML", fmt_pow(ml[layout["i_tau"]], 2) if (ml and layout)
        else r"\textbf{??}", ml[layout["i_tau"]] if (ml and layout) else None)
    if samples is not None and layout:
        put("TauLLNinetyFive", fmt_pow(ts["ll95"], 2), ts["ll95"])
        put("TauLLNinety", fmt_pow(ts["ll90"], 2), ts["ll90"])
        put("TauZeroExclCred", fmt_plain(100 * ts["zero_excl_cred"], 3),
            ts["zero_excl_cred"])
        put("TauZeroExclSigma", fmt_plain(ts["zero_excl_sigma"], 2),
            ts["zero_excl_sigma"])
        put("TauBEvenPriorWidth", fmt_plain(ts["tau_hi_even_odds"], 2),
            ts["tau_hi_even_odds"])
        put("TauSoftMed", fmt_pow(ts["soft"]["median"], 2),
            ts["soft"]["median"])
        put("TauSoftLLNinetyFive", fmt_pow(ts["soft"]["ll95"], 2),
            ts["soft"]["ll95"])
        put("TauSoftULNinetyFive", fmt_pow(ts["soft"]["ul95"], 2),
            ts["soft"]["ul95"])
        put("TauSoftZeroExclSigma", fmt_plain(ts["soft"]["zero_excl_sigma"],
                                              2),
            ts["soft"]["zero_excl_sigma"])
    else:
        for name in ("TauLLNinetyFive", "TauLLNinety", "TauZeroExclCred",
                     "TauZeroExclSigma", "TauBEvenPriorWidth", "TauSoftMed",
                     "TauSoftLLNinetyFive", "TauSoftULNinetyFive",
                     "TauSoftZeroExclSigma"):
            put(name, r"\textbf{??}")

    # ---- column-density conversion (comparison figure) -------------------
    kappa = 3.0 * extras["q_ext"] / (4.0 * extras["grain_rho_gcc"]
                                     * extras["grain_a_cm"])
    put("KappaExt", fmt_pow(kappa, 3), kappa)
    peak = None
    if samples is not None and layout:
        sigma_peak = ts["ul95"] / kappa
        peak = {"tau_ul95": ts["ul95"], "sigma_ul95_cgs": sigma_peak,
                "log_sigma_ul95": math.log10(sigma_peak * 1.0e4)}
        put("SigmaULPeak", fmt_pow(sigma_peak, 2), sigma_peak)
        put("LogSigmaULPeak", fmt_plain(peak["log_sigma_ul95"], 3),
            peak["log_sigma_ul95"])
    else:
        put("SigmaULPeak", r"\textbf{??}")
        put("LogSigmaULPeak", r"\textbf{??}")

    # ---- offsets ---------------------------------------------------------
    post = res.get("posterior", {})
    med = post.get("median")
    std = post.get("stdev")
    if med and std and layout:
        off_med = np.asarray(med, dtype=float)[layout["i_off"]]
        off_std = np.asarray(std, dtype=float)[layout["i_off"]]
        raw["offsets"] = {"median": off_med.tolist(),
                          "stdev": off_std.tolist()}
        put("OffsetSigMin", fmt_plain(off_std.min(), 2), float(off_std.min()))
        put("OffsetSigMax", fmt_plain(off_std.max(), 2), float(off_std.max()))
        put("OffsetAbsMax", fmt_plain(np.abs(off_med).max(), 2),
            float(np.abs(off_med).max()))
        rows = []
        for i in range(len(off_med) // 2):
            dx, sx = off_med[2 * i], off_std[2 * i]
            dy, sy = off_med[2 * i + 1], off_std[2 * i + 1]
            dd = max(_decimals_for(sx), _decimals_for(sy))
            rows.append(
                f"EP{i + 1} & ${dx:.{dd}f} \\pm {sx:.{dd}f}$ & "
                f"${dy:.{dd}f} \\pm {sy:.{dd}f}$ \\\\")
        table_offsets = "\n".join([
            r"\begin{table}",
            r"\caption{Posterior registration offsets.}",
            r"\label{tab:offsets}",
            r"\centering",
            r"\begin{tabular}{lcc}",
            r"\hline\hline",
            r"Epoch & $\delta x$ [px] & $\delta y$ [px] \\",
            r"\hline", *rows, r"\hline",
            r"\end{tabular}",
            r"\tablefoot{$\delta_{i}$ relative to the nominal galaxy positions "
            r"(epoch-0 gauge), in STIS pixels: medians with posterior "
            r"standard deviations.}",
            r"\end{table}", ""])
        (out_dir / "table_offsets.tex").write_text(table_offsets)
    else:
        for k in ("OffsetSigMin", "OffsetSigMax", "OffsetAbsMax"):
            put(k, r"\textbf{??}")

    # ---- per-epoch noise scales (v07) -----------------------------------
    # f_i = exp(ln f_i) from the equal-weighted samples: median with the
    # 16--84 interval per epoch, plus summary macros.  The per-epoch
    # nominal MSR (from --evaluate) is listed next to f_i^2: the two agree
    # when the template degrees of freedom are negligible, and their
    # ratio is the honest statement of how far the fixed noise model was
    # off in that epoch.
    noise_macros = ("NoiseScaleMin", "NoiseScaleMax", "NoiseScaleMean",
                    "NoiseScaleGeoMean", "NoiseScaleSigMax",
                    "NoiseScaleRelSigMax", "NoiseScaleLnSpread")
    if has_noise and samples is not None:
        lnf_s = samples[:, layout["i_lnf"]]
        f_s = np.exp(lnf_s)
        f_q = np.quantile(f_s, [PCT16, 0.5, PCT84], axis=0)   # (3, N)
        lnf_med = np.median(lnf_s, axis=0)
        lnf_std = lnf_s.std(axis=0)
        f_med, f_lo, f_hi = f_q[1], f_q[1] - f_q[0], f_q[2] - f_q[1]
        f_sig = 0.5 * (f_lo + f_hi)
        raw["noise_scale"] = {
            "f_median": f_med.tolist(), "f_p16": f_q[0].tolist(),
            "f_p84": f_q[2].tolist(), "lnf_median": lnf_med.tolist(),
            "lnf_std": lnf_std.tolist(),
            "msr_per_epoch": extras.get("msr_per_epoch"),
            "msr_scaled_per_epoch": extras.get("msr_scaled_per_epoch"),
            "n_weighted_per_epoch": (extras.get("n_weighted_per_epoch")
                                     or card.get("n_weighted_per_epoch")),
        }
        put("NoiseScaleMin", fmt_plain(f_med.min(), 2), float(f_med.min()))
        put("NoiseScaleMax", fmt_plain(f_med.max(), 2), float(f_med.max()))
        put("NoiseScaleMean", fmt_plain(f_med.mean(), 2), float(f_med.mean()))
        put("NoiseScaleGeoMean", fmt_plain(math.exp(lnf_med.mean()), 2),
            float(math.exp(lnf_med.mean())))
        put("NoiseScaleSigMax", fmt_plain(f_sig.max(), 2), float(f_sig.max()))
        put("NoiseScaleRelSigMax", fmt_plain(100 * lnf_std.max(), 2),
            float(lnf_std.max()))
        put("NoiseScaleLnSpread", fmt_plain(lnf_med.max() - lnf_med.min(), 2),
            float(lnf_med.max() - lnf_med.min()))
        for i in range(f_med.shape[0]):
            put(f"NoiseScaleEP{chr(ord('A') + i)}",
                fmt_asym(f_med[i], f_q[0][i], f_q[2][i]),
                float(f_med[i]))
        msr_ep = extras.get("msr_per_epoch")
        msr_ep_s = extras.get("msr_scaled_per_epoch")
        n_ep = (extras.get("n_weighted_per_epoch")
                or card.get("n_weighted_per_epoch"))
        rows = []
        for i in range(f_med.shape[0]):
            dd = _decimals_for(min(f_lo[i], f_hi[i]))
            cells = [f"EP{i}",
                     f"${f_med[i]:.{dd}f}^{{+{f_hi[i]:.{dd}f}}}"
                     f"_{{-{f_lo[i]:.{dd}f}}}$",
                     f"${f_med[i] ** 2:.2f}$"]
            cells.append(f"${msr_ep[i]:.2f}$" if msr_ep else r"\dots")
            cells.append(f"${msr_ep_s[i]:.2f}$" if msr_ep_s else r"\dots")
            cells.append(f"{int(n_ep[i])}" if n_ep else r"\dots")
            rows.append(" & ".join(cells) + r" \\")
        table_noise = "\n".join([
            r"\begin{table}",
            r"\caption{Per-epoch noise scales and goodness of fit.}",
            r"\label{tab:noise}",
            r"\centering",
            r"\begin{tabular}{lccccc}",
            r"\hline\hline",
            r"Epoch & $f_{i}$ & $f_{i}^{2}$ & MSR$_{i}$ & "
            r"MSR$_{i}/f_{i}^{2}$ & $n_{i}$ \\",
            r"\hline", *rows, r"\hline",
            r"\end{tabular}",
            r"\tablefoot{$f_{i}$: posterior median and 16--84\,\% interval; "
            r"MSR$_{i}$: mean squared standardised residual of epoch $i$ at "
            r"the posterior median under the nominal noise model of "
            r"\autoref{eq:perpixel_noise}, and MSR$_{i}/f_{i}^{2}$ under the "
            r"fitted scales; $n_{i}$: number of weighted pixels.}",
            r"\end{table}", ""])
        (out_dir / "table_noise.tex").write_text(table_noise)
    else:
        for k in noise_macros:
            put(k, r"\textbf{??}")
        if neps:
            for i in range(neps):
                put(f"NoiseScaleEP{chr(ord('A') + i)}", r"\textbf{??}")

    # ---- prior sensitivity ----------------------------------------------
    base_ul = None
    for row in sens:
        if row["variant"].startswith("run prior"):
            base_ul = row["tau_95"]
    srow_tex = []
    for row in sens:
        suf = next((s for key, s in VARIANT_SUFFIX
                    if row["variant"].startswith(key)), None)
        if suf is None:
            continue
        ul = row["tau_95"]
        put(f"{suf}UL", fmt_pow(ul, 2), ul)
        if base_ul and suf != "Base":
            shift = 100.0 * (ul / base_ul - 1.0)
            put(f"{suf}ULShift", fmt_signed_pct(shift), shift)
        if "dlnZ_vs_run" in row:
            put(f"{suf}DlnZ", rf"\ensuremath{{{row['dlnZ_vs_run']:+.1f}}}",
                row["dlnZ_vs_run"])
        put(f"{suf}ESS", f"{row['ess']:.0f}", row["ess"])
        label = {"Base": r"run prior (flat $\tau'$, $\lambda$)",
                 "LamTenth": r"$\lambda/10$",
                 "LamThird": r"$\lambda/3$", "LamThree": r"$3\lambda$",
                 "LamTen": r"$10\lambda$",
                 "SoftLog": r"soft-log $\tau'$ prior"}[suf]
        dz = (f"${row['dlnZ_vs_run']:+.1f}$" if "dlnZ_vs_run" in row
              else r"\dots")
        srow_tex.append(f"{label} & ${fmt_pow_bare(ul, 3)}$ & {dz} & "
                        f"{row['ess']:.0f} \\\\")
    if srow_tex:
        table_sens = "\n".join([
            r"\begin{table}",
            r"\caption{Prior sensitivity of the optical-depth limit.}",
            r"\label{tab:sensitivity}",
            r"\centering",
            r"\begin{tabular}{lccc}",
            r"\hline\hline",
            r"Variant & $\tau'_{95}$ & $\Delta\ln\mathcal{Z}$ & ESS \\",
            r"\hline", *srow_tex, r"\hline",
            r"\end{tabular}",
            r"\tablefoot{Computed from the single nested-sampling run by "
            r"importance reweighting (\autoref{app:sensitivity}). ESS is the "
            r"raw Kish effective sample size of the importance weights over "
            r"the $16\,384$ points resampled (with replacement) from the "
            r"weighted posterior, whose own effective sample size is "
            r"\VgESS{}; the quantiles of the $\lambda$ rungs use "
            r"Rao--Blackwellised weights.}",
            r"\end{table}", ""])
        (out_dir / "table_sensitivity.tex").write_text(table_sens)

    # ---- comparison-figure values ---------------------------------------
    comp = {
        "assumptions": {k: extras[k] for k in
                        ("q_ext", "grain_rho_gcc", "grain_a_cm",
                         "pixel_scale_arcsec", "parallax_mas")},
        "kappa_ext_cm2_g": kappa,
        "tau_hi": extras["tau_hi"],
        "noise_scale": has_noise,
        "noise_scale_range": extras.get("noise_scale_range"),
        "peak": peak,
        "sightlines": sightlines,
    }
    (out_dir / "comparison_values.json").write_text(json.dumps(comp, indent=2))
    if sightlines:
        raw["sightlines"] = sightlines
        rows = [f"{s['epoch']} & {s['r_au']:.0f} & {s['dhat']:.2f} & "
                f"${fmt_pow_bare(s['tau_ul95'], 2)}$ & "
                f"${fmt_pow_bare(s['sigma_ul95_cgs'], 2)}$ \\\\"
                for s in sightlines]
        table_sl = "\n".join([
            r"\begin{table}",
            r"\caption{Per-epoch sightlines and optical-depth limits.}",
            r"\label{tab:sightlines}",
            r"\centering",
            r"\begin{tabular}{lcccc}",
            r"\hline\hline",
            r"Epoch & $r$ [au] & $\hat{D}$ & $\tau_{i,95}$ & "
            r"$\Sigma_{95}$ [g\,cm$^{-2}$] \\",
            r"\hline", *rows, r"\hline",
            r"\end{tabular}",
            r"\tablefoot{$r$: projected stellocentric radius of the galaxy; "
            r"$\hat{D}$: local normalised disk surface brightness (mean "
            r"within 2\,px); $\tau_{i,95}$: $95\%$ upper limit on the "
            r"sightline optical depth $\tau_{i} = \tau^{\prime}\hat{D}$; "
            r"$\Sigma_{95}$: implied column density for the grain "
            r"assumptions of \autoref{eq:kappa_ext}, along the line of sight "
            r"(multiply by $\cos i = 0.94$ for the vertical column).}",
            r"\end{table}", ""])
        (out_dir / "table_sightlines.tex").write_text(table_sl)

    # ---- write macro file ------------------------------------------------
    lines = [
        "% " + 70 * "-",
        "%   Values computed by paper_values.py -- do not edit by hand.",
        f"%   Source run: {log_dir}",
        "% " + 70 * "-", "",
        "% " + 70 * "-",
        "%   Optical depth (tau' = optical depth at the disk surface-",
        "%   brightness peak of the transit region)",
        "% " + 70 * "-",
    ]

    order = [
        ("TauMed", "posterior median with 16--84 interval"),
        ("TauMedVal", "posterior median, plain"),
        ("TauPSixteen", "15.9th percentile"),
        ("TauPEightyFour", "84.1st percentile"),
        ("TauULNinetyFive", "95% upper limit"),
        ("TauULNinetyFiveVal", "95% upper limit, plain"),
        ("TauULThreeSig", "99.73% upper limit"),
        ("TauMean", "posterior mean"), ("TauStd", "posterior std"),
        ("TauMAP", "posterior mode (reflected KDE)"),
        ("TauML", "maximum-likelihood value"),
        ("TauDensZeroRel", "posterior density at 0 / density at mode"),
        ("TauSavageDickey", "ln Savage--Dickey ratio at tau'=0 (flat prior)"),
        ("SECTION:Detection statistics", None),
        ("TauLLNinetyFive", "one-sided 95% LOWER credible bound"),
        ("TauLLNinety", "one-sided 90% LOWER credible bound"),
        ("TauZeroExclCred", "HPD credibility [%] at which tau'=0 is excluded"),
        ("TauZeroExclSigma", "Gaussian sigma equivalent of the above"),
        ("TauBEvenPriorWidth", "prior width tau_hi giving even SD odds"),
        ("TauSoftMed", "median under the soft-log tau prior (reweighted)"),
        ("TauSoftLLNinetyFive", "95% lower bound, soft-log prior"),
        ("TauSoftULNinetyFive", "95% upper bound, soft-log prior"),
        ("TauSoftZeroExclSigma", "zero-exclusion sigma, soft-log prior"),
        ("SECTION:Template prior and model", None),
        ("Lam", "template prior precision lambda"),
        ("SigmaG", "per-pixel template prior std [cts/s]"),
        ("LamRel", "empirical-Bayes scale of lambda"),
        ("TauHi", "flat tau' prior ceiling"),
        ("OffsetBound", "offset prior half-width [px] (from MODEL_CARD)"),
        ("NoiseScaleLo", "log-uniform noise-scale prior, lower bound"),
        ("NoiseScaleHi", "log-uniform noise-scale prior, upper bound"),
        ("NDim", "sampled parameters"), ("NEpochs", "epochs"),
        ("NNoiseScales", "per-epoch noise-scale parameters (0 = fixed weights)"),
        ("NTemplatePix", "template pixels P"),
        ("SECTION:Per-epoch noise scales f_i (posterior)", None),
        ("NoiseScaleMin", "smallest per-epoch median f_i"),
        ("NoiseScaleMax", "largest per-epoch median f_i"),
        ("NoiseScaleMean", "mean over epochs of the median f_i"),
        ("NoiseScaleGeoMean", "geometric mean over epochs of the median f_i"),
        ("NoiseScaleSigMax", "largest 68% half-width of f_i"),
        ("NoiseScaleRelSigMax", "largest posterior std of ln f_i [%]"),
        ("NoiseScaleLnSpread", "max - min of the median ln f_i"),
        ("SECTION:Run statistics", None),
        ("LogZ", "ln evidence (see text for caveats)"),
        ("LogZErr", "ln evidence uncertainty"),
        ("Dlogz", "achieved evidence uncertainty (bootstrap)"),
        ("ESS", "posterior effective sample size"),
        ("NCalls", "likelihood evaluations"),
        ("NIter", "nested-sampling iterations"),
        ("NLive", "live points"),
        ("SECTION:Registration", None),
        ("OffsetSigMin", "smallest offset posterior std [px]"),
        ("OffsetSigMax", "largest offset posterior std [px]"),
        ("OffsetAbsMax", "largest |median| offset [px]"),
        ("SECTION:Goodness of fit (nominal noise model unless 'Scaled')", None),
        ("ChiSqMed", "chi^2 at the posterior median, nominal weights"),
        ("ChiSqScaledMed", "chi^2 at the posterior median, fitted scales"),
        ("NWeightedPix", "weighted pixels"),
        ("MSR", "mean squared standardized residual, nominal noise model"),
        ("MSRSqrt", "sqrt(MSR): single-scale equivalent of the nominal MSR"),
        ("MSRScaled", "mean squared standardized residual, fitted scales"),
        ("MSRTercileLow", "nominal MSR, lowest model-flux tercile"),
        ("MSRTercileMid", "nominal MSR, middle model-flux tercile"),
        ("MSRTercileHigh", "nominal MSR, highest model-flux tercile"),
        ("MSRTercileLowScaled", "scaled MSR, lowest model-flux tercile"),
        ("MSRTercileMidScaled", "scaled MSR, middle model-flux tercile"),
        ("MSRTercileHighScaled", "scaled MSR, highest model-flux tercile"),
        ("SECTION:Audit-derived (verify)", None),
        ("VolumeSlope", "d ln det(H+lam I)/d tau' on the data"),
        ("VolumeTilt", "volume-term tilt of lnL per unit tau'"),
        ("VolumeSlopeLnF", "d ln det(H+lam I)/d ln f_0 on the data"),
        ("NModes", "template modes above the lambda floor"),
        ("HCond", "condition number of unridged H"),
        ("FmaxDlnL", "max |f32 - f64| lnL over the posterior"),
        ("NoiseScaleIdentity", "verify(): |rescaled-weights instance - "
                               "marginal at ln f| (implementation check)"),
        ("SECTION:Column-density conversion", None),
        ("KappaExt", "extinction opacity kappa_ext [cm^2 g^-1]"),
        ("SigmaULPeak", "95% Sigma limit at the SB peak [g cm^-2]"),
        ("LogSigmaULPeak", "log10(Sigma_95 x 1e4 g cm^-2) at the peak"),
        ("SECTION:Prior sensitivity (single-run reweighting)", None),
    ]
    for key, s in VARIANT_SUFFIX:
        order += [(f"{s}UL", f"95% UL, {key}")]
        if s != "Base":
            order += [(f"{s}ULShift", f"shift of the 95% UL, {key}"),
                      (f"{s}DlnZ", f"Delta lnZ vs run, {key}")]
        order += [(f"{s}ESS", f"raw ESS, {key}")]
    if neps:
        order += [("SECTION:Per-epoch noise scales (median, 16--84)", None)]
        order += [(f"NoiseScaleEP{chr(ord('A') + i)}", f"f_{i} of epoch {i}")
                  for i in range(neps)]

    for name, desc in order:
        if name.startswith("SECTION:"):
            lines += ["", "% " + 70 * "-", f"%   {name[8:]}",
                      "% " + 70 * "-"]
            continue
        if name not in M:
            put(name, r"\textbf{??}")
        lines.append(f"%   * \\{P}{name:24s} -- {desc}")
        lines.append(f"\\newcommand{{\\{P}{name}}}{{{M[name]}}}")
    (out_dir / "paper_values.tex").write_text("\n".join(lines) + "\n")
    (out_dir / "paper_values.json").write_text(
        json.dumps(raw, indent=2, default=float))

    print(f"wrote {out_dir}/paper_values.tex  ({len(M)} macros)")
    if missing:
        print("WARNING -- placeholders (\\textbf{??}) for: "
              + ", ".join(sorted(set(missing))))
    return 0


if __name__ == "__main__":
    sys.exit(main())