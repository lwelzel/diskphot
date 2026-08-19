"""Verification, identifiability and inspection diagnostics.

Entry points behind ``diskphot verify``, ``diskphot roundtrip``,
``diskphot inspect-data`` and ``diskphot inspect-shifts``:

* **verify** -- machine-checks of every numerical claim the method makes:
  exactness of the shift adjoint and of the squared-tap normal-matrix
  diagonal, agreement of the PCG template solve with a dense direct solve,
  agreement of the autograd (envelope-theorem) gradient with finite
  differences, float32-vs-float64 consistency, and exact noiseless recovery
  (chi-square compatible with zero on noise-free synthetic data).  Writes
  ``verify.json``; every check carries a pass/fail against an explicit
  tolerance.
* **roundtrip** -- inject a known template, offsets and tau with
  :mod:`diskphot.synthetic` and recover them with the same fitting code the
  science run uses; writes recovery plots and a JSON summary.
* **nuisance_spectrum** -- identifiability diagnostics: the eigenvalue
  spectrum (condition number) of the template normal matrix on its
  informative support, and the finite-difference Hessian of the profiled
  log-likelihood with the tau--offset correlation coefficients.
* **inspect-data / inspect-shifts** -- field geometry, forward-model grids
  and per-epoch offset likelihood landscapes.

Everything here is diagnostic; the science numbers come from
:mod:`diskphot.inference` and :mod:`diskphot.calibrate`.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Rectangle

from . import constants
from .config import CGConfig, DataConfig, FitConfig, ProfileConfig, Variant
from .data import EpochData, load_full_frames, make_ellipse_mask, ring_profile, variance_map
from .galaxy import galaxy_image
from .model import JointLogLikelihood
from .synthetic import SyntheticTruth, empirical_template, simulate_epochs
from .transforms import KERNELS, shift2d

log = logging.getLogger(__name__)

__all__ = [
    "batched_logl",
    "make_synthetic_epoch_data",
    "nuisance_spectrum",
    "plot_field_overview",
    "plot_forward_grid",
    "plot_offset_recovery",
    "plot_shift_scan",
    "plot_tau_profile_slice",
    "plot_template_comparison",
    "run_roundtrip",
    "run_verify",
    "theta_from_truth",
    "theta_initial",
]


# --------------------------------------------------------------------------- #
# Parameter-vector helpers
# --------------------------------------------------------------------------- #
def theta_initial(model: JointLogLikelihood) -> torch.Tensor:
    """Initial-guess parameters: zero offsets, tau = 0 (transmission = 1)."""
    return model.mask.new_zeros(model.n_params)


def theta_from_truth(model: JointLogLikelihood, truth: SyntheticTruth, r: int = 0) -> torch.Tensor:
    """Assemble the true parameter vector of synthetic realization ``r``."""
    if model.variant not in (Variant.COMMON, Variant.PER_EPOCH, Variant.NULL):
        raise ValueError(f"no truth-vector mapping for variant {model.variant}")
    ref = model.mask
    theta = ref.new_zeros(model.n_params)
    offsets = torch.as_tensor(truth.offsets_xy[r, 1:], dtype=ref.dtype, device=ref.device)
    theta[: model.n_offset_params] = offsets.reshape(-1)
    if model.n_tau_params:
        tau = torch.as_tensor(truth.tau, dtype=ref.dtype, device=ref.device)
        if tau.numel() == 1 and model.n_tau_params > 1:
            tau = tau.expand(model.n_tau_params)
        theta[model.n_offset_params :] = tau[: model.n_tau_params]
    return theta


def batched_logl(
    model: JointLogLikelihood, thetas: torch.Tensor, chunk: int = 256,
    tight: bool = True,
) -> torch.Tensor:
    """Evaluate the profiled log-likelihood for many parameter vectors.

    Certified (tight-tolerance) by default: diagnostics compare absolute
    likelihood values, so the CG solve must run to ``cg.tol_chi2``.
    """
    outs = []
    with torch.inference_mode():
        for start in range(0, thetas.shape[0], chunk):
            outs.append(model(thetas[start : start + chunk], tight=tight))
    return torch.cat(outs)


# --------------------------------------------------------------------------- #
# Standalone synthetic data (no FITS files needed)
# --------------------------------------------------------------------------- #
def make_synthetic_epoch_data(
    n_epochs: int = 4,
    size: int = 26,
    seed: int = 0,
    sigma_bg: float = 3.0e-3,
    exptime_s: float = 8000.0,
    disk_level: float = 0.05,
    dtype: torch.dtype = torch.float64,
) -> EpochData:
    """Small fully synthetic :class:`EpochData` for tests and ``verify``.

    A parametric galaxy on a smooth synthetic "disk" background with
    randomized sub-pixel epoch positions, mirroring the geometry conventions
    of :func:`diskphot.data.load_epoch_data` (native grids, nominal model
    shifts, fixed per-epoch elliptical weights).  Observations are the
    *noise-free* forward model at ``tau = 0``; add noise / extinction with
    :func:`diskphot.synthetic.simulate_epochs`.
    """
    gen = torch.Generator().manual_seed(seed)
    c = (size - 1) / 2.0
    # Sub-pixel galaxy positions within half a pixel of the cutout centre,
    # exactly as the integer-crop loader guarantees.
    q = c + (torch.rand((n_epochs, 2), generator=gen, dtype=dtype) - 0.5)
    q[0] = c  # reference epoch centred (any value works; this eases reading)
    nominal = q - q[0]

    ys, xs = torch.meshgrid(
        torch.arange(size, dtype=dtype), torch.arange(size, dtype=dtype), indexing="ij"
    )
    # Per-epoch amplitude sweep: as in the real geometry, the galaxy sits at
    # substantially different disk surface brightnesses in different epochs
    # (aperture-mean D-hat contrast ~2x), which is what identifies a common
    # tau' against the free template amplitude.
    disk = torch.stack(
        [
            disk_level
            * (0.45 + 1.1 * i / max(n_epochs - 1, 1))
            * (1.0 + 0.4 * torch.sin(0.15 * xs + 0.9 * i) + 0.01 * ys)
            for i in range(n_epochs)
        ]
    ).clamp(min=1.0e-4)
    disk_hat = disk / float(disk.max())

    template = galaxy_image(size, dtype=dtype).to(dtype)
    shifted = shift2d(template.expand(n_epochs, size, size).contiguous(), nominal.to(dtype))
    obs = disk + shifted

    valid = torch.ones(n_epochs, size, size, dtype=dtype)
    mask = torch.stack(
        [
            make_ellipse_mask(10.5, 6.5, 35.0, size, center_xy=(float(q[i, 0]), float(q[i, 1])))
            for i in range(n_epochs)
        ]
    ).to(dtype)
    sigma = np.full(n_epochs, sigma_bg)
    expt = np.full(n_epochs, exptime_s)
    variance = variance_map(obs, sigma, expt, gain=1.0)

    return EpochData(
        observations=obs.contiguous(),
        disk=disk.contiguous(),
        disk_hat=disk_hat.contiguous(),
        variance=variance,
        valid=valid,
        mask=mask,
        nominal_shift_xy=nominal.contiguous(),
        galaxy_xy_cutout=q,
        galaxy_xy_full=q.numpy() + 100.0,
        crop_origin_xy=np.full((n_epochs, 2), 100, dtype=int),
        star_xy_full=(0.0, 0.0),
        disk_peak_sb=float(disk.max()),
        disk_peak_radius_px=50.0,
        sigma_bg=sigma,
        exptimes_s=expt,
        dates=tuple(f"synthetic-{i}" for i in range(n_epochs)),
        config=DataConfig(),
    )


# --------------------------------------------------------------------------- #
# verify: machine checks of the numerical claims
# --------------------------------------------------------------------------- #
def _check_adjoint(size: int, seed: int, dtype: torch.dtype) -> dict:
    """<S_s x, y> == <x, S_{-s} y> for every kernel (exact transpose)."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn((8, size, size), generator=gen, dtype=dtype)
    y = torch.randn((8, size, size), generator=gen, dtype=dtype)
    s = 3.0 * (torch.rand((8, 2), generator=gen, dtype=dtype) - 0.5)
    out = {}
    for kernel in KERNELS:
        lhs = (shift2d(x, s, kernel=kernel) * y).sum(dim=(-2, -1))
        rhs = (x * shift2d(y, -s, kernel=kernel)).sum(dim=(-2, -1))
        scale = lhs.abs().clamp(min=1.0)
        out[kernel] = float(((lhs - rhs).abs() / scale).max())
    return out


def _check_squared_diag(size: int, seed: int, dtype: torch.dtype) -> float:
    """Squared-tap back-projection equals the exact diagonal of S^T W S."""
    gen = torch.Generator().manual_seed(seed)
    w = torch.rand((1, size, size), generator=gen, dtype=dtype)
    s = 2.0 * (torch.rand((1, 2), generator=gen, dtype=dtype) - 0.5)
    fast = shift2d(w, -s, squared=True)[0]
    npix = size * size
    eye = torch.eye(npix, dtype=dtype).reshape(npix, size, size)
    cols = shift2d(eye, s.expand(npix, 2))
    exact = (w[0].reshape(-1)[None, :] * cols.reshape(npix, npix) ** 2).sum(dim=1)
    return float((fast.reshape(-1) - exact).abs().max())


def run_verify(
    cfg: FitConfig | None = None,
    data: EpochData | None = None,
    out_path: Path | None = None,
    seed: int = 0,
) -> dict:
    """Run every numerical check and return a pass/fail report.

    With no data given, a small standalone synthetic set is used, so the
    command runs anywhere.  Tolerances are explicit in the report.
    """
    cfg = cfg or FitConfig()
    torch.manual_seed(seed)
    if data is None:
        data = make_synthetic_epoch_data(seed=seed)
    data64 = data.to("cpu", torch.float64)

    report: dict = {"checks": {}}

    def record(name: str, value: float, tol: float, note: str = "") -> None:
        report["checks"][name] = {
            "value": value, "tol": tol, "pass": bool(value <= tol), "note": note,
        }

    # 1. Adjoint exactness (all kernels).
    adj = _check_adjoint(20, seed, torch.float64)
    for kernel, err in adj.items():
        record(f"adjoint_{kernel}", err, 1.0e-12,
               "relative <Sx,y> vs <x,S^-y> mismatch")

    # 2. Exact normal-matrix diagonal from squared taps.
    record("squared_tap_diagonal", _check_squared_diag(14, seed, torch.float64),
           1.0e-12, "squared-tap back-projection vs dense diag(S^T W S)")

    # 3. Noiseless exact recovery: chi^2(truth) ~ 0 on noise-free data.
    template, _ = empirical_template(data64, cfg.profile, cfg.cg)
    obs0, truth = simulate_epochs(
        data64, 5.0e-3, template, n_realizations=1, variant=Variant.COMMON,
        calib=cfg.calib, add_noise=False, seed=seed,
    )
    model = JointLogLikelihood(data64, Variant.COMMON, cfg.cg, observations=obs0)
    theta_true = theta_from_truth(model, truth).unsqueeze(0)
    chi2_true = float(-2.0 * batched_logl(model, theta_true)[0])
    record("noiseless_chi2_at_truth", chi2_true, 10.0 * cfg.cg.tol_chi2,
           "the model must reproduce its own noise-free draw exactly")

    # 4a. Certified direct solve vs an independently assembled/solved
    #     dense reference (column-by-column build, LAPACK gelsd
    #     least-squares) at a random parameter point.
    gen = torch.Generator().manual_seed(seed + 1)
    theta = theta_true[0].clone()
    theta[: model.n_offset_params] += 0.3 * (
        torch.rand(model.n_offset_params, generator=gen, dtype=torch.float64) - 0.5
    )
    theta = theta.unsqueeze(0)
    logl_direct = float(batched_logl(model, theta)[0])
    _, logl_dense = model.profile_dense(theta)
    record("direct_vs_lstsq_logl", abs(logl_direct - float(logl_dense[0])),
           cfg.cg.tol_chi2,
           "certified eigh pseudo-inverse vs columnwise gelsd least squares")

    # 4b. Loose search CG vs the certified solve (coarse sanity bound:
    #     catches divergence or wrong operators, not fine accuracy --
    #     search accuracy only shapes the optimizer trajectory).
    logl_loose = float(batched_logl(model, theta, tight=False)[0])
    record("loose_cg_vs_direct_logl", abs(logl_loose - logl_direct), 2.0,
           "search-mode CG sanity vs certified direct solve")

    # 5. Envelope gradient vs central finite differences.
    theta_g = theta[0].clone().requires_grad_(True)
    logl = model(theta_g.unsqueeze(0), tight=True)[0]
    (grad,) = torch.autograd.grad(logl, theta_g)
    eps = 1.0e-4
    fd = torch.zeros_like(grad)
    with torch.inference_mode():
        for j in range(model.n_params):
            tp = theta[0].clone(); tp[j] += eps
            tm = theta[0].clone(); tm[j] -= eps
            fd[j] = (
                model(tp.unsqueeze(0), tight=True)[0]
                - model(tm.unsqueeze(0), tight=True)[0]
            ) / (2 * eps)
    scale = grad.abs().clamp(min=1.0)
    record("envelope_gradient_vs_fd", float(((grad - fd).abs() / scale).max()),
           5.0e-3, "autograd (envelope) gradient vs central differences")

    # 6. float32 vs float64 consistency of the profiled lnL.
    data32 = data.to("cpu", torch.float32)
    model32 = JointLogLikelihood(
        data32, Variant.COMMON, cfg.cg,
        observations=obs0.to(torch.float32),
    )
    logl32 = float(batched_logl(model32, theta.to(torch.float32))[0])
    record("float32_vs_float64_logl", abs(logl32 - logl_direct),
           max(0.05, 10.0 * cfg.cg.tol_chi2),
           "dtype sensitivity of lnL_p at one parameter point")

    report["passed"] = all(c["pass"] for c in report["checks"].values())
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2))
    for name, c in report["checks"].items():
        log.info("verify %-28s %.3e (tol %.1e) %s",
                 name, c["value"], c["tol"], "PASS" if c["pass"] else "FAIL")
    return report


# --------------------------------------------------------------------------- #
# Identifiability: nuisance spectrum and tau--offset correlations
# --------------------------------------------------------------------------- #
def nuisance_spectrum(
    model: JointLogLikelihood,
    theta: torch.Tensor,
    fd_step_offset: float = 2.0e-2,
    fd_step_logtau: float = 5.0e-2,
    out_path: Path | None = None,
) -> dict:
    """Identifiability diagnostics at a parameter point.

    * Eigenvalue spectrum of the template normal matrix ``H`` on its
      informative support (dense build; condition number quantifies how
      well the free template is constrained, including near-half-pixel
      Nyquist softness).
    * Finite-difference Hessian of the *profiled* log-likelihood over the
      fitted parameters (offsets in pixels; ``log10 tau`` for the
      disk-scaled variants) and the implied correlation matrix; the maximum
      absolute tau--offset correlation measures the leakage criticized for
      unregistered fits.
    """
    theta = torch.atleast_2d(theta.to(model.mask))

    # Dense H spectrum on the support.
    delta, taus = model._split(theta)
    t = model.transmission(taus, 1)
    shifts = model.nominal.unsqueeze(0) + delta
    wt2 = model.w * t * t
    h, w = model.y.shape[-2:]
    npix = h * w
    diag = model._shift(wt2, -shifts, kernel=model.kernel, squared=True).sum(dim=1)
    act = (diag[0].reshape(-1) > 1.0e-12 * float(diag[0].max())).nonzero(as_tuple=True)[0]
    eye = torch.eye(npix, dtype=theta.dtype).reshape(npix, h, w)
    cols = [
        model._normal_apply(eye[int(k)].expand(1, h, w), shifts, wt2).reshape(-1)
        for k in act
    ]
    hsub = torch.stack(cols, dim=1)[act]
    evals = torch.linalg.eigvalsh(0.5 * (hsub + hsub.T)).cpu().numpy()
    cond = float(evals.max() / max(evals.min(), np.finfo(float).tiny))

    # Finite-difference Hessian of the profiled objective.
    n_off, n_tau = model.n_offset_params, model.n_tau_params
    use_log_tau = model.variant in (Variant.COMMON, Variant.PER_EPOCH) and n_tau > 0

    def to_internal(th: torch.Tensor) -> np.ndarray:
        u = th[0].cpu().numpy().copy()
        if use_log_tau:
            u[n_off:] = np.log10(np.clip(u[n_off:], 1.0e-12, None))
        return u

    def from_internal(u: np.ndarray) -> torch.Tensor:
        th = torch.as_tensor(u, dtype=theta.dtype, device=theta.device).clone()
        if use_log_tau:
            th[n_off:] = 10.0 ** th[n_off:]
        return th.unsqueeze(0)

    u0 = to_internal(theta)
    p = len(u0)
    steps = np.array(
        [fd_step_offset] * n_off + [fd_step_logtau] * (p - n_off)
    )

    def f(u: np.ndarray) -> float:
        return float(batched_logl(model, from_internal(u))[0])

    hess = np.empty((p, p))
    f0 = f(u0)
    for i in range(p):
        for j in range(i, p):
            hi, hj = steps[i], steps[j]
            if i == j:
                up = u0.copy(); up[i] += hi
                um = u0.copy(); um[i] -= hi
                hess[i, i] = (f(up) - 2.0 * f0 + f(um)) / hi**2
            else:
                upp = u0.copy(); upp[i] += hi; upp[j] += hj
                upm = u0.copy(); upm[i] += hi; upm[j] -= hj
                ump = u0.copy(); ump[i] -= hi; ump[j] += hj
                umm = u0.copy(); umm[i] -= hi; umm[j] -= hj
                hess[i, j] = hess[j, i] = (
                    f(upp) - f(upm) - f(ump) + f(umm)
                ) / (4.0 * hi * hj)

    fisher = -hess  # observed information of lnL_p
    try:
        cov = np.linalg.inv(fisher)
        d = np.sqrt(np.clip(np.diag(cov), np.finfo(float).tiny, None))
        corr = cov / np.outer(d, d)
        tau_off_corr = (
            float(np.abs(corr[n_off:, :n_off]).max()) if n_tau else float("nan")
        )
        info_pd = bool(np.all(np.linalg.eigvalsh(0.5 * (fisher + fisher.T)) > 0))
    except np.linalg.LinAlgError:  # pragma: no cover
        corr = np.full((p, p), np.nan)
        tau_off_corr = float("nan")
        info_pd = False

    out = {
        "normal_matrix": {
            "n_support": int(len(act)),
            "eig_min": float(evals.min()),
            "eig_max": float(evals.max()),
            "condition_number": cond,
        },
        "profiled_hessian": {
            "param_names": model.param_names,
            "internal_scales": ["px"] * n_off
            + (["log10"] * (p - n_off) if use_log_tau else ["linear"] * (p - n_off)),
            "correlation": corr.tolist(),
            "max_abs_tau_offset_correlation": tau_off_corr,
            "information_positive_definite": info_pd,
        },
    }
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(out, indent=2))
    log.info(
        "nuisance spectrum: cond(H) = %.3e on %d px; max |corr(tau, delta)| = %.3f",
        cond, len(act), tau_off_corr,
    )
    return out


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _asinh(img: np.ndarray, scale: float) -> np.ndarray:
    return np.arcsinh(np.nan_to_num(img) / max(scale, 1e-12))


def plot_field_overview(cfg: DataConfig, data: EpochData, out_path: Path) -> Path:
    """Full-frame context: epoch frame, co-aligned disk model, ring profile."""
    frames, disk_full, _ = load_full_frames(cfg)
    fh, fw = frames[0].shape
    star = data.star_xy_full
    scale = float(np.median(data.sigma_bg))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    ax_obs, ax_disk, ax_prof = axes

    ax_obs.imshow(_asinh(frames[0], 3 * scale), origin="lower", cmap="magma")
    ax_obs.set_title("EP0 frame (asinh stretch)", fontsize=10)
    ax_disk.imshow(_asinh(disk_full[:fh, :fw], 3 * scale), origin="lower", cmap="magma")
    ax_disk.set_title("disk model, co-aligned", fontsize=10)

    x0, y0 = cfg.region_origin_xy
    for ax in (ax_obs, ax_disk):
        ax.add_patch(
            Rectangle((x0, y0), cfg.region_size, cfg.region_size,
                      fill=False, ec="cyan", lw=1.0)
        )
        ax.plot(*star, marker="*", ms=12, mec="w", mfc="gold", ls="none")
        ax.add_patch(Circle(star, data.disk_peak_radius_px,
                            fill=False, ec="w", ls="--", lw=0.8))
        ax.add_patch(Circle(star, constants.DISK_MODEL_INNER_VALID_RADIUS_PX,
                            fill=False, ec="w", ls=":", lw=0.8))
        ax.set_xlim(0, fw)
        ax.set_ylim(0, fh)
    cmap = mpl.colormaps["viridis"]
    for i, (gx, gy) in enumerate(np.asarray(data.galaxy_xy_full)):
        ax_obs.plot(gx, gy, "+", color=cmap(i / max(data.n_epochs - 1, 1)), ms=8)
        ax_obs.annotate(str(i), (gx, gy), color="w", fontsize=7,
                        textcoords="offset points", xytext=(3, 3))

    radii, profile = ring_profile(disk_full, star, r_min=1.0, footprint_hw=(fh, fw))
    ax_prof.plot(radii, profile, color="k", lw=1)
    ax_prof.axvline(constants.DISK_MODEL_INNER_VALID_RADIUS_PX, color="0.5",
                    ls=":", lw=0.8, label="model valid limit")
    ax_prof.plot(data.disk_peak_radius_px, data.disk_peak_sb, "o", color="crimson",
                 label=f"ring peak ({data.disk_peak_sb:.3g} @ {data.disk_peak_radius_px:.0f} px)")
    galaxy_r = np.hypot(*(np.asarray(data.galaxy_xy_full) - np.asarray(star)).T)
    for i, r in enumerate(galaxy_r):
        ax_prof.axvline(r, color=cmap(i / max(data.n_epochs - 1, 1)), lw=0.6, alpha=0.7)
    ax_prof.set_xlabel("stellocentric radius [px]")
    ax_prof.set_ylabel("disk SB [counts/s/px]")
    ax_prof.set_title("azimuthal profile; vertical lines: galaxy radii", fontsize=10)
    ax_prof.legend(fontsize=8)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


def plot_forward_grid(
    data: EpochData,
    model: JointLogLikelihood,
    theta: torch.Tensor,
    out_path: Path,
    title: str = "",
    observations: torch.Tensor | None = None,
) -> Path:
    """Per-epoch grid: raw, disk, galaxy model, forward model, residual, chi.

    All panels are in each epoch's native frame; the per-epoch elliptical
    analysis aperture is outlined and the nominal galaxy position marked.
    """
    theta = theta.to(model.mask)
    maps = model.maps(theta)
    logl = float(batched_logl(model, torch.atleast_2d(theta))[0])
    valid = data.valid.cpu().numpy()
    n = data.n_epochs
    raw = (observations if observations is not None else data.observations)
    raw = raw.reshape(n, *raw.shape[-2:]).cpu().numpy()

    rows = {
        "raw": raw,
        "disk model": data.disk.cpu().numpy(),
        "galaxy model": maps["model_epoch"].cpu().numpy(),
        "forward model": (data.disk + maps["model_epoch"]).cpu().numpy(),
        "residual": maps["residual_epoch"].cpu().numpy() * valid,
        "chi": maps["chi_epoch"].cpu().numpy() * valid,
    }
    raw_hi = float(np.nanpercentile(rows["raw"], 99.8))
    r_lim = float(np.nanpercentile(np.abs(rows["residual"]), 99.5)) or 1.0e-6
    scales = {
        "raw": (0.0, raw_hi, "magma"),
        "disk model": (0.0, float(np.nanpercentile(rows["disk model"], 99.8)), "magma"),
        "galaxy model": (0.0, raw_hi, "magma"),
        "forward model": (0.0, raw_hi, "magma"),
        "residual": (-r_lim, r_lim, "RdBu_r"),
        "chi": (-5.0, 5.0, "RdBu_r"),
    }

    mask = data.mask.cpu().numpy()
    pos = data.galaxy_xy_cutout.cpu().numpy()
    fig, axes = plt.subplots(
        len(rows), n, figsize=(1.65 * n, 1.65 * len(rows)),
        constrained_layout=True, sharex=True, sharey=True, squeeze=False,
    )
    for r, (name, field) in enumerate(rows.items()):
        vmin, vmax, cmap = scales[name]
        im = None
        for i in range(n):
            im = axes[r, i].imshow(field[i], origin="lower",
                                   vmin=vmin, vmax=vmax, cmap=cmap)
            if name in ("residual", "chi"):
                axes[r, i].contour(mask[i], levels=[0.5], colors="k", linewidths=0.5)
            axes[r, i].plot(*pos[i], "+", color="cyan", ms=5, mew=0.8)
            if r == 0:
                axes[r, i].set_title(f"EP{i}", fontsize=8)
        axes[r, 0].set_ylabel(name, fontsize=8)
        fig.colorbar(im, ax=list(axes[r]), shrink=0.8, pad=0.005)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{title}  (lnL_p = {logl:.1f})".strip(), fontsize=10)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


def plot_shift_scan(
    model: JointLogLikelihood,
    theta: torch.Tensor,
    out_path: Path,
    halfwidth_px: float = 1.0,
    n_grid: int = 41,
) -> Path:
    """Per-epoch profiled log-likelihood landscapes over that epoch's offset."""
    theta = theta.to(model.mask).reshape(-1)
    n_ep = model.n_epochs
    grid = torch.linspace(-halfwidth_px, halfwidth_px, n_grid, dtype=theta.dtype,
                          device=theta.device)
    gy, gx = torch.meshgrid(grid, grid, indexing="ij")

    ncols = 4
    nrows = int(np.ceil((n_ep - 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.9 * nrows),
                             constrained_layout=True, squeeze=False)
    im = None
    for k, ep in enumerate(range(1, n_ep)):
        ix, iy = 2 * (ep - 1), 2 * (ep - 1) + 1
        thetas = theta.expand(n_grid * n_grid, -1).clone()
        thetas[:, ix] = theta[ix] + gx.reshape(-1)
        thetas[:, iy] = theta[iy] + gy.reshape(-1)
        logl = batched_logl(model, thetas).view(n_grid, n_grid).cpu().numpy()
        dlogl = logl - logl.max()

        ax = axes[k // ncols][k % ncols]
        extent = [
            float(theta[ix] - halfwidth_px), float(theta[ix] + halfwidth_px),
            float(theta[iy] - halfwidth_px), float(theta[iy] + halfwidth_px),
        ]
        im = ax.imshow(dlogl, origin="lower", extent=extent, cmap="viridis",
                       vmin=-12.0, vmax=0.0, aspect="equal")
        ax.contour(dlogl, levels=[-5.90, -3.09, -1.15], extent=extent,
                   colors="w", linewidths=0.7)
        ax.plot(float(theta[ix]), float(theta[iy]), "o", color="crimson", ms=4)
        ax.plot(0.0, 0.0, "+", color="w", ms=8, mew=1.2)
        ax.set_title(f"EP{ep}", fontsize=9)
        ax.set_xlabel(r"$\Delta x$ [px]", fontsize=8)
        ax.set_ylabel(r"$\Delta y$ [px]", fontsize=8)
    for k in range(n_ep - 1, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8,
                 label=r"$\Delta \ln \mathcal{L}_{\rm p}$")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


def plot_tau_profile_slice(
    model: JointLogLikelihood,
    theta: torch.Tensor,
    out_path: Path,
    tau_lo: float = 1.0e-6,
    tau_hi: float = 1.0,
    n_grid: int = 120,
    tau_true: float | None = None,
) -> Path:
    """lnL_p over a common tau, offsets held at ``theta`` (a slice, not the
    re-optimized profile curve of :func:`diskphot.inference.profile_curve`)."""
    if model.n_tau_params == 0:
        raise ValueError("model has no tau parameters")
    theta = theta.to(model.mask).reshape(-1)
    taus = torch.logspace(math.log10(tau_lo), math.log10(tau_hi), n_grid,
                          dtype=theta.dtype, device=theta.device)
    thetas = theta.expand(n_grid, -1).clone()
    thetas[:, model.n_offset_params :] = taus.unsqueeze(1)
    logl = batched_logl(model, thetas).cpu().numpy()
    tau_np = taus.cpu().numpy()

    fig, ax = plt.subplots(figsize=(5.4, 3.4), constrained_layout=True)
    ax.plot(tau_np, logl - logl.max(), color="k", lw=1.2)
    k = int(np.argmax(logl))
    ax.axvline(tau_np[k], color="crimson", ls="--", lw=1,
               label=rf"$\arg\max = {tau_np[k]:.2e}$")
    if tau_true is not None:
        ax.axvline(tau_true, color="tab:blue", ls=":", lw=1.2,
                   label=rf"injected $= {tau_true:.2e}$")
    ax.set_xscale("log")
    ax.set_ylim(-25, 1)
    ax.set_xlabel(r"$\tau^{\prime}$")
    ax.set_ylabel(r"$\Delta \ln \mathcal{L}_{\rm p}$")
    ax.legend(fontsize=8)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


def plot_offset_recovery(
    true_offsets: np.ndarray,
    recovered: np.ndarray,
    out_path: Path,
) -> Path:
    """Recovered vs true per-epoch offsets (epoch 0 pinned to zero)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), constrained_layout=True)
    for a, (ax_i, name) in enumerate(zip(axes, ("x", "y"))):
        t, r = true_offsets[1:, a], recovered[1:, a]
        lim = 1.1 * max(np.abs(np.concatenate([t, r])).max(), 0.1)
        ax_i.plot([-lim, lim], [-lim, lim], color="0.7", lw=0.8)
        ax_i.plot(t, r, "o", ms=5, color="k")
        for ep, (ti, ri) in enumerate(zip(t, r), start=1):
            ax_i.annotate(str(ep), (ti, ri), fontsize=7,
                          textcoords="offset points", xytext=(4, 2))
        rms = float(np.sqrt(np.mean((r - t) ** 2)))
        ax_i.set_title(rf"$\Delta {name}$: RMS error {rms:.3f} px", fontsize=9)
        ax_i.set_xlabel(f"true d{name} [px]")
        ax_i.set_ylabel(f"recovered d{name} [px]")
        ax_i.set_xlim(-lim, lim)
        ax_i.set_ylim(-lim, lim)
        ax_i.set_aspect("equal")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


def plot_template_comparison(
    injected: torch.Tensor,
    recovered: torch.Tensor,
    mask: torch.Tensor,
    out_path: Path,
) -> Path:
    """Injected vs recovered galaxy template and their (masked) difference."""
    inj = injected.cpu().numpy()
    rec = recovered.cpu().numpy()
    m = mask.cpu().numpy()
    if m.ndim == 3:
        m = m[0]
    diff = (rec - inj) * m
    vmax = float(np.nanpercentile(inj, 99.8))
    dmax = float(np.nanpercentile(np.abs(diff), 99.5)) or 1e-6

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), constrained_layout=True)
    for ax, img, title, kw in (
        (axes[0], inj, "injected template", dict(vmin=0, vmax=vmax, cmap="magma")),
        (axes[1], rec, "recovered template", dict(vmin=0, vmax=vmax, cmap="magma")),
        (axes[2], diff, "difference (masked)", dict(vmin=-dmax, vmax=dmax, cmap="RdBu_r")),
    ):
        im = ax.imshow(img, origin="lower", **kw)
        ax.contour(m, levels=[0.5], colors="k", linewidths=0.5)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #
def run_roundtrip(
    cfg: FitConfig,
    data: EpochData,
    tau: float = 1.0e-3,
    add_noise: bool = True,
    seed: int = 0,
    out_dir: Path = Path("outputs/dev"),
) -> dict:
    """Inject a known model and verify it is recovered by the science fitter.

    Uses the same code path as the observed fit (:func:`fit_mle` on the
    exactly profiled likelihood).  On *noise-free* data the recovered
    chi-square at the truth must be compatible with zero within the CG
    tolerance -- the model can reproduce its own draw exactly, a property
    the previous approximate profiling did not have.  Writes recovery plots
    and ``roundtrip_summary_*.json``.
    """
    from .inference import fit_mle  # local import to avoid a cycle

    if cfg.variant not in (Variant.COMMON, Variant.PER_EPOCH):
        raise ValueError("roundtrip supports the disk-scaled variants")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    template_inj, _ = empirical_template(data, cfg.profile, cfg.cg)
    obs, truth = simulate_epochs(
        data, tau, template=template_inj, n_realizations=1,
        variant=cfg.variant, calib=cfg.calib, add_noise=add_noise, seed=seed,
    )
    model = JointLogLikelihood(data, cfg.variant, cfg.cg, observations=obs)

    theta_true = theta_from_truth(model, truth)
    chi2_true = float(-2.0 * batched_logl(model, theta_true.unsqueeze(0))[0])
    logl_init = float(batched_logl(model, theta_initial(model).unsqueeze(0))[0])

    res = fit_mle(model, cfg.profile, seed=seed)
    theta_hat = torch.as_tensor(res.theta[0], dtype=model.mask.dtype)
    logl_hat = float(res.logl[0])
    n_off = model.n_offset_params
    rec_offsets = np.zeros_like(truth.offsets_xy[0])
    rec_offsets[1:] = res.theta[0, :n_off].reshape(-1, 2)
    rec_tau = res.theta[0, n_off:]

    maps = model.maps(theta_hat)
    w_np = model.w.cpu().numpy()
    sel = w_np > 0
    chi = maps["chi_epoch"].cpu().numpy()
    mssr = float(np.mean(chi[sel] ** 2))

    tag = f"tau{tau:.0e}_seed{seed}" + ("" if add_noise else "_noiseless")
    plot_forward_grid(
        data, model, theta_hat, out_dir / f"roundtrip_forward_{tag}.png",
        title=f"roundtrip {tag}, recovered point", observations=obs,
    )
    if cfg.variant is Variant.COMMON:
        plot_tau_profile_slice(
            model, theta_hat, out_dir / f"roundtrip_tau_slice_{tag}.pdf",
            tau_lo=cfg.profile.tau_grid_lo / 10.0, tau_hi=cfg.profile.tau_hi,
            tau_true=tau,
        )
    plot_offset_recovery(
        truth.offsets_xy[0], rec_offsets, out_dir / f"roundtrip_offsets_{tag}.pdf"
    )
    plot_template_comparison(
        template_inj, maps["template"], model.mask,
        out_dir / f"roundtrip_template_{tag}.png",
    )

    summary = {
        "variant": cfg.variant.value,
        "tau_injected": float(tau),
        "tau_recovered": [float(v) for v in np.atleast_1d(rec_tau)],
        "offset_rms_error_px": float(
            np.sqrt(np.mean((rec_offsets[1:] - truth.offsets_xy[0, 1:]) ** 2))
        ),
        "chi2_at_truth": chi2_true,
        "noiseless_exact": (not add_noise)
        and chi2_true <= 10.0 * cfg.cg.tol_chi2,
        "logl_initial": logl_init,
        "logl_recovered": logl_hat,
        # Chi-square distance between the recovered point and the injected
        # truth.  Near the sensitivity floor the tau' profile is flat, so a
        # recovered tau' far from the injected value can still be
        # likelihood-equivalent to it; the calibrated statistics (not point
        # estimates) carry the inference in that regime.
        "delta_chi2_recovered_vs_truth": float(-2.0 * logl_hat - chi2_true),
        "tau_likelihood_equivalent": bool(-2.0 * logl_hat - chi2_true < 0.1),
        "mean_squared_std_residual": mssr,
        "n_weighted_pixels": int(sel.sum()),
        "noise": add_noise,
        "seed": seed,
    }
    if not add_noise and not summary["noiseless_exact"]:
        log.warning(
            "noiseless roundtrip chi2 at truth = %.3e exceeds 10 x tol_chi2 = %.1e",
            chi2_true, 10.0 * cfg.cg.tol_chi2,
        )
    (out_dir / f"roundtrip_summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    log.info("roundtrip summary: %s", json.dumps(summary, indent=2))
    return summary
