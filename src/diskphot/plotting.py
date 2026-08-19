"""Figures.

Manuscript figures of the frequentist pipeline:

* :func:`profile_curve_plot` -- the observed profile-likelihood-ratio curve
  ``q(tau')`` with the simulation-calibrated critical curve(s) and the
  Neyman upper limit(s);
* :func:`detection_efficiency_plot` -- the injection power curve with
  Clopper--Pearson bands, the detection threshold context and the
  sensitivity;
* :func:`residual_panels` -- native-frame residual and chi maps at the MLE
  with the profiled template;
* :func:`corner_plot` -- corner view of the optional UltraNest *objective
  map* (a geometry diagnostic; the axes are the profiled objective's
  parameters, not posterior marginals of a Bayesian model).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import EpochData
from .model import JointLogLikelihood
from .results import ObservedFit

__all__ = [
    "corner_plot",
    "detection_efficiency_plot",
    "profile_curve_plot",
    "residual_panels",
]


# --------------------------------------------------------------------------- #
# Profile curve + calibrated limits
# --------------------------------------------------------------------------- #
def profile_curve_plot(
    fit: ObservedFit,
    summary: dict,
    out_path: Path,
) -> Path:
    """Observed ``q(tau')`` with the calibrated critical curves and limits.

    ``summary`` is the dict of :func:`diskphot.calibrate.analyze_calibration`
    run with the observed payload, so the critical curves and the reported
    upper limits are exactly those of the paper.
    """
    grid = np.asarray(summary["tau_grid"], dtype=float)
    q_obs = fit.q_curve()

    fig, ax = plt.subplots(figsize=(5.6, 3.8), constrained_layout=True)
    ax.plot(grid, q_obs, color="k", lw=1.4, marker="o", ms=3,
            label=r"observed $q(\tau^{\prime})$")
    colors = plt.cm.viridis(np.linspace(0.25, 0.75, max(len(summary["limits"]), 1)))
    for color, entry in zip(colors, summary["limits"].values()):
        crit = np.asarray(entry["critical_curve"], dtype=float)
        cl = entry["cl"]
        ax.plot(grid, crit, color=color, lw=1.0, ls="--",
                label=rf"$c_{{{100 * cl:.1f}\%}}(\tau^{{\prime}})$ (simulated)")
        ul = entry.get("upper_limit")
        if ul is not None and np.isfinite(ul):
            ax.axvline(ul, color=color, lw=1.0, ls=":")
            ax.annotate(
                rf"UL$_{{{100 * cl:.0f}\%}}$ = {ul:.2e}",
                xy=(ul, 0.97), xycoords=("data", "axes fraction"),
                rotation=90, va="top", ha="right", fontsize=7, color=color,
            )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau^{\prime}$ (optical depth at ring SB peak)")
    ax.set_ylabel(r"$q(\tau^{\prime}) = 2\,[\ln L_{\rm p}^{\max} - \ln L_{\rm p}(\tau^{\prime})]$")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=7, loc="upper left")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


def detection_efficiency_plot(
    summary: dict,
    out_path: Path,
) -> Path:
    """Injection power curve with Clopper--Pearson band and the sensitivity."""
    power = summary["power"]
    taus = np.asarray(power["tau_inj"], dtype=float)
    frac = np.asarray(power["frac"], dtype=float)
    lo = np.asarray(power["frac_lo"], dtype=float)
    hi = np.asarray(power["frac_hi"], dtype=float)

    fig, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
    ax.fill_between(taus, lo, hi, color="0.85",
                    label="Clopper–Pearson 95% band")
    ax.plot(taus, frac, color="k", lw=1.3, marker="o", ms=3,
            label=rf"$P(q_0 > c_\mathrm{{det}})$, "
                  rf"$\alpha_\mathrm{{det}}$ = {summary['alpha_det']:g}")
    ax.axhline(summary["power_target"], color="crimson", ls="--", lw=0.9)
    sens = summary.get("sensitivity")
    if sens is not None and np.isfinite(sens):
        ax.axvline(sens, color="crimson", ls=":", lw=1.0,
                   label=rf"sensitivity = {sens:.2e}")
    cons = summary.get("sensitivity_conservative")
    if cons is not None and np.isfinite(cons):
        ax.axvline(cons, color="crimson", ls="-.", lw=0.8, alpha=0.6,
                   label=rf"conservative = {cons:.2e}")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"injected $\tau^{\prime}$")
    ax.set_ylabel("detection probability")
    ax.legend(fontsize=7, loc="lower right")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


# --------------------------------------------------------------------------- #
# Residual panels at the MLE
# --------------------------------------------------------------------------- #
def residual_panels(
    fit: ObservedFit,
    data: EpochData,
    out_path: Path,
    device: str = "cpu",
) -> Path:
    """Native-frame model/residual/chi maps at the MLE plus the template."""
    cfg = fit.config
    model = JointLogLikelihood(data.to(device, data.mask.dtype), fit.variant, cfg.cg)
    theta_np = fit.theta_mle if fit.theta_mle is not None else fit.theta_null
    theta = torch.as_tensor(theta_np, dtype=data.mask.dtype, device=device)
    maps = model.maps(theta)
    mask = maps["mask"].cpu().numpy().astype(bool)  # (N, H, W)
    n = data.n_epochs

    fig, axes = plt.subplots(
        3, n + 1, figsize=(1.9 * (n + 1), 6.2), constrained_layout=True,
        sharex=True, sharey=True,
    )
    rows = ("model_epoch", "residual_epoch", "chi_epoch")
    labels = ("galaxy model", "residual", "chi")
    for r, (key, label) in enumerate(zip(rows, labels)):
        field = maps[key].cpu().numpy()
        sel = np.abs(field[mask])
        vmax = float(np.nanpercentile(sel, 99.5)) if sel.size else 1.0
        cmap, vmin = ("RdBu_r", -vmax) if r > 0 else ("viridis", 0.0)
        im = None
        for i in range(n):
            img = np.where(mask[i], field[i], np.nan)
            im = axes[r, i].imshow(img, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
            if r == 0:
                axes[r, i].set_title(f"EP{i}", fontsize=9)
        axes[r, 0].set_ylabel(label, fontsize=9)
        fig.colorbar(im, ax=list(axes[r, :n]), shrink=0.85, pad=0.01)
    tmpl = maps["template"].cpu().numpy()
    axes[0, -1].imshow(np.where(mask[0], tmpl, np.nan), origin="lower", cmap="viridis")
    axes[0, -1].set_title(r"$\hat G$ (template)", fontsize=9)
    for ax in axes[1:, -1]:
        ax.set_visible(False)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


# --------------------------------------------------------------------------- #
# Objective-map corner (explore diagnostic)
# --------------------------------------------------------------------------- #
def corner_plot(
    map_npz: Path | str,
    out_path: Path,
    tau_only: bool = False,
    truths: np.ndarray | None = None,
) -> Path:
    """Corner view of the UltraNest objective map (``diskphot explore``).

    The axes visualize the geometry of the *profiled objective* (offset--tau
    degeneracies, multimodality); they are not Bayesian posterior marginals.
    """
    import corner

    payload = np.load(Path(map_npz), allow_pickle=False)
    samples = np.asarray(payload["samples"], dtype=float)
    labels = [str(v) for v in payload["param_names"]]
    scales = [str(v) for v in payload["axes_scales"]]
    t = None if truths is None else np.asarray(truths, dtype=float).copy()
    for i, scale in enumerate(scales):
        if scale == "log":
            samples[:, i] = np.log10(np.clip(samples[:, i], 1e-300, None))
            labels[i] = rf"$\log_{{10}}$ {labels[i]}"
            if t is not None:
                t[i] = np.log10(max(t[i], 1e-300))
    if tau_only:
        cols = [i for i, name in enumerate(labels) if "tau" in name or "gamma" in name]
        samples = samples[:, cols]
        labels = [labels[c] for c in cols]
        if t is not None:
            t = t[cols]
    fig = corner.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".2f",
        color="black",
        truths=t,
        truth_color="crimson",
    )
    fig.suptitle("objective map (profiled likelihood; not a posterior)",
                 fontsize=10, y=1.001)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)
