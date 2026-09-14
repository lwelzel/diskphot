"""Figures.  matplotlib is imported inside each function (lazy)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _save(fig, out_path: Path | None):
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(fig)
    return fig


def profile_curve_plot(observed: dict, summary: dict, out_path: Path | None = None):
    """q(tau) with the calibrated critical curves and upper limits."""
    import matplotlib.pyplot as plt

    grid = np.asarray(observed["tau_curve_grid"], dtype=float)
    lnl = np.asarray(observed["logl_curve"], dtype=float)
    lnl_hat = max(float(observed["logl_mle"]), float(observed["logl_null"]), float(lnl.max()))
    q = np.clip(2.0 * (lnl_hat - lnl), 0.0, None)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, q, "k-", lw=1.5, label=r"observed $q(\tau')$")
    for cl, entry in summary.get("limits", {}).items():
        crit = np.asarray(entry["critical_curve"], dtype=float)
        ax.plot(grid, crit, "--", lw=1, label=f"critical curve ({float(cl):.1%})")
        ul = entry.get("upper_limit_observed")
        if ul is not None and np.isfinite(ul):
            ax.axvline(ul, color="C3", alpha=0.5, lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau'$")
    ax.set_ylabel(r"$q(\tau')$")
    ax.legend(fontsize=8)
    return _save(fig, out_path)


def detection_efficiency_plot(summary: dict, out_path: Path | None = None):
    import matplotlib.pyplot as plt

    rows = summary["detection_curve"]
    tau = np.array([r["tau_inj"] for r in rows])
    frac = np.array([r["frac"] for r in rows])
    lo = np.array([r["frac_lo"] for r in rows])
    hi = np.array([r["frac_hi"] for r in rows])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(tau, lo, hi, alpha=0.25, label="Clopper-Pearson 95%")
    ax.plot(tau, frac, "o-", ms=4)
    ax.axhline(0.95, color="k", ls=":", lw=1)
    sens = summary.get("sensitivity_tau")
    if sens is not None and np.isfinite(sens):
        ax.axvline(sens, color="C3", ls="--", lw=1, label=rf"$\tau'_{{95}}$ = {sens:.2e}")
    ax.set_xscale("log")
    ax.set_xlabel(r"injected $\tau'$")
    ax.set_ylabel(rf"P($q_0 > c_{{det}}$), $\alpha$={summary['alpha_det']:g}")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8)
    return _save(fig, out_path)


def residual_panels(maps: dict, out_path: Path | None = None):
    """Data-model standardized residuals per epoch."""
    import matplotlib.pyplot as plt

    chi = maps["chi_epoch"].detach().cpu().numpy()
    n = chi.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.4), squeeze=False)
    vmax = max(3.0, float(np.abs(chi).max()) * 0.8)
    for i, ax in enumerate(axes[0]):
        im = ax.imshow(chi[i], origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"EP{i}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label=r"$\chi$")
    return _save(fig, out_path)


def plot_field_overview(data, out_path: Path | None = None):
    import matplotlib.pyplot as plt

    obs = data.observations.cpu().numpy()
    msk = data.mask.cpu().numpy()
    n = obs.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(1.8 * n, 4.0), squeeze=False)
    vmax = np.percentile(obs, 99.5)
    for i in range(n):
        axes[0, i].imshow(obs[i], origin="lower", vmin=0, vmax=vmax)
        axes[0, i].contour(msk[i], levels=[0.5], colors="w", linewidths=0.6)
        q = data.galaxy_xy_cutout[i].cpu().numpy()
        axes[0, i].plot(q[0], q[1], "rx", ms=4)
        axes[1, i].imshow(data.disk_hat[i].cpu().numpy(), origin="lower")
        for ax in (axes[0, i], axes[1, i]):
            ax.set_xticks([]); ax.set_yticks([])
        axes[0, i].set_title(f"EP{i}", fontsize=8)
    axes[0, 0].set_ylabel("data")
    axes[1, 0].set_ylabel(r"$\hat D$")
    return _save(fig, out_path)


def plot_forward_grid(model, theta: torch.Tensor, out_path: Path | None = None):
    import matplotlib.pyplot as plt

    maps = model.maps(theta)
    y = model.y[0].cpu().numpy()
    mod = maps["model_epoch"].cpu().numpy()
    chi = maps["chi_epoch"].cpu().numpy()
    n = y.shape[0]
    fig, axes = plt.subplots(3, n, figsize=(1.8 * n, 5.4), squeeze=False)
    vmax = np.percentile(y, 99.5)
    for i in range(n):
        axes[0, i].imshow(y[i], origin="lower", vmin=0, vmax=vmax)
        axes[1, i].imshow(mod[i], origin="lower", vmin=0, vmax=vmax)
        axes[2, i].imshow(chi[i], origin="lower", cmap="RdBu_r", vmin=-4, vmax=4)
        for r in range(3):
            axes[r, i].set_xticks([]); axes[r, i].set_yticks([])
    for r, lab in enumerate(("data", "model", r"$\chi$")):
        axes[r, 0].set_ylabel(lab)
    return _save(fig, out_path)


def plot_offset_recovery(true_off: np.ndarray, fit_off: np.ndarray,
                         out_path: Path | None = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    err = fit_off - true_off
    ax.scatter(err[:, 0], err[:, 1], s=16)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$\Delta x$ [px]"); ax.set_ylabel(r"$\Delta y$ [px]")
    lim = max(0.05, float(np.abs(err).max()) * 1.3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    return _save(fig, out_path)


def plot_template_comparison(template: torch.Tensor, reference: torch.Tensor,
                             out_path: Path | None = None):
    import matplotlib.pyplot as plt

    t = template.detach().cpu().numpy()
    r = reference.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.6))
    vmax = np.percentile(r, 99.5)
    axes[0].imshow(t, origin="lower", vmin=0, vmax=vmax); axes[0].set_title("template")
    axes[1].imshow(r, origin="lower", vmin=0, vmax=vmax); axes[1].set_title("reference")
    im = axes[2].imshow(t - r, origin="lower", cmap="RdBu_r"); axes[2].set_title("difference")
    fig.colorbar(im, ax=axes[2], shrink=0.8)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    return _save(fig, out_path)


def plot_tau_profile_slice(model, theta: torch.Tensor, out_path: Path | None = None,
                           n: int = 25, lo: float = 1e-5, hi: float = 3e-2):
    import matplotlib.pyplot as plt

    grid = np.geomspace(lo, hi, n)
    vals = np.empty(n)
    with torch.no_grad():
        for j, tau in enumerate(grid):
            tp = theta.reshape(1, -1).clone()
            tp[0, -1] = tau
            vals[j] = float(model(tp, template=model.template_for(tp)))
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(grid, 2 * (vals.max() - vals))
    ax.set_xscale("log"); ax.set_xlabel(r"$\tau'$")
    ax.set_ylabel(r"$2[\ln L_p(\hat\tau) - \ln L_p(\tau)]$")
    return _save(fig, out_path)


def plot_shift_scan(model, theta: torch.Tensor, epoch: int = 1, axis: int = 0,
                    half: float = 0.75, n: int = 41, out_path: Path | None = None):
    """1-D slice of the profiled objective vs one offset coordinate."""
    import matplotlib.pyplot as plt

    j = 2 * (epoch - 1) + axis
    base = float(theta.reshape(-1)[j])
    grid = base + np.linspace(-half, half, n)
    vals = np.empty(n)
    with torch.no_grad():
        for i, v in enumerate(grid):
            tp = theta.reshape(1, -1).clone()
            tp[0, j] = v
            vals[i] = float(model(tp, template=model.template_for(tp)))
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(grid, -2 * vals)
    ax.axvline(base, color="C3", lw=1)
    ax.set_xlabel(f"offset[{'xy'[axis]}] epoch {epoch} [px]")
    ax.set_ylabel(r"$\chi^2_p$")
    return _save(fig, out_path)


def corner_map(samples: np.ndarray, names: list[str], out_path: Path | None = None):
    import corner  # optional
    import matplotlib.pyplot as plt  # noqa: F401

    fig = corner.corner(samples, labels=names, show_titles=True,
                        quantiles=[0.16, 0.5, 0.84])
    return _save(fig, out_path)
