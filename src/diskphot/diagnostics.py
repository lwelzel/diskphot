"""Self-contained correctness checks (no real data required).

``run_verify`` cross-checks the library-backed operator and the solver:
adjointness of the negated shift against autograd's transpose, the ridge
solve against an independent least-squares solve, noiseless recovery,
envelope gradients against finite differences, and float32 consistency.
``run_roundtrip`` is the end-to-end injection/recovery smoke test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import CalibConfig, DataConfig, FitConfig, SolveConfig, Variant
from .data import EpochData, make_ellipse_mask, variance_map
from .galaxy import GalaxyParams, galaxy_image
from .model import JointLogLikelihood
from .transforms import shift
from . import constants


# ------------------------------------------------------------ synthetic data
def make_synthetic_epoch_data(
    n_epochs: int = 4,
    size: int = 26,
    sigma_bg: float = 3.0e-3,
    seed: int = 0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> EpochData:
    """Small fully synthetic :class:`EpochData` on native-style grids."""
    gen = torch.Generator().manual_seed(seed)
    cx = cy = (size - 1) / 2.0
    nominal = torch.zeros(n_epochs, 2, dtype=dtype)
    nominal[1:] = 0.8 * (torch.rand((n_epochs - 1, 2), generator=gen, dtype=dtype) - 0.5)

    galaxy = galaxy_image(size, GalaxyParams(), dtype=dtype)
    frames = []
    for i in range(n_epochs):
        frames.append(shift(galaxy.unsqueeze(0), nominal[i : i + 1])[0])
    observations = torch.stack(frames)

    # Disk foreground with a per-epoch amplitude sweep (contrast in D-hat).
    ys, xs = torch.meshgrid(
        torch.arange(size, dtype=dtype), torch.arange(size, dtype=dtype), indexing="ij"
    )
    ramp = 0.35 + 0.5 * (xs + ys) / (2 * (size - 1))
    amp = 0.45 + 1.1 * torch.arange(n_epochs, dtype=dtype) / max(n_epochs - 1, 1)
    disk = 0.02 * amp.view(-1, 1, 1) * ramp
    disk_peak_sb = float(disk.max())
    disk_hat = disk / disk_peak_sb
    observations = observations + disk

    exptimes = np.asarray(constants.EPOCH_EXPTIMES_S[:n_epochs], dtype=float)
    sig = np.full(n_epochs, sigma_bg)
    variance = variance_map(torch.clamp(observations, min=0.0), sig, exptimes, 1.0)

    mask = torch.stack([
        make_ellipse_mask(
            9.0 + 1.5, 5.0 + 1.5, 35.0, size,
            center_xy=(cx + float(nominal[i, 0]), cy + float(nominal[i, 1])),
        ).to(dtype)
        for i in range(n_epochs)
    ])
    return EpochData(
        observations=observations,
        disk=disk,
        disk_hat=disk_hat,
        variance=variance,
        valid=torch.ones_like(observations),
        mask=mask,
        nominal_shift_xy=nominal,
        galaxy_xy_cutout=torch.stack(
            [torch.tensor([cx, cy], dtype=dtype) + nominal[i] for i in range(n_epochs)]
        ),
        galaxy_xy_full=np.tile([cx, cy], (n_epochs, 1)) + nominal.numpy(),
        crop_origin_xy=np.zeros((n_epochs, 2), dtype=int),
        star_xy_full=(float(cx), float(cy)),
        disk_peak_sb=disk_peak_sb,
        disk_peak_radius_px=6.0,
        sigma_bg=sig,
        exptimes_s=exptimes,
        dates=constants.EPOCH_DATES[:n_epochs],
        config=DataConfig(cutout_size=size),
    ).to(device, dtype)


def theta_from_truth(model: JointLogLikelihood, offsets_xy: np.ndarray,
                     tau: float = 0.0) -> torch.Tensor:
    p = next(model.buffers())
    parts = [p.new_tensor(np.asarray(offsets_xy)[1:].reshape(-1))]
    if model.n_tau_params:
        parts.append(p.new_full((model.n_tau_params,), float(tau)))
    return torch.cat(parts).reshape(1, -1)


def theta_initial(model: JointLogLikelihood) -> torch.Tensor:
    p = next(model.buffers())
    theta = p.new_zeros(1, model.n_params)
    if model.variant in (Variant.COMMON, Variant.PER_EPOCH):
        theta[:, model.n_offset_params:] = 1.0e-3
    return theta


# ----------------------------------------------------------------- verify --
def run_verify(out_path: Path | None = None, seed: int = 0) -> dict:
    checks: dict[str, dict] = {}

    def add(name: str, value: float, tol: float) -> None:
        checks[name] = {"value": float(value), "tol": tol, "pass": bool(value <= tol)}

    torch.manual_seed(seed)

    # 1) adjoint: negated shift vs autograd transpose of the grid sampler.
    x = torch.randn(3, 26, 26, dtype=torch.float64, requires_grad=True)
    y = torch.randn(3, 26, 26, dtype=torch.float64)
    s = torch.tensor([[0.37, -0.61], [0.5, 0.5], [-1.2, 0.05]], dtype=torch.float64)
    (vjp,) = torch.autograd.grad((shift(x, s) * y).sum(), x)
    direct = shift(y, -s)
    add("adjoint_vjp_vs_negated_shift",
        (vjp - direct).abs().max().item() / direct.abs().max().item(), 1.0e-10)

    # 2) ridge-Cholesky vs independent lstsq on the same normal equations.
    data = make_synthetic_epoch_data(seed=seed)
    model = JointLogLikelihood(data, Variant.COMMON, SolveConfig())
    theta = theta_from_truth(model, np.zeros((data.n_epochs, 2)), tau=1.0e-3)
    delta, tpar = model._split(theta)
    t = model.transmission(tpar)
    shifts = model.nominal + delta
    ghat = model.solve_template(shifts, t)
    # rebuild H, b once more through public pieces for the reference solve
    hh, ww = model.shape
    npix = hh * ww
    wt = model.w * t
    rhs = shift((wt * model.y).reshape(-1, hh, ww), -shifts.reshape(-1, 2))
    rhs = rhs.reshape(-1, hh, ww).sum(0).reshape(npix, 1).double()
    eye = torch.eye(npix, dtype=model.w.dtype).reshape(npix, hh, ww)
    hmat = torch.zeros(npix, npix, dtype=torch.float64)
    for e in range(model.n_epochs):
        se = shifts[0, e].expand(npix, 2)
        cols = shift(eye, se) * (wt * t)[0, e]
        hmat += shift(cols, -se).reshape(npix, npix).double()
    hmat = 0.5 * (hmat + hmat.T)
    gref = torch.linalg.lstsq(hmat, rhs, driver="gelsd").solution.squeeze(-1)
    logl_ridge = model(theta, template=ghat)
    logl_ref = model(theta, template=gref.to(model.w.dtype).reshape(1, hh, ww))
    add("ridge_solve_vs_lstsq_logl", abs(float(logl_ridge - logl_ref)), 1.0e-3)

    # 3) noiseless roundtrip: chi2 ~ 0 at the truth.
    from .synthetic import simulate_epochs

    template, _ = _noise_free_template(data)
    obs, truth = simulate_epochs(
        data, 5.0e-2, template, 1, Variant.COMMON,
        CalibConfig(offset_sigma_px=0.2), add_noise=False, seed=seed,
    )
    m2 = JointLogLikelihood(data, Variant.COMMON, SolveConfig(), observations=obs)
    th = theta_from_truth(m2, truth.offsets_xy[0], tau=5.0e-2)
    add("noiseless_chi2_at_truth", -2.0 * float(m2(th)), 1.0e-2)

    # 4) envelope gradient vs central finite differences (off the optimum,
    #    where the gradient is nonzero).
    th_eval = th.clone(); th_eval[0, 0] += 0.15; th_eval[0, -1] *= 1.3
    g_auto = _envelope_grad(m2, th_eval)
    g_fd = _fd_grad(m2, th_eval)
    denom = max(np.abs(g_fd).max(), 1e-12)
    add("envelope_grad_vs_fd", float(np.abs(g_auto - g_fd).max() / denom), 5.0e-3)

    # 5) float32 vs float64 consistency of the profiled objective.
    data32 = data.to("cpu", torch.float32)
    m32 = JointLogLikelihood(data32, Variant.COMMON, SolveConfig(), observations=obs.float())
    rel = abs(float(m32(th.float())) - float(m2(th))) / max(abs(float(m2(th))), 1.0)
    add("float32_vs_float64_logl_rel", rel, 5.0e-2)

    result = {"all_pass": all(c["pass"] for c in checks.values()), "checks": checks}
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def _noise_free_template(data: EpochData) -> tuple[torch.Tensor, np.ndarray]:
    from .synthetic import empirical_template
    from .config import ProfileConfig

    return empirical_template(data, ProfileConfig(steps=150), SolveConfig())


def _envelope_grad(model: JointLogLikelihood, theta: torch.Tensor) -> np.ndarray:
    th = theta.clone().requires_grad_(True)
    tmpl = model.template_for(th.detach())
    (g,) = torch.autograd.grad(model(th, template=tmpl).sum(), th)
    return g[0].cpu().numpy()


def _fd_grad(model: JointLogLikelihood, theta: torch.Tensor, h: float = 1.0e-4) -> np.ndarray:
    g = np.empty(model.n_params)
    for j in range(model.n_params):
        tp1 = theta.clone(); tp1[0, j] += h
        tp2 = theta.clone(); tp2[0, j] -= h
        with torch.no_grad():
            v1 = float(model(tp1, template=model.template_for(tp1)))
            v2 = float(model(tp2, template=model.template_for(tp2)))
        g[j] = (v1 - v2) / (2 * h)
    return g


# --------------------------------------------------------------- roundtrip --
def run_roundtrip(
    cfg: FitConfig | None = None,
    tau_true: float = 5.0e-2,
    add_noise: bool = True,
    out_dir: Path | None = None,
    seed: int = 0,
) -> dict:
    """Inject a known signal into synthetic epochs and recover it end to end."""
    from .inference import fit_mle, fit_offsets
    from .synthetic import empirical_template, simulate_epochs

    cfg = cfg or FitConfig()
    torch.manual_seed(seed)
    data = make_synthetic_epoch_data(n_epochs=6, seed=seed)
    template, _ = empirical_template(data, cfg.profile, cfg.solve)
    calib = CalibConfig(offset_sigma_px=0.25)
    obs, truth = simulate_epochs(
        data, tau_true, template, 1, Variant.COMMON, calib,
        add_noise=add_noise, seed=seed,
    )
    null = JointLogLikelihood(data, Variant.NULL, cfg.solve, observations=obs)
    res_null = fit_offsets(null, cfg.profile)
    model = JointLogLikelihood(data, Variant.COMMON, cfg.solve, observations=obs)
    res = fit_mle(model, cfg.profile, seed=seed)
    theta_hat = res.theta[0]
    tau_hat = float(theta_hat[-1])
    th_truth = theta_from_truth(model, truth.offsets_xy[0], tau=tau_true)
    logl_truth = float(model(th_truth, template=model.template_for(th_truth)))

    off_hat = theta_hat[: model.n_offset_params].reshape(-1, 2)
    off_err = off_hat - truth.offsets_xy[0, 1:]
    maps = model.maps(torch.as_tensor(theta_hat, dtype=model.w.dtype, device=model.w.device))
    w = maps["weight"]
    msr = float((maps["chi_epoch"][w > 0] ** 2).mean())
    summary = {
        "tau_true": tau_true,
        "tau_hat": tau_hat,
        "tau_rel_err": abs(tau_hat - tau_true) / tau_true,
        "offset_rms_err_px": float(np.sqrt((off_err ** 2).mean())),
        "logl_null": float(res_null.logl[0]),
        "logl_mle": float(res.logl[0]),
        "logl_truth": logl_truth,
        "q0": max(2.0 * (float(res.logl[0]) - float(res_null.logl[0])), 0.0),
        "delta_chi2_recovered_vs_truth": 2.0 * (float(res.logl[0]) - logl_truth),
        "mean_squared_std_residual": msr,
        "noiseless_exact": not add_noise,
    }
    if not add_noise:
        summary["tau_likelihood_equivalent"] = abs(summary["delta_chi2_recovered_vs_truth"])
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "roundtrip.json").write_text(json.dumps(summary, indent=2))
        from .plotting import residual_panels

        residual_panels(maps, out_dir / "roundtrip_residuals.png")
    return summary


# ---------------------------------------------------------------- spectrum --
def nuisance_spectrum(
    model: JointLogLikelihood, theta: torch.Tensor, out_path: Path | None = None
) -> dict:
    """Eigen-spectrum of the template normal matrix and a small FD Hessian."""
    theta = theta.reshape(1, -1)
    delta, tpar = model._split(theta)
    t = model.transmission(tpar)
    shifts = model.nominal + delta
    hh, ww = model.shape
    npix = hh * ww
    wt2 = model.w * t * t
    eye = torch.eye(npix, dtype=model.w.dtype, device=model.w.device).reshape(npix, hh, ww)
    hmat = torch.zeros(npix, npix, dtype=torch.float64, device=model.w.device)
    for e in range(model.n_epochs):
        se = shifts[0, e].expand(npix, 2)
        cols = shift(eye, se) * wt2[0, e]
        hmat += shift(cols, -se).reshape(npix, npix).double()
    evals = torch.linalg.eigvalsh(0.5 * (hmat + hmat.T)).cpu().numpy()

    h = 1.0e-3
    p = model.n_params
    hess = np.empty((p, p))
    with torch.no_grad():
        f0 = float(model(theta, template=model.template_for(theta)))
        for a in range(p):
            for b in range(a, p):
                vals = []
                for sa, sb in ((+1, +1), (+1, -1), (-1, +1), (-1, -1)):
                    tp = theta.clone()
                    tp[0, a] += sa * h
                    tp[0, b] += sb * h
                    vals.append(float(model(tp, template=model.template_for(tp))))
                hess[a, b] = hess[b, a] = (vals[0] - vals[1] - vals[2] + vals[3]) / (4 * h * h)
    fisher = -hess
    d = np.sqrt(np.clip(np.diag(fisher), 1e-300, None))
    corr = fisher / np.outer(d, d)
    out = {
        "normal_matrix_eigenvalues": evals.tolist(),
        "n_nonpositive_modes": int((evals <= 0).sum()),
        "param_names": model.param_names,
        "fisher_diag": np.diag(fisher).tolist(),
        "param_correlations": corr.tolist(),
        "logl_at_theta": f0,
    }
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(out, indent=2))
    return out
