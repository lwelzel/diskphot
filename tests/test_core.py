"""Core numerics: transforms, config, model bookkeeping, solver identity.

Slow end-to-end recovery tests are marked ``slow``; run them on a machine
with a GPU (or patience):  pytest -m slow
"""

import numpy as np
import pytest
import torch

from diskphot.config import FitConfig, ProfileConfig, SolveConfig, Variant
from diskphot.diagnostics import make_synthetic_epoch_data, theta_from_truth
from diskphot.model import JointLogLikelihood
from diskphot.synthetic import empirical_template, simulate_epochs
from diskphot.config import CalibConfig
from diskphot.transforms import shift, translate

DT = torch.float64


def test_shift_adjoint_matches_autograd_transpose():
    torch.manual_seed(0)
    x = torch.randn(3, 24, 24, dtype=DT, requires_grad=True)
    y = torch.randn(3, 24, 24, dtype=DT)
    s = torch.tensor([[0.37, -0.61], [0.5, 0.5], [-1.2, 0.05]], dtype=DT)
    (vjp,) = torch.autograd.grad((shift(x, s) * y).sum(), x)
    direct = shift(y, -s)
    assert torch.allclose(vjp, direct, atol=1e-12, rtol=1e-10)


def test_shift_integer_translation_is_exact():
    torch.manual_seed(1)
    x = torch.zeros(1, 16, 16, dtype=DT)
    x[0, 5:9, 4:8] = torch.rand(4, 4, dtype=DT)
    out = shift(x, torch.tensor([[2.0, -1.0]], dtype=DT))
    assert torch.allclose(out[0, 4:8, 6:10], x[0, 5:9, 4:8], atol=1e-12)


def test_translate_channels_match_plain_shift():
    torch.manual_seed(2)
    x = torch.randn(2, 3, 12, 12, dtype=DT)
    s = torch.tensor([[0.2, -0.4], [1.1, 0.6]], dtype=DT)
    out = translate(x, s)
    for b in range(2):
        for c in range(3):
            assert torch.allclose(out[b, c], shift(x[b, c][None], s[b : b + 1])[0])


def test_config_json_roundtrip(tmp_path):
    cfg = FitConfig(variant=Variant.PER_EPOCH)
    cfg.solve = SolveConfig(ridge=3e-12, chunk=4, resolve_every=5)
    cfg.data.mask_pixels = ((2, 11, 12),)
    path = tmp_path / "config.json"
    cfg.to_json(path)
    back = FitConfig.from_json(path)
    assert back.variant is Variant.PER_EPOCH
    assert back.solve.ridge == pytest.approx(3e-12)
    assert back.solve.resolve_every == 5
    assert back.data.mask_pixels == ((2, 11, 12),)


@pytest.mark.parametrize(
    "variant,extra",
    [(Variant.NULL, 0), (Variant.COMMON, 1), (Variant.PER_EPOCH, 4), (Variant.GREY, 3)],
)
def test_param_counts(variant, extra):
    data = make_synthetic_epoch_data(n_epochs=4, size=20)
    model = JointLogLikelihood(data, variant, SolveConfig())
    assert model.n_params == 2 * 3 + extra
    assert len(model.param_names) == model.n_params


def test_ridge_solve_matches_lstsq_reference():
    data = make_synthetic_epoch_data(n_epochs=4, size=20, seed=3)
    model = JointLogLikelihood(data, Variant.COMMON, SolveConfig())
    theta = theta_from_truth(model, np.zeros((4, 2)), tau=1e-3)
    delta, tpar = model._split(theta)
    t = model.transmission(tpar)
    shifts = model.nominal + delta
    ghat = model.solve_template(shifts, t)
    hh, ww = model.shape
    npix = hh * ww
    wt = model.w * t
    rhs = shift((wt * model.y).reshape(-1, hh, ww), -shifts.reshape(-1, 2))
    rhs = rhs.reshape(-1, hh, ww).sum(0).reshape(npix, 1).double()
    eye = torch.eye(npix, dtype=DT).reshape(npix, hh, ww)
    hmat = torch.zeros(npix, npix, dtype=torch.float64)
    for e in range(model.n_epochs):
        se = shifts[0, e].expand(npix, 2)
        cols = shift(eye, se) * (wt * t)[0, e]
        hmat += shift(cols, -se).reshape(npix, npix).double()
    hmat = 0.5 * (hmat + hmat.T)
    gref = torch.linalg.lstsq(hmat, rhs, driver="gelsd").solution.squeeze(-1)
    l_ridge = float(model(theta, template=ghat))
    l_ref = float(model(theta, template=gref.to(DT).reshape(1, hh, ww)))
    assert abs(l_ridge - l_ref) < 1e-3


def test_noiseless_chi2_at_truth_is_tiny():
    data = make_synthetic_epoch_data(n_epochs=4, size=20, seed=4)
    template, _ = empirical_template(data, ProfileConfig(steps=150), SolveConfig())
    obs, truth = simulate_epochs(
        data, 5e-2, template, 1, Variant.COMMON,
        CalibConfig(offset_sigma_px=0.2), add_noise=False, seed=4,
    )
    model = JointLogLikelihood(data, Variant.COMMON, SolveConfig(), observations=obs)
    th = theta_from_truth(model, truth.offsets_xy[0], tau=5e-2)
    assert -2.0 * float(model(th)) < 1e-2


@pytest.mark.slow
def test_recovery_of_injected_tau():
    from diskphot.inference import fit_mle

    data = make_synthetic_epoch_data(n_epochs=6, size=24, seed=5)
    prof = ProfileConfig(steps=250, n_starts=2)
    template, _ = empirical_template(data, prof, SolveConfig())
    obs, truth = simulate_epochs(
        data, 5e-2, template, 1, Variant.COMMON,
        CalibConfig(offset_sigma_px=0.25), add_noise=True, seed=5,
    )
    model = JointLogLikelihood(data, Variant.COMMON, SolveConfig(), observations=obs)
    res = fit_mle(model, prof, seed=5)
    tau_hat = float(res.theta[0, -1])
    assert abs(tau_hat - 5e-2) / 5e-2 < 0.15
