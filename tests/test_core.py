"""Core numerical tests: shift operator, template solves, config."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from diskphot.config import CGConfig, FitConfig, Variant
from diskphot.diagnostics import make_synthetic_epoch_data, theta_from_truth
from diskphot.model import JointLogLikelihood
from diskphot.synthetic import empirical_template, simulate_epochs
from diskphot.transforms import shift2d

DTYPE = torch.float64


# --------------------------------------------------------------------------- #
# Shift operator
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kernel", ["catmull-rom", "keys", "bilinear"])
def test_adjoint_exact(kernel):
    gen = torch.Generator().manual_seed(1)
    x = torch.randn((3, 17, 17), generator=gen, dtype=DTYPE)
    y = torch.randn((3, 17, 17), generator=gen, dtype=DTYPE)
    s = 3.0 * (torch.rand((3, 2), generator=gen, dtype=DTYPE) - 0.5)
    lhs = (shift2d(x, s, kernel=kernel) * y).sum()
    rhs = (x * shift2d(y, -s, kernel=kernel)).sum()
    assert abs(float(lhs - rhs)) < 1.0e-12 * float(lhs.abs() + 1.0)


@pytest.mark.parametrize("kernel", ["catmull-rom", "keys", "bilinear"])
def test_integer_shift_exact(kernel):
    gen = torch.Generator().manual_seed(2)
    x = torch.randn((1, 12, 12), generator=gen, dtype=DTYPE)
    s = torch.tensor([[2.0, -3.0]], dtype=DTYPE)
    out = shift2d(x, s, kernel=kernel)
    # content moves by (+x right, +y down): out[r, c] = in[r - y, c - x]
    assert torch.allclose(out[0, : 12 - 3, 2:], x[0, 3:, : 12 - 2], atol=1e-14)


def test_partition_of_unity():
    ones = torch.ones((1, 15, 15), dtype=DTYPE)
    s = torch.tensor([[0.37, -0.61]], dtype=DTYPE)
    out = shift2d(ones, s, kernel="catmull-rom")
    # interior pixels (away from the zero-padded border) stay exactly one
    assert torch.allclose(out[0, 3:-3, 3:-3], ones[0, 3:-3, 3:-3], atol=1e-13)


def test_squared_taps_give_diagonal():
    gen = torch.Generator().manual_seed(3)
    size = 10
    w = torch.rand((1, size, size), generator=gen, dtype=DTYPE)
    s = 2.0 * (torch.rand((1, 2), generator=gen, dtype=DTYPE) - 0.5)
    fast = shift2d(w, -s, squared=True)[0]
    npix = size * size
    eye = torch.eye(npix, dtype=DTYPE).reshape(npix, size, size)
    cols = shift2d(eye, s.expand(npix, 2))
    exact = (w[0].reshape(-1)[None, :] * cols.reshape(npix, npix) ** 2).sum(dim=1)
    assert float((fast.reshape(-1) - exact).abs().max()) < 1.0e-12


# --------------------------------------------------------------------------- #
# Template solves
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def small_problem():
    cfg = FitConfig()
    data = make_synthetic_epoch_data(n_epochs=4, size=20, seed=5).to("cpu", DTYPE)
    template, _ = empirical_template(data, cfg.profile, cfg.cg)
    obs, truth = simulate_epochs(
        data, 5.0e-3, template, n_realizations=1, variant=Variant.COMMON,
        calib=cfg.calib, add_noise=False, seed=5,
    )
    model = JointLogLikelihood(data, Variant.COMMON, cfg.cg, observations=obs)
    theta = theta_from_truth(model, truth).unsqueeze(0)
    return cfg, model, theta


def test_direct_solve_reproduces_noiseless_truth(small_problem):
    _, model, theta = small_problem
    with torch.inference_mode():
        chi2 = float(-2.0 * model(theta, tight=True)[0])
    assert chi2 < 1.0e-6


def test_direct_matches_columnwise_lstsq(small_problem):
    _, model, theta = small_problem
    gen = torch.Generator().manual_seed(6)
    th = theta.clone()
    th[0, : model.n_offset_params] += 0.3 * (
        torch.rand(model.n_offset_params, generator=gen, dtype=DTYPE) - 0.5
    )
    with torch.inference_mode():
        logl_direct = float(model(th, tight=True)[0])
    _, logl_dense = model.profile_dense(th)
    assert abs(logl_direct - float(logl_dense[0])) < 1.0e-3


def test_loose_cg_is_lower_bound_and_close(small_problem):
    _, model, theta = small_problem
    with torch.inference_mode():
        loose = float(model(theta, tight=False)[0])
        tight = float(model(theta, tight=True)[0])
    # both are explicit-residual values, so both lower-bound the exact
    # profile; the direct solve attains it
    assert loose <= tight + 1.0e-9
    assert tight - loose < 2.0


def test_envelope_gradient_matches_fd(small_problem):
    _, model, theta = small_problem
    gen = torch.Generator().manual_seed(7)
    th = theta[0].clone()
    th[: model.n_offset_params] += 0.2 * (
        torch.rand(model.n_offset_params, generator=gen, dtype=DTYPE) - 0.5
    )
    th_g = th.clone().requires_grad_(True)
    logl = model(th_g.unsqueeze(0), tight=True)[0]
    (grad,) = torch.autograd.grad(logl, th_g)
    eps = 1.0e-4
    j = 0  # test one offset coordinate and the tau coordinate
    for j in (0, model.n_params - 1):
        tp = th.clone(); tp[j] += eps
        tm = th.clone(); tm[j] -= eps
        with torch.inference_mode():
            fd = (
                float(model(tp.unsqueeze(0), tight=True)[0])
                - float(model(tm.unsqueeze(0), tight=True)[0])
            ) / (2 * eps)
        scale = max(abs(float(grad[j])), 1.0)
        assert abs(float(grad[j]) - fd) / scale < 5.0e-3


def test_param_counts_by_variant():
    data = make_synthetic_epoch_data(n_epochs=4, size=16, seed=8).to("cpu", DTYPE)
    cg = CGConfig()
    n = data.observations.shape[0]
    n_off = 2 * (n - 1)  # epoch-0 offsets gauge-fixed to zero
    assert JointLogLikelihood(data, Variant.NULL, cg).n_params == n_off
    assert JointLogLikelihood(data, Variant.COMMON, cg).n_params == n_off + 1
    assert JointLogLikelihood(data, Variant.PER_EPOCH, cg).n_params == n_off + n
    assert JointLogLikelihood(data, Variant.GREY, cg).n_params == n_off + (n - 1)


# --------------------------------------------------------------------------- #
# Config round-trip
# --------------------------------------------------------------------------- #
def test_config_json_roundtrip(tmp_path):
    cfg = FitConfig(variant=Variant.PER_EPOCH)
    cfg.data.mask_pixels = ((2, 12, 12),)
    cfg.calib.taus = (1.0e-4, 3.0e-4)
    cfg.calib.noise_model = "correlated"
    path = tmp_path / "config.json"
    cfg.to_json(path)
    back = FitConfig.from_json(path)
    assert back.variant is Variant.PER_EPOCH
    assert back.data.mask_pixels == ((2, 12, 12),)
    assert back.calib.taus == (1.0e-4, 3.0e-4)
    assert back.calib.noise_model == "correlated"
    assert back.cg.rcond == cfg.cg.rcond
