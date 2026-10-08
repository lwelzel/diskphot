"""Audit of the marginal likelihood on small synthetic epochs (CPU).

The checks are the ``verify()`` identities of ``diskphot.likelihood`` --
the same audit that ``run_inference`` performs on the real data -- plus the
common-grid identity with the pairwise statistic and the round trips of the
post-processing helpers (parameter layout, model card, column names).
"""

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from diskphot.likelihood import (  # noqa: E402
    NOISE_SCALE_RANGE,
    LogLikelihood,
    make_ellipse_mask,
    model_card_text,
    param_names,
    shift,
    single_run_sensitivity,
)
from diskphot.paper_values import (  # noqa: E402
    load_model_card,
    offsets_from_theta,
    param_layout,
    tau_statistics,
)
from diskphot.subsample_posterior import clean_names  # noqa: E402

N_EPOCHS, HW = 3, 12
TAU_TRUE = 0.05
SIGMA = 0.01
NOMINAL = [[0.0, 0.0], [0.3, -0.2], [-0.4, 0.25]]
DISK_AMPLITUDE = (0.05, 0.6, 1.0)      # epoch 0 off the ring, epoch 2 on its peak


@pytest.fixture(scope="module")
def data():
    """Three epochs of a Gaussian 'galaxy' behind a smooth 'disk', shifted
    by sub-pixel nominal offsets, attenuated by TAU_TRUE and noised."""
    gen = torch.Generator().manual_seed(0)
    ar = torch.arange(HW, dtype=torch.float32)
    yy, xx = torch.meshgrid(ar, ar, indexing="ij")
    c = (HW - 1) / 2.0
    galaxy = 0.5 * torch.exp(-((xx - c) ** 2 / (2 * 2.0 ** 2)
                               + (yy - c) ** 2 / (2 * 1.2 ** 2)))
    # The disk surface brightness differs between epochs (as the galaxy
    # moves across the ring): only the DIFFERENTIAL extinction between
    # epochs constrains tau'; a uniform attenuation is absorbed by the
    # free template.
    ramp = 0.2 + 0.1 * xx / HW                       # smooth, positive
    disk_models = torch.stack([a * ramp for a in DISK_AMPLITUDE])
    dhat = disk_models / disk_models.max()
    nominal = torch.tensor(NOMINAL, dtype=torch.float32)
    obs = []
    for i in range(N_EPOCHS):
        t = torch.exp(-TAU_TRUE * dhat[i])
        model = t * shift(galaxy[None], nominal[i:i + 1])[0] + disk_models[i]
        obs.append(model + SIGMA * torch.randn(HW, HW, generator=gen))
    observations = torch.stack(obs)
    masks = torch.stack([
        make_ellipse_mask(4.0, 2.5, 30.0, HW,
                          center_xy=(c + NOMINAL[i][0], c + NOMINAL[i][1]))
        for i in range(N_EPOCHS)])
    variance = torch.full_like(observations, SIGMA ** 2)
    return observations, disk_models, masks, variance, nominal


@pytest.fixture(scope="module")
def ll(data):
    return LogLikelihood(*data, noise_scale=True)


def _truth(ll, tau=TAU_TRUE):
    theta = torch.zeros(1, ll.ndim)
    theta[0, -1] = tau
    return theta


def test_layout_bookkeeping(ll):
    assert ll.n == N_EPOCHS and ll.size == HW
    assert ll.n_offsets == 2 * (N_EPOCHS - 1)
    assert ll.ndim == 2 * (N_EPOCHS - 1) + N_EPOCHS + 1
    names = param_names(N_EPOCHS, noise_scale=True)
    assert len(names) == ll.ndim and names[-1] == r"$\tau$"
    assert len(param_names(N_EPOCHS, noise_scale=False)) == 2 * (N_EPOCHS - 1) + 1
    assert ll.lam > 0 and math.isclose(ll.template_prior_std, ll.lam ** -0.5)


def test_verify_identities(ll):
    out = ll.verify(verbose=False, lsqr_budgets=(32,))
    assert out["adjoint_rel_f64"] < 1e-12
    assert out["stencil_vs_columns_rel"] < 1e-10
    assert out["logdet_chol_vs_eig"] < 1e-7
    assert out["dlnl_noise_scale_identity"] < 1e-6
    assert out["dlnl_legacy_layout"] < 1e-6
    assert out["grad_quad_fd_rel"] < 2e-2
    assert out["dlnl_marg_f32_vs_ref"] < 0.1
    assert out["gap_residual_f32"] < 1.0           # damped LSQR certificate
    assert np.isfinite(out["dlogdet_dtau_fd"]) and out["dlogdet_dtau_fd"] < 0
    assert np.isfinite(out["dlogdet_dlnf_fd"])


def test_forward_is_batched_and_tracks_reference(ll):
    th = _truth(ll).tile(3, 1)
    th[1, 0] += 0.2                       # perturbed offset
    th[2, ll.i_lnf] = math.log(1.5)       # noise scales
    batched = ll(th)
    rows = torch.cat([ll(th[i:i + 1]) for i in range(3)])
    assert torch.allclose(batched, rows, atol=1e-3, rtol=0)
    ref = ll.reference_logl(th)
    assert torch.allclose(batched, ref, atol=0.1, rtol=0)
    assert ref[0] > ref[1]                # truth beats the mis-registration


def test_tau_is_constrained_near_truth(ll):
    taus = torch.tensor([0.0, TAU_TRUE / 2, TAU_TRUE, 2 * TAU_TRUE, 0.5])
    th = _truth(ll).tile(len(taus), 1)
    th[:, -1] = taus
    lnl = ll.reference_logl(th)
    assert int(torch.argmax(lnl)) == 2
    assert lnl[2] - lnl[0] > 5.0 and lnl[2] - lnl[-1] > 5.0


def test_common_grid_identity(data):
    """Zero shifts: the correctly weighted pairwise statistic equals the
    lam -> 0 profiled chi-square (appendix of the paper)."""
    observations, disk_models, masks, variance, _ = data
    ll0 = LogLikelihood(observations, disk_models, masks, variance,
                        torch.zeros(N_EPOCHS, 2))
    pair = float(ll0.pairwise_chi2(TAU_TRUE))
    prof = float(ll0.profiled_chi2(_truth(ll0), lam=1e-12 * ll0._maxdiag_ref)[0])
    assert abs(pair - prof) / abs(pair) < 1e-6


def test_template_mean_and_error_map(ll):
    th = _truth(ll)
    g = ll.template_mean(th)
    assert g.shape == (1, HW, HW) and torch.isfinite(g).all()
    res, chi = ll.error_map(th)
    assert chi.shape == (1, N_EPOCHS, HW, HW)
    msr = float((chi[0] ** 2).sum() / ll.n_weighted.sum())
    assert 0.3 < msr < 3.0                # standardized residuals ~ unit
    th2 = th.clone()
    th2[0, ll.i_lnf] = math.log(2.0)
    _, chi_s = ll.error_map(th2, scaled=True)
    _, chi_n = ll.error_map(th2, scaled=False)
    assert torch.allclose(chi_s * 2.0, chi_n, atol=1e-5)


def test_reference_parts_consistency(ll):
    th = _truth(ll)
    th[0, ll.i_lnf] = math.log(1.7)
    parts = ll.reference_parts(th)
    assert torch.allclose(parts["chi2_epoch"].sum(-1), parts["chi2"])
    assert torch.allclose(parts["chi2_nominal"], parts["chi2"] * 1.7 ** 2,
                          rtol=1e-6)
    assert torch.allclose(parts["quad"], parts["chi2"] + parts["penalty"])


def test_model_card_round_trip(ll, tmp_path):
    (tmp_path / "MODEL_CARD.txt").write_text(
        model_card_text(ll, N_EPOCHS, 1.5, 1.0, 1e-2))
    card = load_model_card(tmp_path)
    assert math.isclose(card["lam"], ll.lam, rel_tol=1e-6)
    assert card["P"] == HW * HW and card["n_epochs"] == N_EPOCHS
    assert card["offset_halfwidth"] == 1.5 and card["tau_hi"] == 1.0
    assert card["noise_scale_range"] == list(NOISE_SCALE_RANGE)
    assert card["has_noise"] and len(card["n_weighted_per_epoch"]) == N_EPOCHS


def test_single_run_sensitivity(ll, tmp_path):
    rng = np.random.default_rng(1)
    n = 120
    pts = np.zeros((n, ll.ndim))
    pts[:, :ll.n_offsets] = rng.normal(0.0, 0.05, (n, ll.n_offsets))
    pts[:, ll.i_lnf] = rng.normal(0.0, 0.05, (n, ll.n))
    pts[:, -1] = np.abs(rng.normal(TAU_TRUE, 0.02, n))
    rows = single_run_sensitivity(ll, pts, np.full(n, 1.0 / n),
                                  max_points=80, batch=40, rb_bins=4,
                                  out_path=tmp_path / "sens.json",
                                  verbose=False)
    assert [r["variant"] for r in rows] == [
        "run prior", "lam x 0.1", "lam x 0.333", "lam x 3", "lam x 10",
        "soft-log tau prior"]
    assert all(0 < r["ess"] <= 80 for r in rows)
    assert all(np.isfinite(r["dlnZ_vs_run"]) for r in rows[1:])
    assert json.loads((tmp_path / "sens.json").read_text())[0]["variant"] == "run prior"


def test_param_layout_and_clean_names():
    names = param_names(8)
    lay = param_layout(names)
    assert lay["ndim"] == 23 and lay["n_epochs"] == 8 and lay["has_noise"]
    assert lay["i_tau"] == 22 and lay["i_lnf"] == list(range(14, 22))
    assert offsets_from_theta(np.arange(23.0), lay).shape == (8, 2)
    assert not param_layout(param_names(8, noise_scale=False))["has_noise"]
    assert clean_names(param_names(3)) == [
        "dx_1", "dy_1", "dx_2", "dy_2", "lnf_0", "lnf_1", "lnf_2", "tau"]


def test_tau_statistics_half_normal():
    tau = np.abs(np.random.default_rng(2).normal(0.0, 0.01, 20000))
    ts = tau_statistics(tau, tau_hi=1.0)
    assert ts["ll95"] < ts["median"] < ts["ul95"] < ts["ul997"]
    assert 0.9 < ts["dens0_rel"] <= 1.0          # mode at the boundary
    assert ts["zero_excl_cred"] < 0.5 and ts["soft"]["ul95"] < ts["ul95"]
