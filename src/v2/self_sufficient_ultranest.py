"""Current (v0.7) MARGINAL log-likelihood + data loading, one self-sufficient file.

v0.7 = v0.6 + PER-EPOCH NOISE-SCALE NUISANCE PARAMETERS (the ARKS f_sigma
idea).  The v06 production run measured a mean squared standardized
residual of 3.8 over the weighted pixels: the pixel scatter is ~2x the
nominal sigma of the noise model, and with FIXED weights every credible
interval inherits that underestimate.  The minimal repair is to let the
data set the noise level, epoch by epoch:

    sigma_i^2 -> f_i^2 sigma_i^2,   i.e.   w_i -> w_i / f_i^2 ,

with one free scale f_i per epoch, sampled as ln f_i (log-uniform prior,
``NOISE_SCALE_RANGE`` below).  The parameter vector becomes

    theta = [dx_1, dy_1, ..., dx_{N-1}, dy_{N-1},  ln f_0, ..., ln f_{N-1},  tau]

(offsets, then the N noise scales, then tau' LAST -- so every consumer that
takes tau' from theta[-1] keeps working; ``param_names()`` spells the
layout out).  Nothing changes in the forward model or in the algebra: H, b
and chi2 are all linear in the weights, so a per-member, per-epoch weight
scale f^-2 is threaded through the stencil assembly, the right-hand side
and the explicit residual (``wscale`` arguments below), and the Gaussian
data normalization acquires its f-dependence, -sum_i n_i ln f_i (n_i =
weighted pixels of epoch i).  That last term is NOT optional: chi2/f^2
alone is monotone in f and the sampler would run to the upper prior bound.
The template prior lam is a prior on the galaxy image (counts/s/px) and is
independent of the noise model: it stays the empirical-Bayes scale
computed from the NOMINAL weights (f = 1), frozen at construction.
A legacy theta without the ln f block (2(N-1)+1 wide) is still accepted
everywhere and evaluated at f = 1, so v06 run directories can be
re-evaluated with this file.  ``verify()`` gained the machine check that
scaling the weights of a second instance by f^-2 reproduces the marginal
at ln f exactly, plus the FD gradient / volume slope along ln f.

Same shape as v05 (flat module-level loading, one torch Module, batched
``forward(theta) -> log_likelihood`` with
``theta = [dx_1, dy_1, ..., dx_{N-1}, dy_{N-1}, tau]``); same forward model,
same weights, same stencil-assembled dense solve.  What changed is WHAT the
returned number is:

  v05                                   v06
  ----------------------------------    ----------------------------------
  PROFILED (ridged) likelihood:         MARGINAL likelihood: the template
  lnL_p = -chi2/2 at the template       is integrated out under a PROPER
  that minimizes the ridged             Gaussian prior g ~ N(0, lam^-1 I),
  objective; the template is a          lam a FIXED scalar.  Closed form
  plug-in, its uncertainty ignored      (model linear in g):

  ridge = ESTIMATOR parameter,            lnL(theta) = -1/2 [ chi2(ghat)
  lambda = 1e-5 * max diag H(theta),                  + lam ||ghat||^2 ]
  i.e. theta-DEPENDENT                                - 1/2 ln det(H + lam I)
                                                      + P/2 ln lam  + c_data
  nested-sampling map = geometry        with ghat = (H + lam I)^{-1} b.
  diagnostic only; "logz" a             The run IS Bayesian: samples are a
  pseudo-evidence (nuisance volume      posterior, logz IS an evidence
  omitted); science numbers from        under the priors stated below (and
  the external calibration              in MODEL_CARD.txt next to the run).

Why this costs (almost) nothing per evaluation: the hot path already
assembles H exactly from its 7x7 translation stencil and Cholesky-factors
(H + lam I) once per batch member.  ln det(H + lam I) = 2 sum ln diag(L) is
read off that SAME factor -- O(P) extra on top of the O(P^3/3)
factorization, i.e. well under 0.1%.  The penalty lam ||ghat||^2 is another
O(P).  Nothing else changes on the hot path.  (A matrix-free marginal would
need stochastic Lanczos quadrature for the log-determinant; NOT implemented
-- the marginal requires the dense solver.  LSQR is retained only as a
template-solve diagnostic.)

The three semantic changes, spelled out:

  (1) lam is now part of the MODEL (a prior precision), so it must be one
      fixed number, not 1e-5 * max diag H(theta).  Default: an
      empirical-Bayes scale, lam = lam_rel_init * max diag H at the
      reference point (nominal registration, tau' = 0), computed ONCE at
      construction and frozen.  This keeps the well-tested f32 conditioning
      regime (kappa(H + lam I) ~ 1e5 near tau' = 0, smaller at larger tau'
      since extinction shrinks H) while making the prior coherent.  It is a
      data-informed prior choice -- say so in the paper.  The lam ladder
      (x 1/10, 1/3, 3, 10) that stands in for v05's ridge-robustness scan
      is computed from the SAME run by importance reweighting at the
      stored posterior points (``single_run_sensitivity``): no sampling.
      The offsets are likelihood-dominated and only tau' responds to lam,
      so the alternative posteriors overlap strongly; the Kish ESS printed
      per variant certifies each reweighting, and the ctor arg ``lam=``
      remains for an explicit rerun should an ESS ever collapse.
      Equivalently the prior is a per-pixel template std
      sigma_g = lam^{-1/2} (printed at startup); fixing sigma_g from a
      stated physical scale is the purist alternative.
  (2) The nuisance-volume term -1/2 ln det(H + lam I) + P/2 ln lam
      = -1/2 sum_j ln(1 + sigma_j/lam)  (sigma_j the eigenvalues of H)
      is <= 0 and INCREASES with tau' (extinction shrinks H), i.e. the
      marginal shifts posterior weight toward larger tau' relative to the
      v05 profile map.  Expect the tau' posterior/upper limit to move UP
      somewhat versus the v05 pseudo-posterior; ``verify()`` prints the
      finite-difference slope d(ln det)/d(tau') so the size of the effect
      on the real data is on record.  As tau' -> inf the term -> 0 and
      chi2 -> C: the marginal is PROPER (the flat-prior divergence that
      forbade marginalization before is specific to the flat prior).
  (3) Because the reported value now includes the penalty lam ||ghat||^2,
      the envelope gradient became EXACT: autograd through ``forward`` (with
      ghat detached) returns the exact gradient of the quadratic part
      -1/2 (chi2 + lam ||ghat||^2) -- the v05 ridge-coupling bias
      (-2 lam ghat^T d ghat/d theta, 0.1..1.4%) is gone, because that term
      IS the envelope term of the full ridged objective.  The
      log-determinant's theta-gradient is NOT propagated (solve runs under
      no_grad); the nested sampler is gradient-free so nothing needs it,
      but do not use autograd of ``forward`` to optimize the full marginal.

Numerical policy (unchanged in spirit): hot path float32 end-to-end -- the
stencil makes the f32 H the correctly ROUNDED exact matrix, and at the
default lam the factorization is comfortably PD.  If f32 Cholesky ever
reports non-PD (only roundoff can cause it; H + lam I is PD exactly), the
affected chunk is redone in float64 at the SAME lam -- the prior is never
inflated to rescue a factorization.  NEW f32 question introduced by the
marginal: the accuracy of ln det from the f32 factor.  ``verify()`` now
measures it on the data (``dlogdet_f32_vs_ref``, ``dlnl_marg_f32_vs_ref``)
next to the old chi2 check; expected ~<0.1 from the floored tail modes
(~200 modes x ~kappa*eps_f32 with random signs).  If it comes out larger
than your dlogz budget, evaluate the hot path in f64 (``forward`` via
``reference_logl``); the stencil assembly still avoids v04's launch storm,
so that costs the fp64 Cholesky throughput only, not a return to v04.
Reported values (best points, quoted lnL, evidence cross-checks) come from
``reference_logl``: the float64 evaluation of the SAME marginal.

Priors of the sampled parameters (part of the model; also written to
MODEL_CARD.txt next to each run):
  * offsets delta_i: independent uniform on [-OFFSET_HALFWIDTH,
    +OFFSET_HALFWIDTH] px about the nominal registration;
  * tau': FLAT on [0, TAU_HI] -- the simplest proper choice and the
    conservative one for an upper limit (it does not upweight the
    tau' = 0 neighbourhood).  The evidence then scales with 1/TAU_HI:
    state the ceiling wherever lnZ is quoted.  The v05 soft-log prior,
    density p(tau') = 1 / [ (tau' + TAU_SOFT) * ln(1 + TAU_HI/TAU_SOFT) ],
    is kept as a REWEIGHTED variant of the same run (pure prior-ratio
    weights; no likelihood re-evaluation), not as a separate run.
  * noise scales f_i (one per epoch, v07): ln f_i independent and
    uniform on [ln NOISE_SCALE_RANGE[0], ln NOISE_SCALE_RANGE[1]] (the
    Jeffreys choice for a scale).  With ~200 weighted pixels per epoch
    the posterior width is ~1/sqrt(2 n_i) ~ 0.05 in ln f, so the bounds
    only need to contain the answer; the evidence scales with the
    inverse of the prior width per epoch.
  * template g: N(0, lam^{-1} I_P) as above.
``logz`` includes P/2 ln lam and (by default) the Gaussian data
normalization c_data = 1/2 sum_{w>0} ln(w / 2 pi); both are
theta-independent, but the first matters when comparing runs with
different lam and the second when comparing different weight maps.
Evidence differences between model variants fitted to the SAME weights are
unaffected by c_data.

Do NOT ``resume`` a v05 (profiled) run directory: the likelihood changed.

NOTE: numerically exercised on synthetic epochs (CPU) by the accompanying
harness; run ``verify()`` on the real data (wired into __main__) before
trusting numbers.  Science numbers now come from THIS posterior/evidence
(plus the f64 ``reference_logl`` re-evaluations); the external calibration
machinery remains available as a frequentist cross-check.
"""

import json
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import tensor
from torch.nn import Module

from pathlib import Path
from astropy.io import fits

torch.set_default_device("cuda:0" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# constants (old style: module level)
# --------------------------------------------------------------------------- #
DATA_ROOT = Path("/home/lwelzel/git/diskphot/data")
OBS_DIR = DATA_ROOT / "hd107146_reduced"
DISK_MODEL_PATH = DATA_ROOT / "hd107146_disk_model/MODEL20_HD107146_16.5_47_h=0.3_NUP.fits"
CSV_PATH = Path("/home/lwelzel/git/diskphot/data/galaxy_positions.csv")

# UPDATED VALUES FROM THE FITS FILE HEADERS
DISK_MODEL_SHIFT_XY = (-540.45, -541.08)    # known disk-model -> frame alignment
CUTOUT = 24                                 # px; pipeline default is 30
SIGMA_BG = 3.0e-3                           # counts/s/px background RMS
GAIN = 4.0                                  # CCDGAIN =                    4 / commanded gain of CCD    
EXPTIMES_S = (1588.0, 1662.8) + (8314.0,) * 6

KEYS_A = -0.75              # grid_sample "bicubic" = Keys cubic convolution

# v07: per-epoch noise-scale prior, log-uniform f_i in this range.  The
# nominal noise model (f = 1) and the empirical scatter of the v06 run
# (sqrt(MSR) ~ 2) both sit well inside; see the module docstring.
NOISE_SCALE_RANGE = (0.25, 10.0)


def param_names(n_epochs, noise_scale=True):
    """Sampled-parameter names in theta order (the ONE definition of the
    layout, shared with the run's MODEL_CARD and the paper scripts):
    offsets of epochs 1..N-1, then (optionally) ln f_0..ln f_{N-1}, then
    tau' last."""
    names = [rf"${p}_{i}$" for i in range(1, n_epochs)
             for p in (r"\delta x", r"\delta y")]
    if noise_scale:
        names += [rf"$\ln f_{i}$" for i in range(n_epochs)]
    return names + [r"$\tau$"]


# --------------------------------------------------------------------------- #
# sub-pixel shift: torch grid sampler (unchanged from v04)
# --------------------------------------------------------------------------- #
def shift(images: torch.Tensor, shifts_xy: torch.Tensor) -> torch.Tensor:
    """``images (B, H, W)``, ``shifts_xy (B, 2)``; content moves by ``+s``.

    Bicubic (Keys a = -3/4), zero padding, ``align_corners=True``.  For this
    symmetric kernel the exact matrix transpose is a negated shift:
    ``S_s^T x == shift(x, -s)`` -- the only adjoint the LSQR solver needs.
    Applied ONLY to models (disk model, galaxy template), never to data.
    """
    b, h, w = images.shape
    dt, dev = images.dtype, images.device
    shifts_xy = shifts_xy.to(device=dev, dtype=dt)
    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=dt, device=dev),
        torch.arange(w, dtype=dt, device=dev),
        indexing="ij",
    )
    gx = xs - shifts_xy[:, 0].view(-1, 1, 1)
    gy = ys - shifts_xy[:, 1].view(-1, 1, 1)
    grid = torch.stack(
        (2.0 * gx / max(w - 1, 1) - 1.0, 2.0 * gy / max(h - 1, 1) - 1.0), dim=-1
    )
    return F.grid_sample(
        images.unsqueeze(1), grid, mode="bicubic",
        padding_mode="zeros", align_corners=True,
    ).squeeze(1)


def _keys_1d(x: torch.Tensor) -> torch.Tensor:
    """Keys (1981) cubic-convolution kernel, a = -3/4 (grid_sample's)."""
    a = KEYS_A
    x = x.abs()
    near = ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    far = ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
    return torch.where(x <= 1.0, near, torch.where(x < 2.0, far, torch.zeros_like(x)))


def _axis_taps(s: torch.Tensor):
    """Integer base ``m = floor(s)`` and the 4 kernel taps ``c_a = k(a - f)``,
    ``a in {-1, 0, 1, 2}``, ``f = s - m`` -- exactly grid_sample's weights."""
    m = torch.floor(s)
    f = s - m
    args = torch.stack([-1.0 - f, -f, 1.0 - f, 2.0 - f], dim=-1)
    return m.long(), _keys_1d(args)


def _axis_gamma(c: torch.Tensor) -> torch.Tensor:
    """Per-axis stencil coefficients ``Gamma[o+3, a] = c_a * c_{a-o}``,
    ``o in {-3..3}`` (``(..., 4) -> (..., 7, 4)``); zero where ``a - o``
    falls outside the 4-tap window."""
    pad = torch.zeros(c.shape[:-1] + (3,), dtype=c.dtype, device=c.device)
    cpad = torch.cat([pad, c, pad], dim=-1)                       # (..., 10)
    o = torch.arange(-3, 4, device=c.device)
    a = torch.arange(4, device=c.device)
    idx = (a.view(1, 4) - o.view(7, 1) + 3).expand(c.shape[:-1] + (7, 4))
    gathered = torch.gather(
        cpad.unsqueeze(-2).expand(c.shape[:-1] + (7, 10)), -1, idx
    )
    return c.unsqueeze(-2) * gathered                             # (..., 7, 4)


def _int_shift(img: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """``out(p) = img(p + m)`` with zero fill; ``img (Q, H, W)``, ``m (Q, 2)``
    long ``(mx, my)``.  Exact integer translation (no interpolation)."""
    q, h, w = img.shape
    dev = img.device
    ys = torch.arange(h, device=dev).view(1, h, 1) + m[:, 1].view(q, 1, 1)
    xs = torch.arange(w, device=dev).view(1, 1, w) + m[:, 0].view(q, 1, 1)
    valid = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    out = img[torch.arange(q, device=dev).view(q, 1, 1),
              ys.clamp(0, h - 1), xs.clamp(0, w - 1)]
    return out * valid.to(img.dtype)


# --------------------------------------------------------------------------- #
# data loading (unchanged from v04)
# --------------------------------------------------------------------------- #
def load_observations():
    paths = sorted(OBS_DIR.glob("*.fits"))
    return [tensor(np.squeeze(fits.open(p)[0].data).astype(np.float32)) for p in paths]


def load_disk_model():
    disk = tensor(np.squeeze(fits.open(DISK_MODEL_PATH)[0].data).astype(np.float32))
    # sub-pixel shifting a NOISELESS MODEL is fine (and needed here);
    # it is the observations that must never be resampled.
    return shift(disk[None], tensor([DISK_MODEL_SHIFT_XY]))[0]


def load_position_guess():
    df = pd.read_csv(CSV_PATH, comment='#')
    df["x_est"] = df["x"]
    df["y_est"] = df["y"]
    return df


def make_ellipse_mask(a, b, theta_deg, size=CUTOUT, center_xy=None, margin=1.5):
    """Binary elliptical aperture; per-epoch center, margin-enlarged so it
    contains the galaxy for every offset the fit is allowed to try."""
    th = np.deg2rad(theta_deg)
    cx, cy = center_xy if center_xy is not None else ((size - 1) / 2.0,) * 2
    ys, xs = torch.meshgrid(
        torch.arange(size, dtype=torch.float32) - cy,
        torch.arange(size, dtype=torch.float32) - cx,
        indexing="ij",
    )
    u = xs * np.cos(th) + ys * np.sin(th)
    v = -xs * np.sin(th) + ys * np.cos(th)
    r2 = (u / (a + margin)) ** 2 + (v / (b + margin)) ** 2
    return (r2 <= 1.0).to(torch.float32)


def load_native_data(cutout=CUTOUT):
    """NO recentering of the data (see v04 docstring): integer crops only,
    residual sub-pixel geometry carried by the model as nominal shifts."""
    frames = load_observations()
    disk = load_disk_model()
    pos = load_position_guess()
    q_full = np.stack([pos["x_est"].to_numpy(), pos["y_est"].to_numpy()], axis=1)

    obs_list, disk_list, q_cut = [], [], []
    for i, frame in enumerate(frames):
        origin = np.round(q_full[i] - (cutout - 1) / 2.0).astype(int)  # (x0, y0)
        x0, y0 = int(origin[0]), int(origin[1])
        obs_list.append(frame[y0:y0 + cutout, x0:x0 + cutout])          # integer crop
        disk_list.append(disk[y0:y0 + cutout, x0:x0 + cutout])
        q_cut.append(q_full[i] - origin)                                # float, cutout frame

    observations = torch.stack(obs_list)                                # (N, H, W)
    disk_models = torch.stack(disk_list)
    q_cut = np.asarray(q_cut)
    nominal_shift_xy = tensor((q_cut - q_cut[0]).astype(np.float32))    # (N, 2)

    masks = torch.stack([
        make_ellipse_mask(9, 5, 35, cutout, center_xy=(q_cut[i, 0], q_cut[i, 1]))
        for i in range(len(frames))
    ])
    texp = tensor(np.asarray(EXPTIMES_S[: len(frames)], dtype=np.float32)).view(-1, 1, 1)
    variance = SIGMA_BG ** 2 + torch.clamp(observations, min=0.0) / (GAIN * texp)

    return observations, disk_models, masks, variance, nominal_shift_xy, q_cut


# --------------------------------------------------------------------------- #
# likelihood
# --------------------------------------------------------------------------- #
class LogLikelihood(Module):
    """MARGINAL joint log-likelihood (template integrated out), batched.

    theta (B, 2*(N-1) + N + 1): offsets of epochs 1..N-1 w.r.t. epoch 0
    (epoch-0 gauge), then the per-epoch noise scales ln f_0..ln f_{N-1}
    (v07), then the common tau' (always last).  A legacy (B, 2*(N-1) + 1)
    theta without the ln f block is accepted and evaluated at f = 1.  The
    model is linear in the free template g, so under the proper Gaussian
    prior g ~ N(0, lam^{-1} I_P) the marginalization is closed-form:

        lnL(theta) = -1/2 [ chi2(ghat) + lam ||ghat||^2 ]
                     - 1/2 ln det(H + lam I) + P/2 ln lam
                     + c_data - sum_i n_i ln f_i ,

    with H, b the (exact, stencil-assembled) normal-equation pieces built
    from the SCALED weights w_i / f_i^2, ghat = (H + lam I)^{-1} b, chi2
    the explicit weighted (scaled) residual at ghat, and n_i the number
    of weighted pixels of epoch i (the f-dependence of the Gaussian
    normalization).  The first bracket equals C - b^T ghat exactly, but
    the explicit form is the cancellation-safe one in every precision
    (see ``certificate``).  ``lam`` is FIXED at construction (see module
    docstring; computed from the NOMINAL weights, independent of f); pass
    ``lam=...`` to ``reference_parts`` only for prior-sensitivity scans.

    ``forward`` evaluates the marginal with the float32 hot path (stencil
    assembly + one batched f32 Cholesky, whose factor also yields the
    log-determinant); ``reference_logl`` is the float64 evaluation of the
    SAME marginal and produces every reported value.

    ``solver``: "dense" (required for the marginal -- the log-determinant
    comes from the Cholesky factor); "lsqr" is retained ONLY as a
    memory-light template-solve diagnostic (``solve_lsqr``) and raises in
    ``forward``.
    """

    def __init__(self, observations, disk_models, masks, variance,
                 nominal_shift_xy, lam=None, lam_rel_init=1.0e-5,
                 include_data_norm=True, chunk=32,
                 solver="dense", lsqr_iters=256, lsqr_atol=0.0,
                 precondition=True, warm_start=False, accumulate_f64=True,
                 noise_scale=True):
        super().__init__()
        self.n = observations.shape[0]
        self.size = observations.shape[-1]
        # theta layout (see module docstring / ``param_names``): offsets,
        # then the N noise scales when ``noise_scale``, then tau' last.
        self.noise_scale = bool(noise_scale)
        self.n_offsets = 2 * (self.n - 1)
        self.i_lnf = slice(self.n_offsets, self.n_offsets + self.n) \
            if self.noise_scale else slice(self.n_offsets, self.n_offsets)
        self.ndim = self.n_offsets + (self.n if self.noise_scale else 0) + 1
        self.chunk = int(chunk)
        self.solver = str(solver)
        self.lsqr_iters = int(lsqr_iters)
        self.lsqr_atol = float(lsqr_atol)
        self.precondition = bool(precondition)
        self.warm_start = bool(warm_start)
        self.accumulate_f64 = bool(accumulate_f64)
        self._warm_x = None
        self._scatter_cache = {}
        # disk-subtracted data (the galaxy signal); never de-attenuated,
        # never resampled anywhere below.
        y = observations - disk_models
        w = masks / variance
        finite = torch.isfinite(y) & torch.isfinite(w)
        w = torch.where(finite, w, torch.zeros_like(w))
        self.register_buffer("y", torch.nan_to_num(y).contiguous())
        self.register_buffer("dhat", (disk_models / disk_models.max()).contiguous())
        self.register_buffer("w", w.contiguous())
        self.register_buffer("sqw", w.sqrt().contiguous())
        self.register_buffer("nominal", nominal_shift_xy.contiguous())
        # C = sum_i y_i^T W y_i at the NOMINAL weights (f = 1); f64 once
        # (used only by the certificate, which rescales it per member).
        self.register_buffer(
            "c_const", (self.w.double() * self.y.double().square()).sum()
        )
        # n_i = weighted pixels of epoch i: the coefficient of -ln f_i in
        # the Gaussian normalization (v07); f64, (N,).
        self.register_buffer("n_weighted", (self.w > 0).sum(dim=(1, 2)).double())

        # ---- FIXED template-prior precision lam (part of the model) -----
        # Empirical-Bayes default: lam_rel_init * max diag H at the
        # reference point (nominal registration, tau' = 0), computed once
        # from the exact o = 0 stencil plane and then FROZEN.  It is a
        # deterministic function of the data (one scalar); document it as
        # such, and rerun with explicit ``lam`` for prior sensitivity.
        with torch.no_grad():
            t_ref = torch.ones(1, self.n, self.size, self.size,
                               dtype=torch.float64, device=self.w.device)
            s_ref = self.nominal.double().unsqueeze(0)
            diag_ref = self._stencil_planes(s_ref, t_ref, torch.float64,
                                            diag_only=True)
        self._maxdiag_ref = float(diag_ref.max())
        if lam is not None:
            self.lam = float(lam)
        else:
            self.lam = float(lam_rel_init) * self._maxdiag_ref
        if not (self.lam > 0.0):
            raise ValueError("template prior precision lam must be > 0")
        self.template_prior_std = 1.0 / math.sqrt(self.lam)

        # theta-independent Gaussian data normalization over weighted
        # pixels: c_data = 1/2 sum_{w>0} [ln w - ln 2 pi].  Kept so that
        # logz is a bona-fide evidence; irrelevant for posterior shape and
        # for evidence DIFFERENCES between variants sharing these weights.
        self.include_data_norm = bool(include_data_norm)
        self._c_data = self._data_norm(self.w) if include_data_norm else 0.0

    @staticmethod
    def _data_norm(w):
        """1/2 sum_{w>0} [ln w - ln 2 pi] for a weight map (f = 1 part)."""
        wpos = w[w > 0].double()
        return float(0.5 * (wpos.log() - math.log(2.0 * math.pi)).sum())

    def lnl_const(self, lam=None, lnf=None):
        """Additive part of lnL outside the quadratic and log-det terms:
        c_data + P/2 ln lam (theta-independent) minus, when ``lnf`` (B, N)
        is given, the f-dependent normalization sum_i n_i ln f_i -> (B,)
        float64.  The latter is part of the model whenever f != 1,
        regardless of ``include_data_norm`` (which only governs the
        constant): without it the likelihood is monotone in f."""
        lam = self.lam if lam is None else float(lam)
        c = self._c_data + 0.5 * self.size * self.size * math.log(lam)
        if lnf is None:
            return c
        return c - (lnf.double() * self.n_weighted).sum(-1)

    # ------------------------------------------------------------------ theta
    def _split(self, theta):
        """theta -> (total shifts (B, N, 2), tau' (B,), ln f (B, N)).

        Accepts the v07 layout (offsets, ln f block, tau') and the legacy
        one without the ln f block (then ln f = 0, i.e. nominal weights).
        """
        if theta.dim() == 1:
            theta = theta.view(1, -1)
        theta = theta.reshape(-1, theta.shape[-1])
        b, width = theta.shape
        n_off = self.n_offsets
        if width == n_off + self.n + 1:
            lnf = theta[:, n_off:n_off + self.n]
        elif width == n_off + 1:
            lnf = theta.new_zeros(b, self.n)
        else:
            raise ValueError(
                f"theta has {width} parameters; expected {n_off + self.n + 1} "
                f"(offsets, ln f, tau) or {n_off + 1} (legacy: offsets, tau)")
        delta = torch.cat(
            [theta.new_zeros(b, 1, 2), theta[:, :n_off].view(b, self.n - 1, 2)],
            dim=1)
        tau = theta[:, -1].clamp_min(0.0)
        return self.nominal.to(theta.dtype) + delta, tau, lnf

    @staticmethod
    def _wscale(lnf, dtype):
        """Per-member, per-epoch weight scale f_i^-2 as (B, N, 1, 1)."""
        return torch.exp(-2.0 * lnf.to(dtype)).unsqueeze(-1).unsqueeze(-1)

    # -------------------------------------------------------- stencil algebra
    def _stencil_planes(self, shifts, t, dtype, diag_only=False, wscale=None):
        """The 49 diagonals of ``H = sum_i S_i^T diag(w t^2 / f^2) S_i`` --
        exact (``wscale`` = f^-2, (B, N, 1, 1); None = nominal weights).

        A translation by ``s = m + f`` factorizes as ``S_s = T_m S_f``
        (integer then fractional), so ``S_s^T D S_s = S_f^T D' S_f`` with the
        integer-shifted weight image ``d'(p) = d(p + m)``.  Because the
        bicubic kernel has 4 taps per axis, ``S_f^T D' S_f`` couples template
        pixels only within ``|o| <= 3`` per axis, and each of the 49
        diagonals ``plane_o(q) = H[q, q + o]`` is the fixed 16-term
        combination

            plane_o(q) = sum_{a,b in {-1..2}} c^x_a c^x_{a-ox}
                                              c^y_b c^y_{b-oy} d'(q + (a, b)),

        with ``c_a = k(a - f)`` the interpolation taps.  Zero padding of
        ``d`` and of the ``(a, b)`` windows reproduces grid_sample's
        zero-padded borders exactly, so this equals the column-assembled H
        to roundoff (``verify()``: ~5e-16).

        Returns (B, 7, 7, H, W) planes indexed ``[oy + 3, ox + 3]`` -- or the
        (B, H, W) diagonal ``o = 0`` alone (``diag_only``, used by the LSQR
        Jacobi preconditioner).
        """
        b, n = shifts.shape[:2]
        hw = self.size
        d = (self.w.to(dtype) * t.to(dtype).square())              # (B, N, H, W)
        if wscale is not None:
            d = d * wscale.to(dtype)
        s = shifts.to(dtype)
        mx, cx = _axis_taps(s[..., 0])                             # (B,N), (B,N,4)
        my, cy = _axis_taps(s[..., 1])
        m = torch.stack([mx, my], dim=-1).reshape(-1, 2)
        dp = _int_shift(d.reshape(-1, hw, hw), m)                  # (B*N, H, W)
        dpad = F.pad(dp, (1, 2, 1, 2))
        win = dpad.unfold(1, hw, 1).unfold(2, hw, 1)               # (B*N,4b,4a,H,W)
        win = win.reshape(b, n, 4, 4, hw, hw)
        if diag_only:
            gx2 = cx.square()                                      # (B, N, 4)
            gy2 = cy.square()
            return torch.einsum("qeb,qea,qebahw->qhw", gy2, gx2, win)
        gx = _axis_gamma(cx)                                       # (B, N, 7, 4)
        gy = _axis_gamma(cy)
        planes = d.new_zeros(b, 7, 7, hw, hw)
        for e in range(n):                                         # N small
            planes += torch.einsum(
                "qib,qja,qbahw->qijhw", gy[:, e], gx[:, e], win[:, e]
            )
        return planes

    def _scatter_indices(self, device):
        """Static (rows, cols, window) index sets mapping the 49 planes into
        the dense (P, P) matrix; cached per device."""
        key = str(device)
        if key in self._scatter_cache:
            return self._scatter_cache[key]
        hw = self.size
        out = []
        for oy in range(-3, 4):
            for ox in range(-3, 4):
                y0, y1 = max(0, -oy), hw - max(0, oy)
                x0, x1 = max(0, -ox), hw - max(0, ox)
                qy = torch.arange(y0, y1, device=device)
                qx = torch.arange(x0, x1, device=device)
                gy, gx = torch.meshgrid(qy, qx, indexing="ij")
                rows = (gy * hw + gx).reshape(-1)
                cols = ((gy + oy) * hw + (gx + ox)).reshape(-1)
                out.append((oy + 3, ox + 3, y0, y1, x0, x1, rows, cols))
        self._scatter_cache[key] = out
        return out

    def _dense_from_planes(self, planes):
        """(B, 7, 7, H, W) stencil planes -> dense (B, P, P) H (exact)."""
        b = planes.shape[0]
        hw = self.size
        npix = hw * hw
        h = planes.new_zeros(b, npix, npix)
        for iy, ix, y0, y1, x0, x1, rows, cols in self._scatter_indices(planes.device):
            h[:, rows, cols] = planes[:, iy, ix, y0:y1, x0:x1].reshape(b, -1)
        return h

    def _columns_H(self, shifts, t, wscale=None):
        """v04's column-by-column H assembly (float64), kept ONLY as the
        independent cross-check for ``verify()`` -- npix grid_sample calls
        per epoch, per member."""
        b, n = shifts.shape[:2]
        hw = self.size
        npix = hw * hw
        w64 = self.w.double()
        t64, s64 = t.double(), shifts.double()
        wt2 = w64 * t64 * t64
        if wscale is not None:
            wt2 = wt2 * wscale.double()
        eye = torch.eye(npix, dtype=torch.float64, device=w64.device
                        ).reshape(npix, hw, hw)
        acc = torch.zeros(b, npix, npix, dtype=torch.float64, device=w64.device)
        for e in range(n):
            se = s64[:, e].unsqueeze(1).expand(b, npix, 2).reshape(-1, 2)
            cols = shift(eye.unsqueeze(0).expand(b, npix, hw, hw).reshape(-1, hw, hw), se)
            cols = cols.reshape(b, npix, hw, hw) * wt2[:, e].reshape(b, 1, hw, hw)
            acc += shift(cols.reshape(-1, hw, hw), -se).reshape(b, npix, npix)
        return 0.5 * (acc + acc.mT)

    # ----------------------------------------------------------- dense solver
    def _rhs(self, shifts, t, dtype, wscale=None):
        """b = sum_i S_i^T (w t y / f^2), in ``dtype``."""
        b, n = shifts.shape[:2]
        hw = self.size
        wty = (self.w.to(dtype) * t.to(dtype) * self.y.to(dtype))
        if wscale is not None:
            wty = wty * wscale.to(dtype)
        r = shift(wty.reshape(-1, hw, hw), -shifts.to(dtype).reshape(-1, 2))
        return r.reshape(b, n, hw, hw).sum(1).reshape(b, hw * hw)

    def solve_dense(self, shifts, t, dtype=None, lam=None, return_logdet=False,
                    wscale=None):
        """Template posterior mean at the FIXED prior precision ``lam``:
        stencil-assembled H + batched Cholesky of (H + lam I); ``wscale``
        (B, N, 1, 1) = f^-2 scales the weights per member and epoch.

        ``dtype`` float32 is the hot path: at the default lam the
        regularized system has kappa ~ 1e5, comfortably inside f32 Cholesky
        territory, and the stencil assembly makes the f32 H the correctly
        ROUNDED exact matrix rather than an accumulation of interpolation
        roundoff (v04's f32 failures came from column-assembly noise
        towering over a 1e-12 ridge).  ``verify()`` measures the f32-vs-f64
        agreement of BOTH the quadratic part and the log-determinant on the
        real data.

        If the f32 factorization reports non-PD (H + lam I is PD in exact
        arithmetic; only roundoff can break it), the affected CHUNK is
        redone in float64 at the SAME lam -- the prior is never changed to
        rescue a factorization.  ``lam`` may be overridden for
        prior-sensitivity scans only.

        ``return_logdet``: also return ln det(H + lam I) per member
        (float64), read off the same Cholesky factor -- the marginal
        likelihood costs O(P) on top of the O(P^3/3) already spent.
        """
        dtype = self.w.dtype if dtype is None else dtype
        lam = self.lam if lam is None else float(lam)
        b = shifts.shape[0]
        hw = self.size
        npix = hw * hw
        dev = self.w.device
        rhs = self._rhs(shifts, t, dtype, wscale=wscale)
        idn = torch.eye(npix, dtype=dtype, device=dev)
        ghat = torch.empty(b, npix, dtype=dtype, device=dev)
        logdet = torch.empty(b, dtype=torch.float64, device=dev)
        for i0 in range(0, b, self.chunk):
            sl = slice(i0, min(i0 + self.chunk, b))
            ws = None if wscale is None else wscale[sl]
            planes = self._stencil_planes(shifts[sl], t[sl], dtype, wscale=ws)
            hmat = self._dense_from_planes(planes)
            chol, info = torch.linalg.cholesky_ex(hmat + lam * idn)
            if int(info.max()) == 0:
                ghat[sl] = torch.cholesky_solve(rhs[sl].unsqueeze(-1), chol
                                                ).squeeze(-1)
                logdet[sl] = 2.0 * chol.diagonal(dim1=-2, dim2=-1
                                                 ).double().log().sum(-1)
                continue
            if dtype == torch.float64:
                raise RuntimeError(
                    "float64 Cholesky of H + lam I reported non-PD; "
                    "H is PSD by construction and lam > 0 -- investigate."
                )
            # f32 roundoff broke PD: redo this chunk in f64 at the SAME lam
            s64 = shifts[sl].double()
            t64 = t[sl].double()
            ws64 = None if ws is None else ws.double()
            h64 = self._dense_from_planes(
                self._stencil_planes(s64, t64, torch.float64, wscale=ws64))
            idn64 = torch.eye(npix, dtype=torch.float64, device=dev)
            chol64, info64 = torch.linalg.cholesky_ex(h64 + lam * idn64)
            if int(info64.max()) > 0:
                raise RuntimeError("f64 retry of H + lam I still non-PD")
            rhs64 = self._rhs(s64, t64, torch.float64, wscale=ws64)
            ghat[sl] = torch.cholesky_solve(rhs64.unsqueeze(-1), chol64
                                            ).squeeze(-1).to(dtype)
            logdet[sl] = 2.0 * chol64.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        ghat = ghat.reshape(b, hw, hw)
        return (ghat, logdet) if return_logdet else ghat

    # ------------------------------------------------------------ LSQR solver
    def _weighted_data(self, wscale, dtype, b):
        """z = sqrt(w) y / f, (B, N, H, W) (expanded view when f = 1)."""
        z = (self.sqw.to(dtype) * self.y.to(dtype)).unsqueeze(0)
        if wscale is None:
            return z.expand(b, self.n, self.size, self.size)
        return z * wscale.to(dtype).sqrt()

    def _ops(self, shifts, t, wscale=None):
        """Matrix-free ``A`` and ``A^T`` of the stacked weighted design,
        z = sqrt(w) y / f:  A x = (sqrt(w)/f) t S_s x  (one batched
        grid_sample), A^T r = sum_i S_{-s}((sqrt(w)/f) t r) (one batched
        negated shift).  The adjoint is exact for this symmetric kernel
        (S_s^T = S_{-s})."""
        b, n = shifts.shape[:2]
        hw = self.size
        at = self.sqw.to(t.dtype) * t                     # (B, N, H, W)
        if wscale is not None:
            at = at * wscale.to(t.dtype).sqrt()
        flat = shifts.reshape(-1, 2)

        def a_op(x):                                      # (B, P) -> (B, N, H, W)
            g = x.reshape(b, 1, hw, hw).expand(b, n, hw, hw)
            return at * shift(g.reshape(-1, hw, hw), flat).reshape(b, n, hw, hw)

        def at_op(r):                                     # (B, N, H, W) -> (B, P)
            v = shift((at * r).reshape(-1, hw, hw), -flat)
            return v.reshape(b, n, hw, hw).sum(1).reshape(b, hw * hw)

        return a_op, at_op

    def solve_lsqr(self, shifts, t, x0=None, iters=None, atol=None,
                   precondition=None, damp=None, wscale=None):
        """Matrix-free LSQR (Paige & Saunders 1982) for the template solve;
        never forms H, one batched grid_sample pair per iteration.  The
        memory-light DIAGNOSTIC for cutouts where the (B, P, P) dense matrix
        does not fit -- it cannot provide the log-determinant, so it does
        not serve the marginal likelihood (that would need stochastic
        Lanczos quadrature; not implemented).

        ``damp`` selects the target system:
          * None (default): the MODEL system -- delta = sqrt(lam) with the
            fixed prior precision, i.e. the same (H + lam I) g = b the
            dense path solves.  Implemented by running the (validated)
            undamped recurrences on the AUGMENTED operator [A; delta I]
            with rhs [z; 0].  The lam floor removes the stall of the pure
            least-squares solve, but the misfit still converges through the
            spectral continuum (diagonal scaling cannot uniformize the
            floor), so budget a few hundred iterations; check ``verify()``
            on the real data.
          * 0.0: the lam -> 0 minimum-energy problem.  Measured
            APPROXIMATE-to-DIVERGENT: on small well-conditioned cases the
            iterates stall ~0.5..2 chi2 above the tiny-ridge answer; on
            the real data (kappa(H) ~ 1e13) finite-precision Lanczos
            diverges outright -- chi2 error ~1e7 and GROWING with k --
            which is precisely why the lam floor exists.  Diagnostics
            only; nothing downstream consumes this path.
          * float or (B,) tensor: explicit delta, penalty delta^2 ||g||^2.

        Jacobi column scaling (``precondition``, sqrt(diag H) from the
        o = 0 stencil plane) composes with the damping: the penalty block
        is scaled consistently ([A M; delta M]), so the metric of the
        penalty stays ||g||^2 in ORIGINAL pixel variables and the target
        is exactly the dense system.  Warm starts solve the correction
        system with rhs [z - A x0; -delta x0].  Fixed ``iters`` with
        ``atol = 0`` keeps the solve a deterministic function of the data.
        Any iterate evaluated through the explicit residual remains a
        valid chi2 upper bound; ``certificate`` returns the exact
        optimality gap.  NOT used for reported values.
        """
        iters = self.lsqr_iters if iters is None else int(iters)
        atol = self.lsqr_atol if atol is None else float(atol)
        precondition = self.precondition if precondition is None else bool(precondition)
        b, n = shifts.shape[:2]
        hw = self.size
        npix = hw * hw
        dt = t.dtype
        tiny = torch.finfo(dt).tiny
        a_op, at_op = self._ops(shifts, t, wscale)
        z = self._weighted_data(wscale, dt, b)

        diag = None
        if precondition:
            diag = self._stencil_planes(shifts, t, dt, diag_only=True,
                                        wscale=wscale).reshape(b, npix)
        if damp is None:
            delta = torch.full((b,), math.sqrt(self.lam), dtype=dt,
                               device=z.device)
        else:
            delta = torch.as_tensor(damp, dtype=dt, device=z.device
                                    ).expand(b).clone()
        damped = bool((delta > 0).any())
        dv = delta.view(b, 1)
        if precondition:
            scale = torch.where(diag > 0, diag.clamp_min(tiny).rsqrt(),
                                torch.ones_like(diag))
        else:
            scale = None

        def am(xv):                # (B, P) -> data block, penalty block
            xs = xv * scale if scale is not None else xv
            return a_op(xs), (dv * xs if damped else None)

        def amt(u1, u2):           # blocks -> (B, P)
            v = at_op(u1)
            if damped:
                v = v + dv * u2
            return v * scale if scale is not None else v

        def _norm(u1, u2):
            s2 = u1.reshape(b, -1).square().sum(1)
            if damped:
                s2 = s2 + u2.square().sum(1)
            return s2.sqrt()

        if x0 is not None:
            x0f = x0.reshape(b, npix)
            u1 = z - a_op(x0f)
            u2 = (-dv * x0f) if damped else None
        else:
            u1 = z.clone()
            u2 = torch.zeros(b, npix, dtype=dt, device=z.device) if damped else None
        beta = _norm(u1, u2)
        bs = beta.clamp_min(tiny)
        u1 = u1 / bs.view(b, 1, 1, 1)
        u2 = u2 / bs.view(b, 1) if damped else None
        v = amt(u1, u2)
        alpha = v.norm(dim=1)
        v = v / alpha.clamp_min(tiny).view(b, 1)
        wv = v.clone()
        x = torch.zeros(b, npix, dtype=dt, device=z.device)
        phibar, rhobar = beta.clone(), alpha.clone()
        arnorm0 = (alpha * beta).clamp_min(tiny)
        for _ in range(iters):
            n1, n2 = am(v)
            u1 = n1 - alpha.view(b, 1, 1, 1) * u1
            u2 = (n2 - alpha.view(b, 1) * u2) if damped else None
            beta = _norm(u1, u2)
            bs = beta.clamp_min(tiny)
            u1 = u1 / bs.view(b, 1, 1, 1)
            u2 = u2 / bs.view(b, 1) if damped else None
            v = amt(u1, u2) - beta.view(b, 1) * v
            alpha = v.norm(dim=1)
            v = v / alpha.clamp_min(tiny).view(b, 1)
            rho = torch.sqrt(rhobar.square() + beta.square()).clamp_min(tiny)
            c, s = rhobar / rho, beta / rho
            theta = s * alpha
            rhobar = -c * alpha
            phi = c * phibar
            phibar = s * phibar
            x = x + (phi / rho).view(b, 1) * wv
            wv = v - (theta / rho).view(b, 1) * wv
            if atol > 0.0:
                arnorm = phibar * alpha * c.abs()
                if bool((arnorm <= atol * arnorm0).all()):
                    break
        if scale is not None:
            x = x * scale
        if x0 is not None:
            x = x0.reshape(b, npix) + x
        return x.reshape(b, hw, hw)

    # ------------------------------------------------------ template dispatch
    def solve_template(self, shifts, t, wscale=None):
        if self.solver == "lsqr":
            x0 = None
            if self.warm_start and self._warm_x is not None \
                    and self._warm_x.shape[0] == shifts.shape[0]:
                x0 = self._warm_x
            ghat = self.solve_lsqr(shifts, t, x0=x0, wscale=wscale)
            if self.warm_start:
                self._warm_x = ghat.detach()
            return ghat
        return self.solve_dense(shifts, t, wscale=wscale)

    def template_mean(self, theta, dtype=torch.float64):
        """Template posterior mean ghat (B, H, W) at ``theta`` (f64 reference
        solve by default) -- the one entry point the paper scripts use, so
        they need not know the theta layout or the weight scaling."""
        with torch.no_grad():
            shifts, tau, lnf = self._split(theta.to(dtype))
            t = torch.exp(-tau.view(-1, 1, 1, 1) * self.dhat.to(dtype))
            return self.solve_dense(shifts, t, dtype=dtype,
                                    wscale=self._wscale(lnf, dtype))

    def reset_warm_start(self):
        self._warm_x = None

    # --------------------------------------------------------------- forward
    def forward(self, theta):
        """Batched MARGINAL log-likelihood (float64 values returned).

        The template posterior mean and ln det(H + lam I) come from ONE
        f32 stencil-Cholesky per member; chi2 is the explicit residual
        (cancellation-safe in any precision, and any approximate template
        yields a valid bound of the quadratic part).

        Autograd through this evaluation is the ENVELOPE gradient of the
        quadratic part -1/2 (chi2 + lam ||ghat||^2), and -- unlike v05 --
        it is now EXACT for that part: the penalty completes the ridged
        objective, whose envelope derivative at the optimum has no
        ghat-coupling term (machine-checked in ``verify()``).  The
        log-determinant's theta-gradient is NOT propagated (the solve runs
        under no_grad); the nested sampler is gradient-free, but do not
        use this autograd to optimize the full marginal.
        """
        if self.solver != "dense":
            raise RuntimeError(
                "the marginal likelihood needs the dense solver's "
                "log-determinant; 'lsqr' is a template-solve diagnostic only"
            )
        shifts, tau, lnf = self._split(theta)
        t = torch.exp(-tau.view(-1, 1, 1, 1) * self.dhat)        # (B, N, H, W)
        wscale = self._wscale(lnf, self.w.dtype)                 # (B, N, 1, 1)
        with torch.no_grad():                                    # envelope
            ghat, logdet = self.solve_dense(shifts.detach(), t.detach(),
                                            return_logdet=True,
                                            wscale=wscale.detach())
        b = shifts.shape[0]
        hw = self.size
        g = ghat.unsqueeze(1).expand(b, self.n, hw, hw)
        model = t * shift(g.reshape(-1, hw, hw), shifts.reshape(-1, 2)).reshape(
            b, self.n, hw, hw
        )
        r2 = (self.w * wscale * (self.y - model).square())
        chi2 = r2.double().sum(dim=(1, 2, 3)) if self.accumulate_f64 \
            else r2.sum(dim=(1, 2, 3)).double()
        pen = self.lam * ghat.double().square().sum(dim=(1, 2))
        lnl = -0.5 * (chi2 + pen) - 0.5 * logdet + self.lnl_const(lnf=lnf)
        return lnl.contiguous()

    def reference_parts(self, theta, lam=None):
        """Float64 evaluation of the SAME marginal, in labelled pieces.

        Returns dict with ``chi2`` (explicit residual at ghat, SCALED
        weights w/f^2 -- what enters the likelihood), ``chi2_epoch`` (B, N)
        its per-epoch split, ``chi2_nominal`` / ``chi2_nominal_epoch`` (the
        same residual under the NOMINAL weights, = chi2_epoch * f^2: the
        goodness-of-fit numbers of the fixed-noise model, at the template
        fitted WITH the scales), ``penalty`` (lam ||ghat||^2), ``quad``
        (chi2 + penalty, == C - b^T ghat exactly), ``logdet``
        (ln det(H + lam I)), ``lnl`` (the marginal incl. constants and the
        -sum n_i ln f_i normalization), ``lnf`` / ``f`` (B, N) and ``lam``.
        Passing ``lam`` evaluates an alternative PRIOR (sensitivity scans);
        constants change consistently so lnl values remain comparable as
        evidences.
        """
        lam = self.lam if lam is None else float(lam)
        with torch.no_grad():
            shifts, tau, lnf = self._split(theta.double())
            t = torch.exp(-tau.view(-1, 1, 1, 1) * self.dhat.double())
            wscale = self._wscale(lnf, torch.float64)
            ghat, logdet = self.solve_dense(shifts, t, dtype=torch.float64,
                                            lam=lam, return_logdet=True,
                                            wscale=wscale)
            chi2_ep = self._chi2_at(shifts, t, ghat, wscale=wscale,
                                    per_epoch=True)
            chi2 = chi2_ep.sum(-1)
            f2 = torch.exp(2.0 * lnf)
            pen = lam * ghat.square().sum(dim=(1, 2))
            lnl = (-0.5 * (chi2 + pen) - 0.5 * logdet
                   + self.lnl_const(lam, lnf=lnf))
        return {"chi2": chi2, "chi2_epoch": chi2_ep,
                "chi2_nominal": (chi2_ep * f2).sum(-1),
                "chi2_nominal_epoch": chi2_ep * f2,
                "penalty": pen, "quad": chi2 + pen,
                "logdet": logdet, "lnl": lnl, "lam": lam,
                "lnf": lnf, "f": torch.exp(lnf)}

    def reference_logl(self, theta, lam=None):
        """The reference evaluation: float64 marginal at the model lam
        (every REPORTED value comes from here; the f32 hot path tracks it
        -- ``verify()`` measures how closely on the real data)."""
        return self.reference_parts(theta, lam=lam)["lnl"]

    def profiled_chi2(self, theta, lam=None):
        """Float64 explicit chi2 at the ridged template (NO penalty, NO
        log-determinant, NO constants; SCALED weights) -- the v05 profile
        objective, kept for the pairwise-identity demo and for
        profile-vs-marginal comparisons.  ``lam`` defaults to the model
        lam; pass a tiny value (e.g. ``1e-12 * self._maxdiag_ref``) for
        the lam -> 0 profile."""
        lam = self.lam if lam is None else float(lam)
        with torch.no_grad():
            shifts, tau, lnf = self._split(theta.double())
            t = torch.exp(-tau.view(-1, 1, 1, 1) * self.dhat.double())
            wscale = self._wscale(lnf, torch.float64)
            ghat = self.solve_dense(shifts, t, dtype=torch.float64, lam=lam,
                                    wscale=wscale)
            chi2 = self._chi2_at(shifts, t, ghat, wscale=wscale)
        return chi2

    # ------------------------------------------------------- diagnostics ----
    def certificate(self, theta, **lsqr_kw):
        """Optimality certificate for the LSQR path at ``theta``.

        For ANY template x the algebraic identity

            chi2_explicit(x) - (C - b^T x) = -(A x)^T r ,   r = z - A x,

        holds, so the cancellation-prone shortcut ``C - b^T x`` and the
        safe explicit residual can be cross-checked without extra solves.
        At the optimum (H + lam) x = b the gap does not vanish but equals
        -lam ||x||^2 (equivalently: the shortcut equals the full ridged
        objective chi2 + lam ||x||^2 -- exactly the quadratic part of the
        marginal), so the consistency measure is
        ``gap_residual = gap_identity + penalty``: it vanishes at the
        optimum (for lam -> 0 solves, damp = 0.0, the gap itself does).
        This is a NECESSARY condition, not a tight bound on the objective
        distance; use verify()'s dquad_lsqr_damped_* keys for the actual
        objective error.  In float32 the shortcut additionally carries
        catastrophic-cancellation noise of order eps * C (~0.05..0.1 chi2
        here) even at convergence -- which is why the hot path always
        evaluates the explicit residual and the shortcut lives only here.
        """
        with torch.no_grad():
            shifts, tau, lnf = self._split(theta)
            t = torch.exp(-tau.view(-1, 1, 1, 1) * self.dhat)
            wscale = self._wscale(lnf, t.dtype)
            x = self.solve_lsqr(shifts, t, wscale=wscale, **lsqr_kw)
            b, n = shifts.shape[:2]
            a_op, at_op = self._ops(shifts, t, wscale)
            xf = x.reshape(b, -1)
            ax = a_op(xf)
            z = self._weighted_data(wscale, t.dtype, b)
            r = z - ax
            explicit = r.square().reshape(b, -1).sum(1)
            bvec = at_op(z)
            c_scaled = z.square().reshape(b, -1).sum(1)    # C = sum w y^2/f^2
            shortcut = c_scaled - (bvec * xf).sum(1)
            gap = -(ax * r).reshape(b, -1).sum(1)
            damp = lsqr_kw.get("damp", None)
            if damp is None:                       # auto: the model lam
                lam = torch.full((b,), self.lam, dtype=x.dtype,
                                 device=x.device)
            else:
                lam = torch.as_tensor(damp, dtype=x.dtype,
                                      device=x.device).expand(b).square()
            penalty = lam * xf.square().sum(1)
        return {"chi2_explicit": explicit, "chi2_shortcut": shortcut,
                "gap_identity": gap, "penalty": penalty,
                "gap_residual": gap + penalty,
                "shortcut_error": explicit - shortcut}

    def error_map(self, theta, reference=True, scaled=True):
        """Residuals and standardized-residual maps (reference f64 solve by
        default: report-quality templates should not depend on the hot
        path's dtype or warm starts).

        The template is always solved with the SCALED weights w/f^2 of
        ``theta``.  ``chi`` is the standardized residual (y - m) sqrt(w) / f
        under the fitted noise scales (``scaled=True``, the likelihood's
        units), or (y - m) sqrt(w) under the NOMINAL noise model
        (``scaled=False``: the goodness-of-fit map of the fixed-noise
        model; per epoch the two differ by the factor f_i)."""
        with torch.no_grad():
            shifts, tau, lnf = self._split(theta)
            t = torch.exp(-tau.view(-1, 1, 1, 1) * self.dhat)
            wscale = self._wscale(lnf, self.w.dtype)
            if reference:
                ghat = self.solve_dense(shifts.double(), t.double(),
                                        dtype=torch.float64,
                                        wscale=wscale.double()).to(self.w.dtype)
            else:
                ghat = self.solve_template(shifts, t, wscale=wscale)
            b = shifts.shape[0]
            hw = self.size
            g = ghat.unsqueeze(1).expand(b, self.n, hw, hw)
            model = t * shift(g.reshape(-1, hw, hw), shifts.reshape(-1, 2)
                              ).reshape(b, self.n, hw, hw)
            residuals = self.y - model
            chi = residuals * self.w.sqrt()
            if scaled:
                chi = chi * wscale.sqrt()
        return residuals, chi

    def pairwise_chi2(self, tau: float) -> torch.Tensor:
        """The old pairwise statistic with the CORRECT weights (see v04
        docstring: identical to the profiled chi-square exactly when all
        total shifts vanish -- the common-grid identity; the comparison is
        against the lam -> 0 profile, so evaluate the profiled side with
        ``profiled_chi2(theta, lam=1e-12 * self._maxdiag_ref)``)."""
        t = torch.exp(-float(tau) * self.dhat).double()          # (N, H, W)
        w, y = self.w.double(), self.y.double()
        num = torch.zeros_like(w[0])
        for i in range(self.n):
            for j in range(i + 1, self.n):
                num += w[i] * w[j] * (t[j] * y[i] - t[i] * y[j]).square()
        den = (w * t * t).sum(0)
        chi2 = torch.where(den > 0, num / den.clamp_min(1e-30), torch.zeros_like(num))
        return chi2.sum()

    # ------------------------------------------------------------- the audit
    def verify(self, theta=None, fd_components=None, fd_h=1.0e-4,
               lsqr_budgets=(64, 128, 256), verbose=True):
        """Machine-checked audit (the reviewer's approximate-vs-exact
        comparison, institutionalized): run on the REAL data before
        trusting any number from the fast path.  Returns a dict; prints a
        summary when ``verbose``.

        theta defaults to three points: the nominal registration at
        tau' = 0 with nominal noise (exact half-pixel worst case if the
        nominal shifts contain any), a randomly perturbed registration at
        tau' = 1e-2 with nominal noise, and the same perturbed registration
        with noise scales f ~ 2 (where the v07 posterior lives).  The
        conditioning documentation (``H_cond_unridged``, ``lam_over_maxeig``,
        ``n_modes_above_lam``) refers to the FIRST point, i.e. the nominal
        weights the empirical-Bayes lam was derived from.

        New in v06 (the marginal introduces exactly one new numerical
        object, the f32 log-determinant, and one new model knob, lam):
        ``logdet_chol_vs_eig`` (f64 Cholesky ln det against an
        independent eigendecomposition sum ln(sigma_j + lam)),
        ``dlogdet_f32_vs_ref`` / ``dquad_f32_vs_ref`` /
        ``dlnl_marg_f32_vs_ref`` (hot path against the f64 reference,
        split and total -- compare against your dlogz budget),
        ``dlnl_lam_x0.1`` / ``dlnl_lam_x10`` (PRIOR-sensitivity of lnL at
        the test points; a model property, not an error), and
        ``dlogdet_dtau_fd`` (the finite-difference slope of the
        nuisance-volume term along tau' -- the size of the profile-vs-
        marginal difference on this data set, on record).

        New in v07 (the noise scales introduce no new numerical object --
        they rescale the weights -- but one new implementation path):
        ``dlnl_noise_scale_identity`` (a second instance whose weight map
        is scaled by f^-2 epoch by epoch, with its data normalization
        recomputed, must reproduce THIS instance's f64 marginal at ln f
        exactly; the definitive check that the scale enters H, b, chi2 and
        the normalization consistently), the FD gradient check now
        covers ln f_0, and ``dlogdet_dlnf_fd`` records the volume-term
        slope along ln f_0 (H shrinks with f, so the volume term pushes
        toward LARGER f -- an O(1) effect against the O(n_i) normalization).
        """
        dev = self.w.device
        n_off = self.n_offsets
        if theta is None:
            g = torch.Generator(device=dev).manual_seed(0)
            th = torch.zeros(3, self.ndim)
            pert = (torch.rand((n_off,), generator=g, device=dev)
                    - 0.5).to(th.device)
            th[1, :n_off] = pert
            th[2, :n_off] = pert
            if self.noise_scale:
                th[2, self.i_lnf] = math.log(2.0) + 0.2 * (
                    torch.rand((self.n,), generator=g, device=dev) - 0.5
                ).to(th.device)
            th[:, -1] = tensor([0.0, 1.0e-2, 1.0e-2]).to(th.device)
            theta = th
        theta = torch.atleast_2d(theta)
        if fd_components is None:
            fd_components = (0, 1, n_off, -1) if self.noise_scale \
                else (0, 1, -1)
        out = {}
        with torch.no_grad():
            shifts, tau, lnf = self._split(theta.double())
            t64 = torch.exp(-tau.view(-1, 1, 1, 1) * self.dhat.double())
            ws64 = self._wscale(lnf, torch.float64)
            ws32 = ws64.float()

            # (a) adjoint identity <S x, y> = <x, S^T y> in f32 and f64
            for dt, tag in ((torch.float32, "f32"), (torch.float64, "f64")):
                gr = torch.Generator(device=dev).manual_seed(1)
                xr = torch.rand((4, self.size, self.size), generator=gr,
                                device=dev).to(dt)
                yr = torch.rand((4, self.size, self.size), generator=gr,
                                device=dev).to(dt)
                sr = (torch.rand((4, 2), generator=gr, device=dev).to(dt)
                      - 0.5) * 3.0
                lhs = (shift(xr, sr) * yr).sum(dim=(1, 2))
                rhs = (xr * shift(yr, -sr)).sum(dim=(1, 2))
                out[f"adjoint_rel_{tag}"] = float(
                    ((lhs - rhs).abs() / lhs.abs().clamp_min(1e-30)).max())

            # (b) stencil vs column H (both f64, independent code paths)
            h_st = self._dense_from_planes(
                self._stencil_planes(shifts, t64, torch.float64, wscale=ws64))
            h_col = self._columns_H(shifts, t64, wscale=ws64)
            out["stencil_vs_columns_rel"] = float(
                (h_st - h_col).abs().max() / h_col.abs().max())

            # conditioning documentation (unridged H) + where lam sits --
            # at the FIRST test point (nominal weights, the ones lam was
            # derived from; v06 quoted the last point, which is now f != 1)
            ev = torch.linalg.eigvalsh(h_st)               # (B, P) ascending
            evl = ev[0]
            out["H_cond_unridged"] = float(
                evl.max() / evl[evl > evl.max() * 1e-15].min())
            out["H_tail_1e-4_frac"] = float(
                ((evl > 0) & (evl < evl.max() * 1e-4)).sum() / (evl > 0).sum())
            out["lam_over_maxeig"] = float(self.lam / evl.max())
            out["n_modes_above_lam"] = float((evl > self.lam).sum())

            # (c) f64 marginal reference + independent log-det cross-check
            # via the eigendecomposition: ln det(H + lam I) =
            # sum_j ln(sigma_j + lam) -- a genuinely different algorithm
            # from the Cholesky factor (torch's slogdet proved
            # bit-identical to the Cholesky route on some builds, i.e. not
            # independent).  Expect ~1e-9..1e-7 from eigensolver roundoff
            # on the floored tail.
            ref = self.reference_parts(theta)
            out["lnl_reference"] = ref["lnl"].cpu()
            ld_ind = torch.log(ev + self.lam).sum(-1)
            out["logdet_chol_vs_eig"] = float(
                (ref["logdet"] - ld_ind).abs().max())

            # hot path (f32 dense, model lam) vs f64 reference: quadratic
            # part, log-determinant, and the full marginal separately
            x32, ld32 = self.solve_dense(shifts.float(), t64.float(),
                                         return_logdet=True, wscale=ws32)
            quad32 = (self._chi2_at(shifts, t64, x32.double(), wscale=ws64)
                      + self.lam * x32.double().square().sum(dim=(1, 2)))
            out["dquad_f32_vs_ref"] = float((quad32 - ref["quad"]).abs().max())
            out["dlogdet_f32_vs_ref"] = float(
                (ld32 - ref["logdet"]).abs().max())
            out["dlnl_marg_f32_vs_ref"] = float(
                (self.forward(theta.float()) - ref["lnl"]).abs().max())

            # strictly-all-f32 evaluation (f32 solve + f32 accumulation)
            keep = self.accumulate_f64
            self.accumulate_f64 = False
            l32 = self.forward(theta.float())
            self.accumulate_f64 = keep
            out["dlnl_strict_f32_eval"] = float((l32 - ref["lnl"]).abs().max())

            # (d) PRIOR sensitivity: the marginal at lam/10 and 10*lam
            # (constants included consistently).  A property of the MODEL
            # family, not an error metric; the paper-grade version is the
            # single-run reweighting (``single_run_sensitivity``), which
            # these two-point numbers cross-check.
            for fac in (0.1, 10.0):
                alt = self.reference_parts(theta, lam=self.lam * fac)
                out[f"dlnl_lam_x{fac:g}"] = float(
                    (alt["lnl"] - ref["lnl"]).abs().max())

            # (e) f32 PD margin: smallest relative ridge at which the f32
            # Cholesky of the stencil-assembled f32 H stays PD; the model
            # lam (as a fraction of the local max diag) should sit
            # comfortably above this, or the f64-retry path will engage
            h32 = self._dense_from_planes(
                self._stencil_planes(shifts.float(), t64.float(),
                                     torch.float32, wscale=ws32))
            idn32 = torch.eye(h32.shape[-1], dtype=torch.float32,
                              device=h32.device)
            dmax32 = h32.diagonal(dim1=-2, dim2=-1).amax(-1)
            rel = torch.full_like(dmax32, 1.0e-8)
            min_ridge = float("nan")
            for _ in range(10):
                _, info = torch.linalg.cholesky_ex(
                    h32 + (rel * dmax32).view(-1, 1, 1) * idn32)
                if int(info.max()) == 0:
                    min_ridge = float(rel.max())
                    break
                rel = torch.where(info > 0, rel * 10.0, rel)
            out["min_pd_ridge_f32_dense"] = min_ridge
            out["lam_rel_at_test"] = float(self.lam / dmax32.double().max())

            # (f) matrix-free fallback vs the SAME system (damped LSQR,
            # delta = sqrt(lam)); measured on the quadratic part
            for k in lsqr_budgets:
                xk = self.solve_lsqr(shifts.float(), t64.float(), iters=k,
                                     atol=0.0, wscale=ws32)
                quadk = (self._chi2_at(shifts, t64, xk.double(), wscale=ws64)
                         + self.lam * xk.double().square().sum(dim=(1, 2)))
                out[f"dquad_lsqr_damped_f32_k{k}_vs_ref"] = float(
                    (quadk - ref["quad"]).abs().max())

            # ... and vs the lam -> 0 profile (damp = 0.0): documents the
            # stall/divergence of the pure least-squares solve (expected
            # to be catastrophic at real-data kappa(H) ~ 1e13)
            lam0 = 1.0e-12 * self._maxdiag_ref
            ref0 = self.profiled_chi2(theta, lam=lam0)
            for k in lsqr_budgets:
                xk = self.solve_lsqr(shifts.float(), t64.float(), iters=k,
                                     atol=0.0, damp=0.0, wscale=ws32)
                out[f"dchi2_lsqr_f32_k{k}_vs_lam0"] = float(
                    (self._chi2_at(shifts, t64, xk.double(), wscale=ws64)
                     - ref0).abs().max())

            # (g) shortcut cancellation in f32 (why the hot path never uses
            # it) + the damped optimality certificate (gap_residual -> 0)
            cert = self.certificate(theta.float(), iters=64)
            out["shortcut_err_f32"] = float(cert["shortcut_error"].abs().max())
            out["gap_identity_f32"] = float(cert["gap_identity"].abs().max())
            out["gap_residual_f32"] = float(cert["gap_residual"].abs().max())

            # (i) v07 noise-scale identity: a SECOND instance with the
            # weight map scaled by f^-2 (and c_data, C recomputed from it)
            # evaluated at ln f = 0 must reproduce THIS instance at ln f --
            # same lam, same template prior.  Exercises H, b, chi2 and the
            # normalization together.  The check uses DISTINCT powers of
            # two per epoch (0.5, 1, 2, 0.5, ...): exact in the f32 weight
            # buffer of the copy, so the residual is f64 summation-order
            # roundoff (~1e-9), and a scale applied to the wrong epoch
            # axis cannot cancel.
            if self.noise_scale:
                f_chk = torch.as_tensor(
                    2.0 ** ((np.arange(self.n) % 3) - 1.0), dtype=torch.float64,
                    device=dev)
                th_chk = theta[-1:].double().clone()
                th_chk[0, self.i_lnf] = f_chk.log()
                th_alt = th_chk.clone()
                th_alt[0, self.i_lnf] = 0.0
                alt = self._rescaled_copy(f_chk)
                out["dlnl_noise_scale_identity"] = float(
                    (alt.reference_logl(th_alt)
                     - self.reference_logl(th_chk)).abs().max())
                # and the legacy (offsets, tau) layout == ln f = 0 exactly
                th_leg = torch.cat([theta[:1, :n_off], theta[:1, -1:]], dim=1)
                out["dlnl_legacy_layout"] = float(
                    (self.reference_logl(th_leg) - ref["lnl"][:1]).abs().max())

        # (h) envelope gradient (f32 hot path) vs central FD of the f64
        # reference QUADRATIC part (plus the analytic f-normalization, so
        # that the ln f components are covered too).  With the penalty
        # included in the objective the envelope is now EXACT (the v05
        # ridge-coupling bias is gone), so this should agree to FD
        # accuracy.  The log-det part is deliberately outside autograd;
        # its FD slopes along tau' and ln f_0 are reported separately as
        # the size of the nuisance-volume effect.
        th1 = theta[-1:].float().clone().requires_grad_(True)
        lnl = self.forward(th1)
        grad = torch.autograd.grad(lnl.sum(), th1)[0][0]
        fd_quad, fd_logdet = [], []
        for c in fd_components:
            tp = theta[-1:].double().clone()
            tm = theta[-1:].double().clone()
            tp[0, c] += fd_h
            tm[0, c] -= fd_h
            pp = self.reference_parts(tp)
            pm = self.reference_parts(tm)
            fd_quad.append(float(((-0.5 * (pp["quad"] - pm["quad"]))
                                  + (self.lnl_const(lnf=pp["lnf"])
                                     - self.lnl_const(lnf=pm["lnf"])))
                                 / (2 * fd_h)))
            fd_logdet.append(float((pp["logdet"] - pm["logdet"])
                                   / (2 * fd_h)))
        fd_quad = torch.tensor(fd_quad).cpu()
        ga = grad[list(fd_components)].double().cpu()
        out["grad_quad_fd_rel"] = float(((ga - fd_quad).abs()
                                         / fd_quad.abs().clamp_min(1e-6)).max())
        comps = list(fd_components)
        out["dlogdet_dtau_fd"] = float(fd_logdet[comps.index(-1)]) \
            if -1 in comps else float("nan")
        out["dlogdet_dlnf_fd"] = float(fd_logdet[comps.index(n_off)]) \
            if (self.noise_scale and n_off in comps) else float("nan")

        if verbose:
            print("--- verify() -------------------------------------------")
            for k, v in out.items():
                if isinstance(v, float):
                    print(f"{k:34s} {v: .3e}")
            print("--------------------------------------------------------")
        return out

    def _rescaled_copy(self, f):
        """Deep copy of this instance whose NOMINAL weights are w_i / f_i^2
        (``f`` (N,)), with every weight-derived quantity recomputed and
        the same lam -- the reference of verify()'s noise-scale identity."""
        import copy
        alt = copy.deepcopy(self)
        f2 = f.to(self.w.dtype).view(self.n, 1, 1).square()
        alt.w = (self.w / f2).contiguous()
        alt.sqw = alt.w.sqrt().contiguous()
        alt.c_const = (alt.w.double() * alt.y.double().square()).sum()
        alt._c_data = self._data_norm(alt.w) if self.include_data_norm \
            else 0.0
        return alt

    def _chi2_at(self, shifts, t, x, wscale=None, per_epoch=False):
        """Explicit chi2 at an ARBITRARY template x (f64 path), with the
        weights scaled by ``wscale`` = f^-2 when given; (B,) or, with
        ``per_epoch``, the (B, N) per-epoch split."""
        b = shifts.shape[0]
        hw = self.size
        g = x.double().reshape(b, 1, hw, hw).expand(b, self.n, hw, hw)
        model = t.double() * shift(
            g.reshape(-1, hw, hw), shifts.double().reshape(-1, 2)
        ).reshape(b, self.n, hw, hw)
        r2 = self.w.double() * (self.y.double() - model).square()
        if wscale is not None:
            r2 = r2 * wscale.double()
        per = r2.sum(dim=(2, 3))
        return per if per_epoch else per.sum(-1)


# --------------------------------------------------------------------------- #
def single_run_sensitivity(ll, points, weights, stored_logl=None,
                           lam_factors=(0.1, 1.0 / 3.0, 3.0, 10.0),
                           tau_soft=1.0e-2, tau_hi=1.0, max_points=16384,
                           batch=512, rb_bins=16, out_path=None,
                           verbose=True):
    """Prior-sensitivity table from ONE nested-sampling run: no reruns.

    Importance reweighting at the stored (physical-space) posterior points:
      * template-prior ladder lam -> fac * lam: the LIKELIHOOD changes, so
        the new weights are w0 * exp(lnL_fac - lnL_base), both sides
        evaluated with the float64 reference marginal at the same points
        (a few thousand chunked f64 solves per rung -- minutes, once);
      * tau'-prior variant flat -> soft-log: the likelihood is unchanged,
        so the weights are the pure prior ratio p_soft(tau')/p_flat(tau').

    All weight combinations happen in log space, shifted by the maximum of
    the COMBINED log-weight (see ``reweight``): a negligible-weight dead
    point with an extreme dlnL must not drag the bulk into underflow.

    Why the raw ESS drops on the lam rungs, and what is done about it:
    changing lam re-prices the shell of template modes whose eigenvalues
    sit within ~a decade of the floor, and those near-floor eigenvalues /
    projections are the most theta-sensitive quantities in the model (they
    ride the sub-pixel phase of the bicubic stencil; each carries an
    O(chi2_1) noise coefficient).  The resulting dlnL therefore scatters
    by O(1) nats ACROSS theta AT FIXED tau' -- almost orthogonal to tau'
    (the soft-log row, whose weight depends on tau' alone, keeps near-full
    ESS; the medians shift far too little for a tau'-tilt to explain the
    ESS).  Orthogonal scatter adds quantile noise but no bias, and the
    tau'-marginal only needs E[weight | tau'], so the quoted quantiles are
    RAO-BLACKWELLIZED: per-tau'-bin conditional mean weights
    (``rb_bins`` equal-base-weight bins), which removes the orthogonal
    scatter, preserves any true tau'-dependence, and leaves dlnZ exactly
    unchanged.  Each lam row reports the RAW Kish ESS (the honest overlap
    diagnostic) next to the RB ESS; rows under ~200 raw are flagged as
    indicative, and lowering lam costs more ESS than raising it because it
    ACTIVATES the denser decade of modes below the floor.  An explicit
    rerun (ctor ``lam=``) remains the escape hatch if an outer rung must
    be quoted with precision.  Each row also reports the evidence shift
    dlnZ = ln E_w0[ratio].  When ``stored_logl`` is given (the sampler's
    own values at those points), the maximum |f32 run - f64 reference| is
    printed too: the run-level version of verify()'s
    ``dlnl_marg_f32_vs_ref``.

    Returns the list of row dicts; writes JSON to ``out_path`` if given.
    """
    points = np.asarray(points, dtype=np.float64)
    w0 = np.asarray(weights, dtype=np.float64)
    w0 = w0 / w0.sum()
    if stored_logl is not None:
        stored_logl = np.asarray(stored_logl, dtype=np.float64)
    if points.shape[0] > max_points:            # cap the f64 re-evaluations
        rng = np.random.default_rng(0)
        idx = rng.choice(points.shape[0], size=max_points, replace=True,
                         p=w0)
        points = points[idx]
        if stored_logl is not None:
            stored_logl = stored_logl[idx]
        w0 = np.full(max_points, 1.0 / max_points)

    def f64_lnl(lam=None):
        out = []
        for i0 in range(0, points.shape[0], batch):
            th = torch.as_tensor(points[i0:i0 + batch])
            out.append(ll.reference_logl(th, lam=lam).cpu().numpy())
        return np.concatenate(out)

    def wq(x, w, qs):
        o = np.argsort(x)
        cw = np.cumsum(w[o])
        cw = cw / cw[-1]
        return [float(np.interp(q, cw, x[o])) for q in qs]

    tau = points[:, -1]
    rows = []
    with np.errstate(divide="ignore"):
        lw0 = np.log(w0)                       # -inf for zero-weight points

    def reweight(dlw):
        # Combine in LOG space and shift by the max of the COMBINED
        # log-weight ln w0 + dln.  Shifting by max(dln) alone is wrong: a
        # negligible-weight dead point near the prior (e.g. tau ~ 1, where
        # a 10x weaker template prior relieves a lot of shrinkage and
        # nearly all modes sit below lam) can carry the largest dln by
        # hundreds of nats; dividing the bulk weights by it underflows
        # every square to exact zero, so ESS becomes 0/0 = nan while dlnZ
        # and the quantiles silently survive on denormals.  Shifting by
        # the combined maximum puts the LARGEST ACTUAL WEIGHT at 1, which
        # is immune to both under- and overflow.
        dlw = np.where(np.isfinite(dlw), dlw, -np.inf)
        lw = lw0 + dlw
        m = float(lw.max())
        w = np.exp(lw - m)
        s = float(w.sum())
        return w / s, m + math.log(s)          # weights, dlnZ = ln E_w0[e^dln]

    def _lse(a):
        m = np.max(a)
        if not np.isfinite(m):
            return -np.inf
        return float(m + np.log(np.exp(a - m).sum()))

    def rb_tilt(dlw, nbins):
        # log E_base[e^dlw | tau] on equal-base-weight tau bins.
        # The tau-marginal of a variant only needs the CONDITIONAL mean of
        # the weight given tau: p_fac(tau) = p_base(tau) E[e^dln | tau].
        # Averaging weights inside tau bins (Rao-Blackwellization) removes
        # the component of dln that scatters ACROSS theta at fixed tau --
        # the near-floor mode-shell chatter that crushes the raw ESS --
        # while leaving any true tau-dependence intact.  Consistent for
        # the tau quantiles as bins shrink; the per-bin means inherit the
        # heavy tail, so the RAW ESS remains the honesty metric, and the
        # total variant mass (hence dlnZ) is unchanged by construction.
        dlw = np.where(np.isfinite(dlw), dlw, -np.inf)
        order = np.argsort(tau, kind="stable")
        cw = np.cumsum(w0[order])
        cw = cw / cw[-1]
        edges = np.searchsorted(
            cw, np.linspace(0.0, 1.0, nbins + 1)[1:-1], side="right")
        m = np.zeros_like(dlw)
        for b in np.split(order, edges):
            if b.size == 0:
                continue
            base = _lse(lw0[b])
            m[b] = (_lse(lw0[b] + dlw[b]) - base) if np.isfinite(base) else 0.0
        return m

    def add(name, w, dlnz=None, ess_raw=None):
        ess = float(w.sum() ** 2 / (w ** 2).sum())
        med, q84, q95 = wq(tau, w, (0.5, 0.841, 0.95))
        row = {"variant": name, "tau_median": med, "tau_84": q84,
               "tau_95": q95, "ess": ess if ess_raw is None else ess_raw}
        if ess_raw is not None:
            row["ess_rb"] = ess
        if dlnz is not None:
            row["dlnZ_vs_run"] = float(dlnz)
        rows.append(row)
        if verbose:
            msg = (f"{name:22s} tau med={med:.4e} 84%={q84:.4e} "
                   f"95%={q95:.4e} ESS={row['ess']:9.1f}")
            if ess_raw is not None:
                msg += f" (RB {ess:.0f})"
            if dlnz is not None:
                msg += f" dlnZ={dlnz:+.3f}"
            if row["ess"] < 200:
                msg += "  << low overlap: quantiles indicative only"
            print(msg)
        return row

    if verbose:
        print("--- single-run prior sensitivity (importance reweighting) ---")
    lnl0 = f64_lnl()
    f32_max = None
    if stored_logl is not None:
        f32_max = float(np.abs(lnl0 - stored_logl).max())
        if verbose:
            print(f"f32 run vs f64 reference at posterior points: "
                  f"max|dlnL| = {f32_max:.3e}")
    row0 = add("run prior", w0)
    if f32_max is not None:      # on record next to the run-prior row
        row0["f32_max_abs_dlnl"] = f32_max
    for fac in lam_factors:
        dl = f64_lnl(lam=float(fac) * ll.lam) - lnl0
        wf, dlnz = reweight(dl)
        ess_raw = float(wf.sum() ** 2 / (wf ** 2).sum())
        w_rb, _ = reweight(rb_tilt(dl, rb_bins))
        add(f"lam x {fac:.3g}", w_rb, dlnz=dlnz, ess_raw=ess_raw)
    # v05 soft-log tau' prior vs the flat run prior (same support [0, hi]):
    # the weight is a function of tau alone, so no RB is needed -- its
    # near-full ESS is also the built-in proof that the lam rows' ESS loss
    # comes from theta-scatter at fixed tau, not from the tau shift.
    ratio = tau_hi / ((tau + tau_soft) * math.log1p(tau_hi / tau_soft))
    wp, dlnz = reweight(np.log(ratio))
    add("soft-log tau prior", wp, dlnz=dlnz)
    if out_path is not None:
        Path(out_path).write_text(json.dumps(rows, indent=2))
    return rows

def model_card_text(ll, n_images, offset_halfwidth, tau_hi, tau_soft):
    """The MODEL_CARD.txt of a run: every prior, the theta layout and the
    constants, in the fixed wording that ``paper_values.load_model_card``
    parses (keep the two in step)."""
    n_off = ll.n_offsets
    ndim = ll.ndim
    layout = (f"offsets[0:{n_off}], lnf[{n_off}:{n_off + n_images}], "
              f"tau[{ndim - 1}]" if ll.noise_scale
              else f"offsets[0:{n_off}], tau[{ndim - 1}]")
    if ll.noise_scale:
        noise_lines = (
            f"  * per-epoch noise scales f_i, i = 0..{n_images - 1} (v07):\n"
            f"    ln f_i uniform on [ln {NOISE_SCALE_RANGE[0]:g}, "
            f"ln {NOISE_SCALE_RANGE[1]:g}], i.e. log-uniform\n"
            f"    f in [{NOISE_SCALE_RANGE[0]:g}, {NOISE_SCALE_RANGE[1]:g}]; "
            "the weights enter as w_i / f_i^2 and the\n"
            "    Gaussian normalization carries -n_i ln f_i, n_i = "
            + ", ".join(str(int(v)) for v in ll.n_weighted.cpu().numpy())
            + ";\n")
    else:
        noise_lines = "  * noise scales: NONE (fixed nominal weights, f = 1);\n"
    return (
        "Fully Bayesian run: the integrand is the MARGINAL likelihood\n"
        "(galaxy template integrated out in closed form under a proper\n"
        "Gaussian prior), so the samples are a posterior and logz is a\n"
        "Bayesian evidence under the following priors:\n"
        f"  * template g ~ N(0, lam^-1 I_P), P = {ll.size**2},\n"
        f"    lam = {ll.lam:.8e}  (sigma_g = "
        f"{ll.template_prior_std:.6e} cts/s/px);\n"
        "    empirical-Bayes scale = lam_rel_init * max diag H at the\n"
        "    nominal registration, tau' = 0 and the NOMINAL weights\n"
        "    (f = 1), computed once and frozen;\n"
        f"  * per-epoch offsets: uniform on +/- {offset_halfwidth} px about\n"
        "    the nominal registration (epoch-0 gauge);\n"
        + noise_lines
        + f"  * tau': flat on [0, {tau_hi}] (the evidence scales with the\n"
        "    inverse of this ceiling).\n"
        f"theta layout: {layout}  (tau' last; N = {n_images} epochs,\n"
        f"ndim = {ndim}).\n"
        "lnL includes the theta-independent constants c_data\n"
        f"({ll._c_data:.8e}) and P/2 ln lam; evidence\n"
        "DIFFERENCES between variants on the same weights are unaffected\n"
        "by c_data, but comparisons across different lam require the\n"
        "P/2 ln lam term (included).\n"
        "Reported values are re-evaluated with reference_logl (float64).\n"
        "Robustness (sensitivity.json, computed from THIS run by\n"
        "importance reweighting -- no extra sampling): the template-\n"
        "prior ladder lam x {1/10, 1/3, 3, 10}, and the v05 soft-log\n"
        "tau' prior\n"
        f"p(tau') = 1/[(tau' + {tau_soft}) ln(1 + {tau_hi/tau_soft:g})];\n"
        "each row lists tau' quantiles, Kish ESS (validity certificate)\n"
        "and dlnZ.  An ESS collapse (< ~5% of the points) is the signal\n"
        "that a variant needs its own run; not expected here.\n"
        "Audit numbers (verify()) are in verify.json next to this file.\n"
    )


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    observations, disk_models, masks, variance, nominal_shift_xy, q_cut = load_native_data()
    N_IMAGES = observations.shape[0]

    log_likelihood = LogLikelihood(observations, disk_models, masks, variance,
                                   nominal_shift_xy, noise_scale=True)
    PARAM_NAMES = param_names(N_IMAGES, noise_scale=log_likelihood.noise_scale)
    assert len(PARAM_NAMES) == log_likelihood.ndim
    print(f"template prior: g ~ N(0, lam^-1 I), lam = {log_likelihood.lam:.6e}"
          f"  (sigma_g = {log_likelihood.template_prior_std:.4e} cts/s/px;"
          f"  empirical-Bayes scale from the NOMINAL weights, frozen at construction)")
    print(f"lnL constants: c_data = {log_likelihood._c_data:.6e},"
          f"  P/2 ln lam = {0.5 * log_likelihood.size**2 * math.log(log_likelihood.lam):.6e}")
    print("weighted pixels per epoch n_i (coefficients of -ln f_i):",
          log_likelihood.n_weighted.cpu().numpy().astype(int).tolist())

    truths = torch.zeros(log_likelihood.ndim)  # offsets 0, ln f = 0 (f = 1)
    truths[-1] = 1.0e-2                    # tau' at the ring peak (not 10!)
    tile_truths = truths.view(1, -1).tile(2, 1)
    print("marginal lnL (f32 hot path):", log_likelihood(tile_truths))
    print("marginal lnL (f64 reference):", log_likelihood.reference_logl(tile_truths))
    truths_f2 = truths.clone()
    truths_f2[log_likelihood.i_lnf] = math.log(2.0)
    print("marginal lnL (f64 reference) at f = 2 for every epoch:",
          log_likelihood.reference_logl(truths_f2.view(1, -1)))

    # the audit -- run BEFORE trusting fast-path numbers on this data.
    # In particular check dlogdet_f32_vs_ref / dlnl_marg_f32_vs_ref against
    # the dlogz budget below; if too large, evaluate through reference_logl.
    # dlnl_noise_scale_identity must be at roundoff (v07 implementation).
    audit = log_likelihood.verify()

    # quick-look, old style
    from matplotlib import pyplot as plt

    residuals, chi = log_likelihood.error_map(truths.view(1, -1))
    # fig, axes = plt.subplots(3, 3, constrained_layout=True,
    #                          sharex=True, sharey=True, figsize=(9, 9))
    # axes = np.array(axes).flatten()
    # for i in range(N_IMAGES):
    #     im = axes[i].imshow(chi[0, i].cpu().numpy(), origin="lower",
    #                         cmap="RdBu_r", vmin=-4, vmax=4)
    #     axes[i].scatter(q_cut[i, 0], q_cut[i, 1], marker="+", c="k", s=50,
    #                     linewidths=0.75)
    #     plt.colorbar(im, ax=axes[i])
    # axes[-1].axis("off")
    # plt.show()

    # old-vs-new consistency demo (common grid): correctly weighted pairwise
    # statistic == profiled chi-square (lam -> 0), same number from two
    # derivations.  (Profile objects; the marginal adds the volume term.)
    ll0 = LogLikelihood(observations, disk_models, masks, variance,
                        torch.zeros(N_IMAGES, 2))
    print("pairwise (correct weights):", float(ll0.pairwise_chi2(1.0e-2)))
    print("profiled template (lam->0): ",
          float(ll0.profiled_chi2(truths.view(1, -1),
                                  lam=1.0e-12 * ll0._maxdiag_ref)[0]))

    # ------------------------------------------------------------- ultranest
    # Same hookup as v05; the integrand is now the MARGINAL likelihood, so
    #  - the samples ARE a posterior and "logz" IS a Bayesian evidence
    #    under the priors stated in the module docstring (and written to
    #    MODEL_CARD.txt below);
    #  - per-evaluation cost is unchanged: the log-determinant is read off
    #    the same batched f32 Cholesky the solve already performs;
    #  - the hot path is a deterministic function of theta (fixed lam, no
    #    warm starts, no iteration knobs), so the run is reproducible;
    #  - reported values come from reference_logl (f64) regardless; check
    #    verify()'s dlnl_marg_f32_vs_ref against dlogz before believing
    #    the evidence error bar;
    #  - do NOT resume a v05 (profiled) log_dir: the likelihood changed.
    import ultranest
    from ultranest.popstepsampler import (
        PopulationSliceSampler, PopulationSimpleSliceSampler,
        generate_region_oriented_direction, generate_mixture_random_direction,
    )

    NDIM = log_likelihood.ndim      # 2 (N-1) offsets + N ln f + tau'
    N_OFF = log_likelihood.n_offsets
    I_LNF = log_likelihood.i_lnf
    OFFSET_HALFWIDTH = 1.5          # px box around the nominal registration
    TAU_HI = 1.0                    # conservative ceiling; tau=1 is already rejectable
    TAU_SOFT = 1.0e-2               # scale of the ALTERNATIVE soft-log prior
                                    # (post-processing reweighting only)
    LNF_LO, LNF_HI = (math.log(NOISE_SCALE_RANGE[0]),
                      math.log(NOISE_SCALE_RANGE[1]))

    def prior_transform(cube):
        params = cube.copy()
        params[:, :N_OFF] = (2.0 * cube[:, :N_OFF] - 1.0) * OFFSET_HALFWIDTH

        # v07: per-epoch noise scales, ln f_i uniform on [ln lo, ln hi]
        # (log-uniform f_i: the Jeffreys prior for a scale; the nominal
        # noise model f = 1 and the empirical sqrt(MSR) ~ 2 of the v06
        # run both sit well inside).  The ln f block is well constrained
        # (~1/sqrt(2 n_i) ~ 0.05 wide), so the bounds only need to
        # contain the answer; the evidence scales with the inverse of
        # the prior width per epoch.
        if log_likelihood.noise_scale:
            params[:, I_LNF] = LNF_LO + cube[:, I_LNF] * (LNF_HI - LNF_LO)

        # FLAT prior on tau' in [0, TAU_HI]: the simplest proper choice
        # and the conservative one for an upper limit (no upweighting of
        # the tau' = 0 neighbourhood).  The prior transform is
        # computationally free either way -- run length is dominated by
        # the 2*(N-1) offset dimensions, so this choice is statistical,
        # not a cost item.  The v05 soft-log map (density
        # 1/[(tau'+TAU_SOFT) ln(1+TAU_HI/TAU_SOFT)], linear near the
        # null, ~log above TAU_SOFT) is recovered from the SAME run by
        # prior-ratio reweighting in single_run_sensitivity below.
        # Under a flat prior the EVIDENCE scales with 1/TAU_HI: state the
        # ceiling wherever lnZ is quoted.
        params[:, NDIM - 1] = cube[:, NDIM - 1] * TAU_HI
        return params

    def wrapper_log_likelihood(theta):
        theta = torch.as_tensor(np.asarray(theta, dtype=np.float32))
        with torch.inference_mode():
            out = log_likelihood(theta)
        out = out.detach().cpu().numpy().astype(np.float64)
        return np.where(np.isfinite(out), out, -1.0e30)

    # fresh directory: the v07 likelihood has N more parameters than v06
    LOG_DIR = "ultranest_run/marginal_common_tau_noisescale_v16"
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    # the audit numbers, machine-readable for paper_values.py (which
    # otherwise falls back to hard-coded defaults from an older run)
    (Path(LOG_DIR) / "verify.json").write_text(json.dumps(
        {k: v for k, v in audit.items() if isinstance(v, float)}, indent=2))

    sampler = ultranest.ReactiveNestedSampler(
        PARAM_NAMES, wrapper_log_likelihood, prior_transform,
        vectorized=True,
        log_dir=LOG_DIR,
        resume="resume",    # fresh dir above -- never resume a v05/v06 run
        ndraw_min=128,
        ndraw_max=1024,
    )
    sampler.stepsampler = PopulationSimpleSliceSampler( # PopulationSimpleSliceSampler( # PopulationSliceSampler(
        popsize=128,        # headroom restored; raise further on big GPUs
        nsteps=32,
        generate_direction=generate_mixture_random_direction,
    )
    sampler.run(
        min_num_live_points=521,
        dlogz=0.5,          # posterior-grade; tighten (e.g. 0.1) if the
                            # EVIDENCE itself is a headline number
        min_ess=512,
        update_interval_volume_fraction=0.4,
        max_num_improvement_loops=4,
    )
    sampler.print_results()
    sampler.plot() # axes_scale=["linear"] * (NDIM - 1) + ["log"])

    # single-run sensitivity: lam ladder (likelihood reweighting,
    # f64 reference at the stored points) and the v05 soft-log tau' prior
    # (prior-ratio reweighting) -- no additional sampling.  Check the ESS
    # column before quoting any variant.
    ws = sampler.results["weighted_samples"]
    single_run_sensitivity(
        log_likelihood,
        np.asarray(ws["points"]), np.asarray(ws["weights"]),
        stored_logl=np.asarray(ws["logl"]) if "logl" in ws else None,
        tau_soft=TAU_SOFT, tau_hi=TAU_HI,
        out_path=Path(LOG_DIR) / "sensitivity.json",
    )

    (Path(LOG_DIR) / "MODEL_CARD.txt").write_text(
        model_card_text(log_likelihood, N_IMAGES, OFFSET_HALFWIDTH, TAU_HI,
                        TAU_SOFT))
    print("note: this run maps the MARGINAL likelihood -- samples are a "
          "posterior and logz an evidence under the priors in MODEL_CARD.txt")