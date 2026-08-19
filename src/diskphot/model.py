r"""Joint forward model and exactly profiled log-likelihood.

Model
-----
Each disk-subtracted epoch, on its own native pixel grid, is modelled as a
single shared galaxy image :math:`G` (free, nonparametric, defined on the
epoch-0 grid), shifted to the epoch position and attenuated by the trial
extinction field:

.. math::

    y_i(u) = T_i(u)\,[S_{s_i} G](u) + n_i(u),
    \qquad n_i(u) \sim \mathcal N\bigl(0, \sigma_i^2(u)\bigr),

with the total shift :math:`s_i = s_i^{\rm nom} + \delta_i` composed of the
known nominal sub-pixel geometry (from the integer-crop bookkeeping of
:mod:`diskphot.data`) and the fitted residual offsets
(:math:`\delta_0 \equiv 0`).  The transmission is
:math:`T_i = \exp[-\tau'\,\hat D_i]` (common :math:`\tau'`; variants:
per-epoch :math:`\tau'_i \ge 0`, signed grey screens
:math:`e^{-\gamma_i}` with :math:`\gamma_0 \equiv 0`, or none).

Exact profiling of the template
-------------------------------
Let :math:`W_i = \mathrm{diag}(w_i)` with the *fixed* weights
:math:`w_i = m_i\,M_i/\sigma_i^2` (validity x analysis aperture / variance)
and :math:`A_i = T_i S_{s_i}`.  The conditional maximum-likelihood template
solves the normal equations

.. math::

    H\,\hat G = b, \qquad
    H = \sum_i A_i^{\top} W_i A_i, \qquad
    b = \sum_i A_i^{\top} W_i\, y_i .

Because interpolation couples neighbouring template pixels, :math:`H` is
not diagonal; the pixelwise ratio :math:`B/C` used previously is only the
diagonal (shift-and-add) approximation and is *not* the optimizer of the
evaluated objective.  Here the normal equations are solved with a
preconditioned conjugate-gradient iteration in which every operator is
exact:

* :math:`S_{s}^{\top} = S_{-s}` holds to machine precision for the
  symmetric, explicitly implemented kernels of :mod:`diskphot.transforms`
  (zero padding included), so :math:`H` is applied matrix-free as
  :math:`v \mapsto \sum_i S_{-s_i}[\,w_i T_i^2\, (S_{s_i} v)\,]` and is
  exactly symmetric PSD;
* the Jacobi preconditioner uses the *exact* diagonal
  :math:`\mathrm{diag}(H)_q = \sum_i \sum_p k^2(p - s_i - q)\,
  w_i(p)\,T_i^2(p)` from the squared-tap back-projection (nonnegative by
  construction -- no clamped, possibly negative interpolated weights);
* the iteration starts from the nonnegative bilinear shift-and-add
  estimate and stops when the chi-square decrease over a trailing window
  falls below ``cg.tol_chi2``, so the returned statistic is a certified
  lower bound of the exact profile log-likelihood within
  ``tol_chi2 / 2``.

The same weights :math:`w_i` appear in :math:`H`, :math:`b`, and the
residual chi-square, and they do not depend on fitted parameters, so the
profiled statistic

.. math::

    \ln\mathcal L_{\rm p}(\tau', \{\delta_i\})
    = -\tfrac12 \sum_i \sum_u w_i\,
      \bigl(y_i - T_i\,[S_{s_i}\hat G]\bigr)^2

is (up to a parameter-independent constant) the exact concentrated Gaussian
log-likelihood of the stated model.  It is a *profile* likelihood: its
sampling properties (detection threshold, confidence limits, coverage) are
calibrated by simulation in :mod:`diskphot.calibrate`, not read off a
nested-sampling integral.  A flat-prior marginalization of :math:`G` would
instead add :math:`-\tfrac12 \ln\det H(\theta)`, which diverges as
:math:`\tau' \to \infty` (:math:`H \to 0` on attenuated sightlines) and
makes the marginal improper; a proper image prior would render it finite at
the price of per-call determinants and image-prior sensitivity.  Neither is
used here.

Gradients use the envelope theorem: :math:`\hat G` is the (numerically
converged) optimizer of the same objective, so
:math:`\partial_{\theta} \ln\mathcal L_{\rm p} =
\partial_{\theta} \ln\mathcal L|_{G = \hat G}` and the CG solve runs
detached from autograd.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import CGConfig, Variant
from .data import EpochData
from .transforms import shift2d

__all__ = ["JointLogLikelihood"]


class JointLogLikelihood(nn.Module):
    """Vectorized, exactly profiled joint log-likelihood over all epochs.

    Parameters
    ----------
    data
        Native-grid epoch data (already moved to the target device/dtype).
    variant
        Extinction-model variant.
    cg
        Conjugate-gradient settings of the template solve.
    observations
        Optional stack of raw count-rate observations, shape
        ``(R, N, H, W)``: ``R`` independent realizations sharing the
        geometry, weights and disk model of ``data`` (used by the batched
        injection--recovery fits).  Default: the single real data set.
    kernel
        Interpolation kernel of the model warp (see
        :mod:`diskphot.transforms`).
    """

    def __init__(
        self,
        data: EpochData,
        variant: Variant = Variant.COMMON,
        cg: CGConfig | None = None,
        observations: Tensor | None = None,
        kernel: str = "catmull-rom",
    ) -> None:
        super().__init__()
        self.variant = variant
        self.cg = cg or CGConfig()
        self.kernel = kernel
        self.n_epochs = data.n_epochs

        if observations is None:
            y = data.galaxy_only.unsqueeze(0)  # (1, N, H, W)
        else:
            if observations.ndim != 4 or observations.shape[1:] != data.observations.shape:
                raise ValueError(
                    "observations must have shape (R, N, H, W) matching the data"
                )
            y = (observations - data.disk.unsqueeze(0)).to(data.observations)
        w = data.weights
        if float(w.sum()) <= 0:
            raise ValueError("empty analysis weights; check masks / variance")

        self.register_buffer("y", y.contiguous())
        self.register_buffer("w", w.contiguous())
        self.register_buffer("disk_hat", data.disk_hat.clone())
        self.register_buffer("nominal", data.nominal_shift_xy.clone())
        self.register_buffer("mask", data.mask.clone())
        self.register_buffer("valid", data.valid.clone())

    # ------------------------------------------------------------ metadata --
    @property
    def n_realizations(self) -> int:
        return int(self.y.shape[0])

    @property
    def n_offset_params(self) -> int:
        return 2 * (self.n_epochs - 1)

    @property
    def n_tau_params(self) -> int:
        match self.variant:
            case Variant.COMMON:
                return 1
            case Variant.PER_EPOCH:
                return self.n_epochs
            case Variant.GREY:
                return self.n_epochs - 1  # gamma_0 = 0 gauge, others signed
            case Variant.NULL:
                return 0

    @property
    def n_params(self) -> int:
        return self.n_offset_params + self.n_tau_params

    @property
    def param_names(self) -> list[str]:
        names = [f"{p}_{i}" for i in range(1, self.n_epochs) for p in ("dx", "dy")]
        match self.variant:
            case Variant.COMMON:
                names += ["tau"]
            case Variant.PER_EPOCH:
                names += [f"tau_{i}" for i in range(self.n_epochs)]
            case Variant.GREY:
                names += [f"gamma_{i}" for i in range(1, self.n_epochs)]
            case Variant.NULL:
                pass
        return names

    @property
    def param_labels(self) -> list[str]:
        labels = [
            rf"$\Delta {a}_{{{i}}}$" for i in range(1, self.n_epochs) for a in ("x", "y")
        ]
        match self.variant:
            case Variant.COMMON:
                labels += [r"$\tau^{\prime}$"]
            case Variant.PER_EPOCH:
                labels += [rf"$\tau^{{\prime}}_{{{i}}}$" for i in range(self.n_epochs)]
            case Variant.GREY:
                labels += [rf"$\gamma_{{{i}}}$" for i in range(1, self.n_epochs)]
            case Variant.NULL:
                pass
        return labels

    @property
    def n_weighted_pixels(self) -> int:
        """Number of pixels with nonzero likelihood weight (all epochs)."""
        return int((self.w > 0).sum())

    # ---------------------------------------------------------- components --
    def _split(self, theta: Tensor) -> tuple[Tensor, Tensor | None]:
        """Split ``(B, P)`` parameters into offsets ``(B, N, 2)`` and taus."""
        b = theta.shape[0]
        n_off = self.n_offset_params
        delta = theta.new_zeros((b, self.n_epochs, 2))
        delta[:, 1:] = theta[:, :n_off].view(b, self.n_epochs - 1, 2)
        taus = theta[:, n_off:] if self.n_tau_params else None
        return delta, taus

    def transmission(self, taus: Tensor | None, batch: int) -> Tensor:
        """Per-epoch transmission fields ``(B, N, H, W)`` in the epoch frames."""
        h, w = self.disk_hat.shape[-2:]
        match self.variant:
            case Variant.NULL:
                return self.disk_hat.new_ones((batch, self.n_epochs, h, w))
            case Variant.COMMON:
                assert taus is not None
                exponent = taus.view(batch, 1, 1, 1) * self.disk_hat
            case Variant.PER_EPOCH:
                assert taus is not None
                exponent = taus.view(batch, self.n_epochs, 1, 1) * self.disk_hat
            case Variant.GREY:
                assert taus is not None
                gam = torch.cat([taus.new_zeros(batch, 1), taus], dim=1)
                exponent = gam.view(batch, self.n_epochs, 1, 1).expand(-1, -1, h, w)
        return torch.exp(-exponent)

    def _shift(self, fields: Tensor, shifts: Tensor, **kw) -> Tensor:
        """Shift ``(B, N, H, W)`` fields by per-epoch ``(B, N, 2)`` shifts."""
        b, n, h, w = fields.shape
        return shift2d(
            fields.reshape(b * n, h, w), shifts.reshape(b * n, 2), **kw
        ).view(b, n, h, w)

    # --------------------------------------------------- template profiling --
    def _normal_apply(self, v: Tensor, shifts: Tensor, wt2: Tensor) -> Tensor:
        """Apply the normal operator ``H v`` for template images ``v (B,H,W)``."""
        b, n = shifts.shape[:2]
        sv = self._shift(
            v.unsqueeze(1).expand(b, n, *v.shape[-2:]), shifts, kernel=self.kernel
        )
        return self._shift(wt2 * sv, -shifts, kernel=self.kernel).sum(dim=1)

    def _solve_direct(
        self, shifts: Tensor, wt: Tensor, wt2: Tensor, y: Tensor,
        rcond: float | None = None,
    ) -> Tensor:
        """Exact template solve: dense normal matrix + eigen pseudo-inverse.

        The template lives on the cutout grid (``npix = H * W`` ~ 1e3), so
        the normal matrix :math:`H = \\sum_i S_i^T \\mathrm{diag}(w_i t_i^2)
        S_i` fits in memory.  Its columns come from *one* ``npix``-batched
        application of the matrix-free operator per batch member (the same
        code path as CG, so no independent discretization enters).  ``H`` is
        positive semi-definite and typically singular on the aperture
        (masked checkerboard modes near half-pixel shifts), so the solution
        is the eigen-decomposition pseudo-inverse with relative cutoff
        ``rcond``; null-space components do not affect the chi-square.
        """
        rcond = self.cg.rcond if rcond is None else rcond
        b_theta, n = shifts.shape[:2]
        h, w = y.shape[-2:]
        npix = h * w
        rhs = self._shift(wt * y, -shifts, kernel=self.kernel).sum(dim=1)
        eye = torch.eye(npix, dtype=y.dtype, device=y.device).reshape(npix, h, w)
        ghat = torch.empty(b_theta, npix, dtype=y.dtype, device=y.device)
        for i in range(b_theta):
            cols = self._normal_apply(
                eye, shifts[i : i + 1].expand(npix, n, 2), wt2[i : i + 1]
            ).reshape(npix, npix)
            # Decompose in float64 regardless of the working dtype: the
            # normal matrix routinely has condition numbers ~1e12 near
            # half-pixel shifts, beyond single precision.
            hm = 0.5 * (cols + cols.mT).double()  # symmetrized columns of H
            evals, evecs = torch.linalg.eigh(hm)
            cut = rcond * evals[-1].clamp(min=torch.finfo(torch.float64).tiny)
            inv = torch.where(evals > cut, 1.0 / evals.clamp(min=cut), evals.new_zeros(()))
            sol = evecs @ (inv * (evecs.mT @ rhs[i].reshape(-1).double()))
            ghat[i] = sol.to(ghat)
        return ghat.reshape(b_theta, h, w)

    def solve_template(
        self, shifts: Tensor, t: Tensor, y: Tensor, tight: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Profiled template solve.

        Parameters are the per-epoch total shifts ``(B, N, 2)``, transmission
        ``(B, N, H, W)`` and data ``(B or 1, N, H, W)``.  Returns
        ``(ghat, gap)`` with ``ghat (B, H, W)``.

        Two modes share the same matrix-free operator:

        * ``tight=True`` (certified): exact dense direct solve of the
          template normal equations (:meth:`_solve_direct`; ``gap`` is
          zero).  Every log-likelihood that enters a reported statistic is
          evaluated this way.
        * ``tight=False`` (search): Jacobi-preconditioned conjugate
          gradients with the loose ``cg.opt_*`` stopping settings -- cheap
          and differentiable-friendly inside optimizer steps, where only
          the search direction matters.  ``gap`` is a heuristic
          rate-corrected estimate of the remaining chi-square decrease
          (trailing-window sums ``S_cur``, ``S_prev``;
          ``gap ~ S_cur * rho / (1 - rho)`` for ``rho = S_cur/S_prev < 1``);
          CG staircase convergence means this is *not* a certificate,
          which is why reported values use the direct solve.
        """
        cfg = self.cg
        wt = self.w * t
        wt2 = wt * t
        if tight:
            ghat = self._solve_direct(shifts, wt, wt2, y)
            zero = ghat.new_zeros(ghat.shape[0])
            return ghat, zero
        tol = cfg.opt_tol_chi2
        max_iters = cfg.opt_max_iters
        b = self._shift(wt * y, -shifts, kernel=self.kernel).sum(dim=1)
        diag = self._shift(wt2, -shifts, kernel=self.kernel, squared=True).sum(dim=1)
        # Nonnegative bilinear shift-and-add initial estimate.
        c_bl = self._shift(wt2, -shifts, kernel="bilinear").sum(dim=1)
        b_bl = self._shift(wt * y, -shifts, kernel="bilinear").sum(dim=1)
        x = torch.where(c_bl > 0, b_bl / c_bl.clamp(min=torch.finfo(b.dtype).tiny), b.new_zeros(()))

        floor = 1.0e-12 * diag.amax(dim=(-2, -1), keepdim=True).clamp(min=torch.finfo(b.dtype).tiny)
        pre = torch.where(diag > 0, 1.0 / diag.clamp(min=floor), b.new_zeros(()))

        r = b - self._normal_apply(x, shifts, wt2)
        z = r * pre
        p = z.clone()
        rz = (r * z).sum(dim=(-2, -1))
        nb = x.shape[0]
        # Ring buffer of the last 2*window per-step chi-square decreases.
        hist = torch.zeros((nb, 2 * cfg.window), dtype=x.dtype, device=x.device)
        gap = torch.full((nb,), float("inf"), dtype=x.dtype, device=x.device)
        active = torch.ones(nb, dtype=torch.bool, device=x.device)
        for k in range(max_iters):
            hp = self._normal_apply(p, shifts, wt2)
            php = (p * hp).sum(dim=(-2, -1))
            alpha = torch.where(
                active & (php > 0), rz / php.clamp(min=torch.finfo(b.dtype).tiny), rz.new_zeros(())
            )
            x = x + alpha.view(-1, 1, 1) * p
            r = r - alpha.view(-1, 1, 1) * hp
            z = r * pre
            rz_new = (r * z).sum(dim=(-2, -1))
            dchi2 = alpha * rz  # chi-square decrease of this step
            hist[:, k % (2 * cfg.window)] = torch.where(active, dchi2, dchi2.new_zeros(()))
            if k + 1 >= max(cfg.min_iters, 2 * cfg.window):
                idx = torch.arange(2 * cfg.window, device=x.device)
                cur = (k - idx) % (2 * cfg.window) < cfg.window  # newest `window` slots
                s_cur = hist[:, cur].sum(dim=1)
                s_prev = hist[:, ~cur].sum(dim=1)
                rho = s_cur / s_prev.clamp(min=torch.finfo(b.dtype).tiny)
                gap_est = torch.where(
                    (s_prev > 0) & (rho < 1.0),
                    s_cur * rho / (1.0 - rho).clamp(min=torch.finfo(b.dtype).tiny),
                    torch.full_like(s_cur, float("inf")),
                )
                converged = (gap_est <= tol) | (s_cur <= 1.0e-3 * tol)
                gap_record = torch.where(torch.isfinite(gap_est), gap_est, s_cur)
                gap = torch.where(active & converged, gap_record, gap)
                active = active & ~converged
                if not bool(active.any()):
                    break
            beta = torch.where(
                active & (rz > 0), rz_new / rz.clamp(min=torch.finfo(b.dtype).tiny), rz.new_zeros(())
            )
            p = z + beta.view(-1, 1, 1) * p
            rz = rz_new
        if bool(active.any()):
            # Ran out of iterations: report the last gap estimate (may be inf).
            idx = torch.arange(2 * cfg.window, device=x.device)
            cur = ((max_iters - 1) - idx) % (2 * cfg.window) < cfg.window
            s_cur = hist[:, cur].sum(dim=1)
            s_prev = hist[:, ~cur].sum(dim=1)
            rho = s_cur / s_prev.clamp(min=torch.finfo(b.dtype).tiny)
            gap_tail = torch.where(
                (s_prev > 0) & (rho < 1.0),
                s_cur * rho / (1.0 - rho).clamp(min=torch.finfo(b.dtype).tiny),
                torch.full_like(s_cur, float("inf")),
            )
            gap = torch.where(active, gap_tail, gap)
        return x, gap

    # -------------------------------------------------------------- forward --
    def forward(self, theta: Tensor, tight: bool = False) -> Tensor:
        """Profiled log-likelihood for ``theta`` of shape ``(B, P)``.

        The template solve runs detached (envelope theorem); the returned
        statistic is the explicit-residual chi-square -- a true lower bound
        of the exact profile log-likelihood -- and differentiable with
        respect to ``theta``.  With ``tight=True`` the template comes from
        the exact dense direct solve; the loose CG settings apply
        otherwise.  Every value that enters a reported statistic must be
        evaluated with ``tight=True``.
        """
        if theta.ndim != 2 or theta.shape[1] != self.n_params:
            raise ValueError(
                f"expected (B, {self.n_params}) parameters, got {tuple(theta.shape)}"
            )
        b = theta.shape[0]
        if self.n_realizations > 1 and b != self.n_realizations:
            raise ValueError(
                f"batched data with R={self.n_realizations} requires theta batch "
                f"B == R, got B={b}"
            )
        delta, taus = self._split(theta)
        t = self.transmission(taus, b)
        shifts = self.nominal.unsqueeze(0) + delta
        with torch.no_grad():
            ghat, _ = self.solve_template(shifts.detach(), t.detach(), self.y, tight=tight)
        g_ep = self._shift(
            ghat.unsqueeze(1).expand(b, self.n_epochs, *ghat.shape[-2:]),
            shifts,
            kernel=self.kernel,
        )
        residual = self.y - t * g_ep
        chi2 = (self.w * residual.square()).sum(dim=(-3, -2, -1))
        logl = -0.5 * chi2
        floor = torch.finfo(logl.dtype).min / 4.0
        return torch.where(torch.isfinite(logl), logl, torch.full_like(logl, floor))

    # --------------------------------------------------------- exact checks --
    @torch.inference_mode()
    def profile_dense(self, theta: Tensor, jitter: float = 0.0) -> tuple[Tensor, Tensor]:
        """Reference dense solve of the normal equations (verification only).

        Builds :math:`H` column-by-column via the matrix-free operator and
        solves with a direct factorization on the constrained subspace
        (pixels with nonzero exact diagonal).  Returns ``(ghat, logl)``.
        Cost scales with the pixel count; intended for a handful of
        parameter points in ``diskphot verify``.
        """
        theta = torch.atleast_2d(theta)
        b = theta.shape[0]
        delta, taus = self._split(theta)
        t = self.transmission(taus, b)
        shifts = self.nominal.unsqueeze(0) + delta
        wt = self.w * t
        wt2 = wt * t
        h, w = self.y.shape[-2:]
        npix = h * w
        rhs = self._shift(wt * self.y, -shifts, kernel=self.kernel).sum(dim=1)
        diag = self._shift(wt2, -shifts, kernel=self.kernel, squared=True).sum(dim=1)
        eye = torch.eye(npix, dtype=self.y.dtype, device=self.y.device).reshape(npix, h, w)
        cols = []
        for k in range(npix):
            cols.append(self._normal_apply(eye[k].expand(b, h, w), shifts, wt2).reshape(b, npix))
        hm = torch.stack(cols, dim=-1)  # (B, npix, npix), columns of H
        ghat = torch.zeros(b, npix, dtype=self.y.dtype, device=self.y.device)
        for i in range(b):
            act = (diag[i].reshape(-1) > 1.0e-12 * float(diag[i].max())).nonzero(as_tuple=True)[0]
            sub = hm[i][act][:, act]
            sub = 0.5 * (sub + sub.mT)  # symmetrize roundoff
            if jitter:
                sub = sub + jitter * torch.eye(len(act), dtype=sub.dtype, device=sub.device)
            # H is positive semi-definite and typically singular on the
            # aperture (masked checkerboard modes near half-pixel shifts),
            # so use the minimum-norm least-squares solution.  Null-space
            # components of ghat do not change the chi-square.
            sol = torch.linalg.lstsq(
                sub.cpu(), rhs[i].reshape(-1)[act].cpu().unsqueeze(-1), driver="gelsd"
            ).solution.squeeze(-1)
            ghat[i, act] = sol.to(ghat)
        ghat = ghat.reshape(b, h, w)
        g_ep = self._shift(
            ghat.unsqueeze(1).expand(b, self.n_epochs, h, w), shifts, kernel=self.kernel
        )
        resid = self.y - t * g_ep
        logl = -0.5 * (self.w * resid.square()).sum(dim=(-3, -2, -1))
        return ghat, logl

    # ------------------------------------------------------------ diagnostics
    @torch.inference_mode()
    def maps(self, theta: Tensor, realization: int = 0) -> dict[str, Tensor]:
        """Diagnostic maps for a single parameter vector.

        Returns (native epoch frames unless noted):

        ``template``
            Exactly profiled galaxy image :math:`\\hat G` (epoch-0 frame).
        ``model_epoch``
            Epoch-frame galaxy forward model :math:`T_i\\,[S_{s_i}\\hat G]`
            (add ``data.disk`` for the full forward model of the raw images).
        ``residual_epoch``
            ``y_i - model_epoch`` -- the quantity whose weighted square is
            the likelihood chi-square.
        ``chi_epoch``
            Signed per-pixel deviation ``residual_epoch * sqrt(w_i)``
            (units of sigma; zero where a pixel carries no weight).
        ``aligned``
            Display-only de-extincted stack aligned to the template frame
            (nonnegative bilinear weights; not used by the likelihood).
        ``weight``
            The fixed likelihood weights ``w_i``.
        ``mask`` / ``valid``
            Analysis aperture and validity per epoch.
        """
        theta = torch.atleast_2d(theta)
        if self.n_realizations > 1:
            theta = theta[realization : realization + 1]
            y = self.y[realization : realization + 1]
        else:
            theta = theta[:1]
            y = self.y
        delta, taus = self._split(theta)
        t = self.transmission(taus, 1)
        shifts = self.nominal.unsqueeze(0) + delta
        ghat, _ = self.solve_template(shifts, t, y, tight=True)
        g_ep = self._shift(
            ghat.unsqueeze(1).expand(1, self.n_epochs, *ghat.shape[-2:]),
            shifts,
            kernel=self.kernel,
        )
        model_epoch = (t * g_ep)[0]
        residual_epoch = y[0] - model_epoch
        chi_epoch = residual_epoch * torch.sqrt(self.w)

        wt = self.w * t
        num = self._shift(wt * y, -shifts, kernel="bilinear").sum(dim=1)[0]
        den = self._shift(wt * t, -shifts, kernel="bilinear").sum(dim=1)[0]
        aligned_num = self._shift(wt * y, -shifts, kernel="bilinear")[0]
        aligned_den = self._shift(wt * t, -shifts, kernel="bilinear")[0]
        tiny = torch.finfo(den.dtype).tiny
        return {
            "template": ghat[0],
            "template_stack": num / den.clamp(min=tiny) * (den > 0),
            "aligned": aligned_num / aligned_den.clamp(min=tiny) * (aligned_den > 0),
            "model_epoch": model_epoch,
            "residual_epoch": residual_epoch,
            "chi_epoch": chi_epoch,
            "weight": self.w,
            "mask": self.mask,
            "valid": self.valid,
        }
