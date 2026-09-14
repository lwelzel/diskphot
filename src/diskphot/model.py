"""Joint profiled likelihood of all epochs.

The disk-subtracted epochs are modelled as one free galaxy template,
shifted to each epoch (torch grid sampler) and attenuated by the trial
transmission.  For fixed shifts and extinction the template is linear, so
it is profiled *exactly*: the dense normal matrix ``H`` (the cutout has
~1e3 pixels) is assembled from the same operator as the forward model and
solved by batched, ridge-regularized Cholesky in float64.  One solver is
used everywhere -- inside optimizer steps and for every reported value --
so the estimator is a single, deterministic function of the data.

The returned statistic is the explicit residual chi-square at the solved
template; with the template detached, autograd yields the envelope-theorem
gradient of the profiled objective with respect to the offsets and the
extinction parameters.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import SolveConfig, Variant
from .data import EpochData
from .transforms import shift


class JointLogLikelihood(nn.Module):
    """Profiled log-likelihood ``lnL_p(theta)`` for one extinction variant.

    ``theta (B, P)`` stacks the free offsets ``(dx_i, dy_i)`` for epochs
    ``1..N-1`` (epoch 0 gauge-fixed) followed by the extinction parameters
    of the variant.  ``observations`` (optional, ``(R, N, H, W)``) replaces
    the observed disk-subtracted cutouts by simulated ones; batched
    evaluation then requires ``B == R`` (or ``R == 1``).
    """

    def __init__(
        self,
        data: EpochData,
        variant: Variant,
        solve: SolveConfig | None = None,
        observations: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.solve_cfg = solve or SolveConfig()
        y = data.galaxy_only.unsqueeze(0) if observations is None else observations
        self.register_buffer("y", y)                     # (R, N, H, W)
        self.register_buffer("w", data.weights)          # (N, H, W)
        self.register_buffer("dhat", data.disk_hat)      # (N, H, W)
        self.register_buffer("nominal", data.nominal_shift_xy)  # (N, 2)
        self.n_epochs = int(self.w.shape[0])
        self.shape = tuple(self.w.shape[-2:])

    # ------------------------------------------------------------ bookkeeping
    @property
    def n_offset_params(self) -> int:
        return 2 * (self.n_epochs - 1)

    @property
    def n_tau_params(self) -> int:
        return {
            Variant.NULL: 0,
            Variant.COMMON: 1,
            Variant.PER_EPOCH: self.n_epochs,
            Variant.GREY: self.n_epochs - 1,
        }[self.variant]

    @property
    def n_params(self) -> int:
        return self.n_offset_params + self.n_tau_params

    @property
    def param_names(self) -> list[str]:
        names = [f"d{a}_{i}" for i in range(1, self.n_epochs) for a in ("x", "y")]
        if self.variant is Variant.COMMON:
            names += ["tau"]
        elif self.variant is Variant.PER_EPOCH:
            names += [f"tau_{i}" for i in range(self.n_epochs)]
        elif self.variant is Variant.GREY:
            names += [f"gamma_{i}" for i in range(1, self.n_epochs)]
        return names

    @property
    def n_weighted_pixels(self) -> int:
        return int((self.w > 0).sum())

    def _split(self, theta: Tensor) -> tuple[Tensor, Tensor]:
        """``theta (B, P)`` -> offsets ``(B, N, 2)`` (row 0 zero), tau block."""
        b = theta.shape[0]
        delta = theta.new_zeros(b, self.n_epochs, 2)
        delta[:, 1:] = theta[:, : self.n_offset_params].reshape(b, -1, 2)
        return delta, theta[:, self.n_offset_params :]

    def transmission(self, tpar: Tensor) -> Tensor:
        """Per-epoch transmission field, broadcastable to ``(B, N, H, W)``."""
        b = tpar.shape[0] if tpar.ndim else 1
        if self.variant is Variant.NULL:
            return self.w.new_ones(b, 1, 1, 1)
        if self.variant is Variant.COMMON:
            return torch.exp(-tpar.view(-1, 1, 1, 1) * self.dhat)
        if self.variant is Variant.PER_EPOCH:
            return torch.exp(-tpar.view(b, self.n_epochs, 1, 1) * self.dhat)
        gamma = torch.cat([tpar.new_zeros(b, 1), tpar], dim=1)  # gamma_0 = 0
        return torch.exp(-gamma).view(b, self.n_epochs, 1, 1)

    # -------------------------------------------------------------- solver --
    def solve_template(self, shifts: Tensor, t: Tensor, y: Tensor | None = None) -> Tensor:
        """Exact profiled template for total shifts ``(B, N, 2)``.

        Assembles the dense normal matrix ``H = sum_i S_i^T diag(w t^2) S_i``
        column-by-column with the grid-sampler operator (adjoint = negated
        shift), adds the relative ridge, and solves by batched Cholesky in
        float64.  Returns ``ghat (B, H, W)`` in the working dtype.
        """
        y = self.y if y is None else y
        bsz, n = shifts.shape[:2]
        hh, ww = self.shape
        npix = hh * ww
        wt = self.w * t
        wt2 = wt * t                                   # (B, N, H, W)
        yb = y if y.shape[0] == bsz else y.expand(bsz, -1, -1, -1)
        rhs = shift((wt * yb).reshape(-1, hh, ww), -shifts.reshape(-1, 2))
        rhs = rhs.reshape(bsz, n, hh, ww).sum(dim=1).reshape(bsz, npix)

        eye = torch.eye(npix, dtype=wt.dtype, device=wt.device).reshape(npix, hh, ww)
        idn = torch.eye(npix, dtype=torch.float64, device=wt.device)
        ghat = torch.empty(bsz, npix, dtype=torch.float64, device=wt.device)
        cs = max(int(self.solve_cfg.chunk), 1)
        for i0 in range(0, bsz, cs):
            sl = slice(i0, min(i0 + cs, bsz))
            nm = sl.stop - sl.start
            acc = wt.new_zeros(nm, npix, npix)
            for e in range(n):
                se = shifts[sl, e].unsqueeze(1).expand(nm, npix, 2).reshape(-1, 2)
                cols = shift(eye.unsqueeze(0).expand(nm, npix, hh, ww).reshape(-1, hh, ww), se)
                w2 = wt2[sl, e] if wt2.shape[0] == bsz else wt2[:, e].expand(nm, hh, ww)
                cols = cols.reshape(nm, npix, hh, ww) * w2.reshape(nm, 1, hh, ww)
                back = shift(cols.reshape(-1, hh, ww), -se)
                acc += back.reshape(nm, npix, npix)
            h64 = (0.5 * (acc + acc.mT)).double()      # symmetrize roundoff
            # Ridge + Cholesky in double precision (H is near-singular;
            # condition numbers ~1e12 exceed float32).
            lam = self.solve_cfg.ridge * h64.diagonal(dim1=-2, dim2=-1).amax(dim=-1)
            lam = lam.clamp_min(torch.finfo(torch.float64).tiny)
            chol, info = torch.linalg.cholesky_ex(h64 + lam.view(-1, 1, 1) * idn)
            if int(info.max()) > 0:                    # bump ridge where needed
                bad = info > 0
                h64[bad] = h64[bad] + (100.0 * lam[bad]).view(-1, 1, 1) * idn
                chol, info = torch.linalg.cholesky_ex(h64 + lam.view(-1, 1, 1) * idn)
            ghat[sl] = torch.cholesky_solve(
                rhs[sl].double().unsqueeze(-1), chol
            ).squeeze(-1)
        return ghat.to(wt.dtype).reshape(bsz, hh, ww)

    def template_for(self, theta: Tensor) -> Tensor:
        """Fresh (detached) template at ``theta`` -- the block-descent anchor."""
        with torch.no_grad():
            delta, tpar = self._split(theta)
            t = self.transmission(tpar)
            return self.solve_template(self.nominal + delta, t)

    # ------------------------------------------------------------- forward --
    def forward(self, theta: Tensor, template: Tensor | None = None) -> Tensor:
        """Profiled log-likelihood ``(B,)`` (explicit residual chi-square).

        With ``template=None`` the template is solved fresh (detached ->
        exact envelope gradient).  Passing a previously solved template
        evaluates the joint objective at that fixed template -- a rigorous
        lower bound of the profile, used between re-solves inside the
        optimizer.
        """
        delta, tpar = self._split(theta)
        shifts = self.nominal + delta                  # (B, N, 2)
        t = self.transmission(tpar)
        if template is None:
            with torch.no_grad():
                template = self.solve_template(shifts.detach(), t.detach())
        bsz = shifts.shape[0]
        hh, ww = self.shape
        g = template.unsqueeze(1).expand(bsz, self.n_epochs, hh, ww)
        model = t * shift(g.reshape(-1, hh, ww), shifts.reshape(-1, 2)).reshape(
            bsz, self.n_epochs, hh, ww
        )
        y = self.y if self.y.shape[0] == bsz else self.y.expand(bsz, -1, -1, -1)
        chi2 = (self.w * (y - model).square()).sum(dim=(-3, -2, -1))
        return -0.5 * chi2

    # ---------------------------------------------------------- diagnostics --
    def maps(self, theta: Tensor) -> dict[str, Tensor]:
        """Template, per-epoch model, residual and standardized residual."""
        theta = theta.reshape(1, -1)
        with torch.no_grad():
            delta, tpar = self._split(theta)
            shifts = self.nominal + delta
            t = self.transmission(tpar)
            ghat = self.solve_template(shifts, t)
            hh, ww = self.shape
            g = ghat.expand(self.n_epochs, hh, ww)
            model = (t * shift(g, shifts[0]).unsqueeze(0))[0]
            resid = self.y[0] - model
            chi = resid * self.w.sqrt()
        return {
            "template": ghat[0],
            "model_epoch": model,
            "residual_epoch": resid,
            "chi_epoch": chi,
            "weight": self.w,
        }
