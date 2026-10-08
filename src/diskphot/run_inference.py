"""Nested-sampling driver of the paper: audit, UltraNest run, single-run
prior sensitivity and the run's MODEL_CARD.

Run from the repository root with the data in place (``DISKPHOT_DATA``,
see ``diskphot.likelihood`` and README.md)::

    python -m diskphot.run_inference [--log-dir DIR] [--verify-only] [...]

The defaults are the settings of the production run of the paper
(``ultranest_run/marginal_common_tau_noisescale_v16``).  Steps:

1. load the native-grid data and build the marginal ``LogLikelihood``
   (``noise_scale=True``: v07 layout, theta = offsets, ln f_i, tau');
2. run ``LogLikelihood.verify()`` on the data and write ``verify.json`` --
   check ``dlogdet_f32_vs_ref`` / ``dlnl_marg_f32_vs_ref`` against the
   dlogz budget before believing the evidence error bar;
3. print the common-grid consistency demo (correctly weighted pairwise
   statistic == lam -> 0 profiled chi-square at zero shifts);
4. sample the posterior with UltraNest's population slice sampler
   (vectorized f32 hot path; reported values are re-evaluated in f64 by
   ``paper_values --evaluate``);
5. importance-reweight the stored posterior for the template-prior ladder
   and the soft-log tau' prior (``sensitivity.json``), write
   ``MODEL_CARD.txt``.

Priors (also written to MODEL_CARD.txt): offsets uniform on
+/- OFFSET_HALFWIDTH px, ln f_i uniform on ln NOISE_SCALE_RANGE, tau' flat
on [0, TAU_HI].  Never ``resume`` a run directory produced by an older
(profiled, v05) likelihood: the integrand changed.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from diskphot.likelihood import (
    NOISE_SCALE_RANGE,
    LogLikelihood,
    load_native_data,
    model_card_text,
    param_names,
    single_run_sensitivity,
)

DEFAULT_LOG_DIR = "ultranest_run/marginal_common_tau_noisescale_v16"
OFFSET_HALFWIDTH = 1.5          # px box around the nominal registration
TAU_HI = 1.0                    # conservative ceiling; tau = 1 is already rejectable
TAU_SOFT = 1.0e-2               # scale of the ALTERNATIVE soft-log prior
                                # (post-processing reweighting only)


def build_likelihood(verbose=True):
    """Data + the production-run likelihood (v07: per-epoch noise scales)."""
    observations, disk_models, masks, variance, nominal_shift_xy, q_cut = \
        load_native_data()
    ll = LogLikelihood(observations, disk_models, masks, variance,
                       nominal_shift_xy, noise_scale=True)
    names = param_names(observations.shape[0], noise_scale=ll.noise_scale)
    assert len(names) == ll.ndim
    if verbose:
        print(f"template prior: g ~ N(0, lam^-1 I), lam = {ll.lam:.6e}"
              f"  (sigma_g = {ll.template_prior_std:.4e} cts/s/px;"
              "  empirical-Bayes scale from the NOMINAL weights, frozen at "
              "construction)")
        print(f"lnL constants: c_data = {ll._c_data:.6e},"
              f"  P/2 ln lam = {0.5 * ll.size**2 * math.log(ll.lam):.6e}")
        print("weighted pixels per epoch n_i (coefficients of -ln f_i):",
              ll.n_weighted.cpu().numpy().astype(int).tolist())
    return ll, names, (observations, disk_models, masks, variance,
                       nominal_shift_xy, q_cut)


def consistency_demo(ll, data):
    """Old-vs-new consistency (common grid): the correctly weighted pairwise
    statistic equals the profiled chi-square (lam -> 0) -- the same number
    from two derivations.  (Profile objects; the marginal adds the volume
    term.)"""
    observations, disk_models, masks, variance, _, _ = data
    truths = torch.zeros(ll.ndim)          # offsets 0, ln f = 0 (f = 1)
    truths[-1] = 1.0e-2                    # tau' at the ring peak
    tile = truths.view(1, -1).tile(2, 1)
    print("marginal lnL (f32 hot path):", ll(tile))
    print("marginal lnL (f64 reference):", ll.reference_logl(tile))
    truths_f2 = truths.clone()
    truths_f2[ll.i_lnf] = math.log(2.0)
    print("marginal lnL (f64 reference) at f = 2 for every epoch:",
          ll.reference_logl(truths_f2.view(1, -1)))
    ll0 = LogLikelihood(observations, disk_models, masks, variance,
                        torch.zeros(observations.shape[0], 2))
    pair = float(ll0.pairwise_chi2(1.0e-2))
    prof = float(ll0.profiled_chi2(truths.view(1, -1),
                                   lam=1.0e-12 * ll0._maxdiag_ref)[0])
    print(f"pairwise (correct weights): {pair:.6f}")
    print(f"profiled template (lam->0): {prof:.6f}   "
          f"(relative difference {abs(pair - prof) / abs(pair):.2e})")


def run_sampler(ll, names, log_dir, args):
    import ultranest
    from ultranest.popstepsampler import (
        PopulationSimpleSliceSampler,
        generate_mixture_random_direction,
    )

    ndim = ll.ndim                  # 2 (N-1) offsets + N ln f + tau'
    n_off = ll.n_offsets
    i_lnf = ll.i_lnf
    lnf_lo, lnf_hi = (math.log(NOISE_SCALE_RANGE[0]),
                      math.log(NOISE_SCALE_RANGE[1]))

    def prior_transform(cube):
        params = cube.copy()
        params[:, :n_off] = (2.0 * cube[:, :n_off] - 1.0) * OFFSET_HALFWIDTH
        # per-epoch noise scales, ln f_i uniform on [ln lo, ln hi]
        # (log-uniform f_i: the Jeffreys prior for a scale).  The ln f block
        # is well constrained (~1/sqrt(2 n_i) ~ 0.05 wide), so the bounds
        # only need to contain the answer; the evidence scales with the
        # inverse of the prior width per epoch.
        if ll.noise_scale:
            params[:, i_lnf] = lnf_lo + cube[:, i_lnf] * (lnf_hi - lnf_lo)
        # FLAT prior on tau' in [0, TAU_HI]: the simplest proper choice and
        # the conservative one for an upper limit.  The soft-log prior
        # (density 1/[(tau'+TAU_SOFT) ln(1+TAU_HI/TAU_SOFT)]) is recovered
        # from the SAME run by prior-ratio reweighting below.  Under a flat
        # prior the EVIDENCE scales with 1/TAU_HI: state the ceiling
        # wherever lnZ is quoted.
        params[:, ndim - 1] = cube[:, ndim - 1] * TAU_HI
        return params

    def wrapper_log_likelihood(theta):
        theta = torch.as_tensor(np.asarray(theta, dtype=np.float32))
        with torch.inference_mode():
            out = ll(theta)
        out = out.detach().cpu().numpy().astype(np.float64)
        return np.where(np.isfinite(out), out, -1.0e30)

    sampler = ultranest.ReactiveNestedSampler(
        names, wrapper_log_likelihood, prior_transform,
        vectorized=True,
        log_dir=str(log_dir),
        resume="resume",    # never resume a v05/v06 run directory
        ndraw_min=128,
        ndraw_max=1024,
    )
    sampler.stepsampler = PopulationSimpleSliceSampler(
        popsize=args.popsize,
        nsteps=args.nsteps,
        generate_direction=generate_mixture_random_direction,
    )
    sampler.run(
        min_num_live_points=args.min_live_points,
        dlogz=args.dlogz,   # posterior-grade; tighten (e.g. 0.1) if the
                            # EVIDENCE itself is a headline number
        min_ess=args.min_ess,
        update_interval_volume_fraction=0.4,
        max_num_improvement_loops=args.max_improvement_loops,
    )
    sampler.print_results()
    sampler.plot()
    return sampler


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", default=DEFAULT_LOG_DIR,
                    help=f"UltraNest log directory (default {DEFAULT_LOG_DIR})")
    ap.add_argument("--verify-only", action="store_true",
                    help="run the audit and the consistency demo only, "
                         "write verify.json, do not sample")
    ap.add_argument("--min-live-points", type=int, default=521)
    ap.add_argument("--dlogz", type=float, default=0.5)
    ap.add_argument("--min-ess", type=int, default=512)
    ap.add_argument("--max-improvement-loops", type=int, default=4)
    ap.add_argument("--popsize", type=int, default=128,
                    help="population size of the slice sampler (GPU batch)")
    ap.add_argument("--nsteps", type=int, default=32,
                    help="slice-sampling steps per population draw")
    args = ap.parse_args(argv)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    ll, names, data = build_likelihood()
    n_images = data[0].shape[0]

    # the audit -- run BEFORE trusting fast-path numbers on this data.  In
    # particular check dlogdet_f32_vs_ref / dlnl_marg_f32_vs_ref against
    # the dlogz budget; dlnl_noise_scale_identity must be at roundoff.
    audit = ll.verify()
    (log_dir / "verify.json").write_text(json.dumps(
        {k: v for k, v in audit.items() if isinstance(v, float)}, indent=2))
    consistency_demo(ll, data)
    if args.verify_only:
        print(f"verify.json written to {log_dir}; sampling skipped")
        return 0

    sampler = run_sampler(ll, names, log_dir, args)

    # single-run sensitivity: lam ladder (likelihood reweighting, f64
    # reference at the stored points) and the soft-log tau' prior
    # (prior-ratio reweighting) -- no additional sampling.  Check the ESS
    # column before quoting any variant.
    ws = sampler.results["weighted_samples"]
    single_run_sensitivity(
        ll, np.asarray(ws["points"]), np.asarray(ws["weights"]),
        stored_logl=np.asarray(ws["logl"]) if "logl" in ws else None,
        tau_soft=TAU_SOFT, tau_hi=TAU_HI,
        out_path=log_dir / "sensitivity.json",
    )
    (log_dir / "MODEL_CARD.txt").write_text(
        model_card_text(ll, n_images, OFFSET_HALFWIDTH, TAU_HI, TAU_SOFT))
    print("note: this run maps the MARGINAL likelihood -- samples are a "
          "posterior and logz is an evidence under the priors of "
          "MODEL_CARD.txt; reported values come from reference_logl (f64) "
          "via paper_values --evaluate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
