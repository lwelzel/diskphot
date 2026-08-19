"""Command-line interface.

Typical campaign
----------------
Observed fits (the common-tau' science fit, the null model, and the
appendix variants)::

    diskphot fit --variant common
    diskphot fit --variant per-epoch
    diskphot fit --variant grey
    diskphot fit --variant common --mask-pixel 2,12,12 --run-name common_ep2masked

Simulation calibration of the statistics (detection threshold, power,
Neyman critical curves; the expensive step -- run on a GPU)::

    diskphot calibrate --run outputs/runs/common

Assemble the calibrated limits, p-value, epoch table and figures::

    diskphot limits --run outputs/runs/common

Numerical verification, stress reruns and diagnostics::

    diskphot verify
    diskphot roundtrip --tau 1e-3 --no-noise
    diskphot calibrate --run outputs/runs/common --noise-model correlated \\
        --out-name calibration_correlated
    diskphot explore            # optional objective map (diagnostic only)
    diskphot inspect-data
    diskphot inspect-shifts --run outputs/runs/common
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from .config import CalibConfig, DataConfig, FitConfig, ProfileConfig, Variant

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Argument plumbing
# --------------------------------------------------------------------------- #
def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--device", default="auto", help="auto | cuda | cuda:N | cpu")
    parser.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--variant",
        default="common",
        choices=[v.value for v in Variant],
        help="extinction model variant",
    )
    parser.add_argument(
        "--mask-pixel",
        action="append",
        default=[],
        metavar="EP,X,Y",
        help="exclude a cutout pixel from an epoch (repeatable); "
        "use for the EP2 bulge-pixel robustness rerun",
    )
    parser.add_argument("--sigma-bg", type=float, default=None,
                        help="fixed background RMS (default: measured per epoch)")
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument(
        "--variance-source", default="model",
        choices=("model", "plugin", "background"),
        help="rate term of the per-pixel variance (model-predicted by default)",
    )
    parser.add_argument("--offset-halfwidth", type=float, default=1.5)
    parser.add_argument("--tau-hi", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=None,
                        help="Adam steps of every profile optimization")


def _cfg_from_args(args: argparse.Namespace) -> FitConfig:
    mask_pixels = tuple(
        tuple(int(v) for v in spec.split(",")) for spec in args.mask_pixel
    )
    for spec in mask_pixels:
        if len(spec) != 3:
            raise SystemExit(f"--mask-pixel expects EP,X,Y, got {spec}")
    profile = ProfileConfig(
        offset_halfwidth_px=args.offset_halfwidth,
        tau_hi=args.tau_hi,
    )
    if args.steps is not None:
        profile.steps = args.steps
    calib = CalibConfig()
    if getattr(args, "repeats", None) is not None:
        calib.n_repeats = args.repeats
    if getattr(args, "n_null", None) is not None:
        calib.n_null = args.n_null
    if getattr(args, "noise_model", None) is not None:
        calib.noise_model = args.noise_model
    if getattr(args, "half_pixel_offsets", False):
        calib.half_pixel_offsets = True
    if getattr(args, "taus", None):
        calib.taus = tuple(float(v) for v in args.taus)
    return FitConfig(
        variant=Variant(args.variant),
        data=DataConfig(
            data_dir=args.data_dir,
            mask_pixels=mask_pixels,
            sigma_bg=args.sigma_bg,
            gain=args.gain,
            variance_source=args.variance_source,
        ),
        profile=profile,
        calib=calib,
        log_dir=args.log_dir,
        run_name=args.run_name,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
    )


def _cfg_from_run(run: Path) -> FitConfig:
    cfg_path = Path(run) / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"no config.json in {run}; run 'diskphot fit' first")
    return FitConfig.from_json(cfg_path)


# --------------------------------------------------------------------------- #
# Subcommand bodies
# --------------------------------------------------------------------------- #
def _cmd_fit(args: argparse.Namespace) -> None:
    from .inference import run_fit

    cfg = _cfg_from_args(args)
    payload = run_fit(cfg)
    log.info("fit written to %s", cfg.run_dir)
    if "q0" in payload:
        print(f"lnL_p(null) = {payload['logl_null']:.3f}  "
              f"lnL_p(mle) = {payload['logl_mle']:.3f}  q0 = {payload['q0']:.3f}")


def _cmd_calibrate(args: argparse.Namespace) -> None:
    from .calibrate import run_calibration
    from .inference import prepare_data

    cfg = _cfg_from_run(args.run)
    if args.noise_model is not None:
        cfg.calib.noise_model = args.noise_model
    if args.repeats is not None:
        cfg.calib.n_repeats = args.repeats
    if args.n_null is not None:
        cfg.calib.n_null = args.n_null
    if args.half_pixel_offsets:
        cfg.calib.half_pixel_offsets = True
    if args.taus:
        cfg.calib.taus = tuple(float(v) for v in args.taus)
    out_dir = Path(args.run) / (args.out_name or "calibration")
    data = prepare_data(cfg)
    run_calibration(cfg, data, out_dir=out_dir)
    log.info("calibration written to %s", out_dir)


def _cmd_limits(args: argparse.Namespace) -> None:
    from astropy.table import Table

    from .calibrate import analyze_calibration
    from .inference import prepare_data
    from .plotting import detection_efficiency_plot, profile_curve_plot, residual_panels
    from .results import epoch_table, load_fit, summary_lines

    run = Path(args.run)
    cfg = _cfg_from_run(run)
    fit = load_fit(run)
    calib_dir = run / (args.calibration or "calibration")
    trials = Table.read(calib_dir / "trials.ecsv", format="ascii.ecsv")
    nulls = Table.read(calib_dir / "nulls.ecsv", format="ascii.ecsv")
    summary = analyze_calibration(
        trials, nulls, cfg, observed=fit.as_payload(),
        out_path=run / "limits_summary.json",
    )
    for line in summary_lines(summary):
        print(line)

    data = prepare_data(cfg)
    epoch_table(summary, data, out_path=run / "epoch_table.ecsv")
    profile_curve_plot(fit, summary, run / "profile_curve.pdf")
    detection_efficiency_plot(summary, run / "detection_efficiency.pdf")
    residual_panels(fit, data, run / "residual_panels.png")
    log.info("limits, epoch table and figures written to %s", run)


def _cmd_explore(args: argparse.Namespace) -> None:
    from .inference import explore

    cfg = _cfg_from_args(args)
    explore(cfg)
    print("objective map written (diagnostic only; see NOT_AN_EVIDENCE.txt)")


def _cmd_verify(args: argparse.Namespace) -> None:
    from .diagnostics import run_verify

    cfg = _cfg_from_args(args)
    data = None
    if args.on_data:
        from .inference import prepare_data

        data = prepare_data(cfg, dtype=None)
    out = Path(args.out or "outputs/dev/verify.json")
    report = run_verify(cfg, data=data, out_path=out, seed=args.seed)
    status = "PASS" if report["passed"] else "FAIL"
    print(f"verify: {status} ({out})")
    if not report["passed"]:
        raise SystemExit(1)


def _cmd_roundtrip(args: argparse.Namespace) -> None:
    from .diagnostics import run_roundtrip
    from .inference import prepare_data

    cfg = _cfg_from_args(args)
    if args.synthetic:
        import torch

        from .diagnostics import make_synthetic_epoch_data
        from .inference import resolve_device, resolve_dtype

        data = make_synthetic_epoch_data(seed=args.seed).to(
            resolve_device(cfg.device), resolve_dtype(cfg.dtype)
        )
    else:
        data = prepare_data(cfg)
    summary = run_roundtrip(
        cfg, data, tau=args.tau, add_noise=not args.no_noise,
        seed=args.seed, out_dir=args.out_dir,
    )
    print(json.dumps(summary, indent=2))


def _cmd_inspect_data(args: argparse.Namespace) -> None:
    import torch

    from .diagnostics import plot_field_overview, plot_forward_grid, theta_initial
    from .inference import prepare_data
    from .model import JointLogLikelihood

    cfg = _cfg_from_args(args)
    data = prepare_data(cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_field_overview(cfg.data, data, out_dir / "field_overview.png")
    model = JointLogLikelihood(data, Variant.NULL, cfg.cg)
    plot_forward_grid(
        data, model, theta_initial(model), out_dir / "forward_initial.png",
        title="initial guess (zero offsets, tau = 0)",
    )
    print(f"inspection figures written to {out_dir}")


def _cmd_inspect_shifts(args: argparse.Namespace) -> None:
    import torch

    from .diagnostics import nuisance_spectrum, plot_forward_grid, plot_shift_scan
    from .inference import prepare_data
    from .model import JointLogLikelihood
    from .results import load_fit

    run = Path(args.run)
    cfg = _cfg_from_run(run)
    fit = load_fit(run)
    data = prepare_data(cfg)
    model = JointLogLikelihood(data, fit.variant, cfg.cg)
    theta_np = fit.theta_mle if fit.theta_mle is not None else fit.theta_null
    theta = torch.as_tensor(theta_np, dtype=data.mask.dtype)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_shift_scan(model, theta, out_dir / "shift_scan.png",
                    halfwidth_px=cfg.profile.offset_halfwidth_px)
    plot_forward_grid(data, model, theta, out_dir / "forward_mle.png",
                      title="maximum-likelihood point")
    nuisance_spectrum(model, theta, out_path=out_dir / "nuisance_spectrum.json")
    print(f"shift diagnostics written to {out_dir}")


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="diskphot", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_fit = sub.add_parser("fit", help="observed profile-likelihood fit of one variant")
    _add_common(p_fit)
    p_fit.set_defaults(func=_cmd_fit)

    p_cal = sub.add_parser(
        "calibrate", help="injection--recovery calibration of a finished fit"
    )
    p_cal.add_argument("--run", type=Path, required=True,
                       help="run directory of the observed fit (config.json)")
    p_cal.add_argument("--taus", type=float, nargs="+", default=None,
                       help="injected tau' grid (default: the profile-curve grid)")
    p_cal.add_argument("--repeats", type=int, default=None)
    p_cal.add_argument("--n-null", type=int, default=None)
    p_cal.add_argument("--noise-model", default=None,
                       choices=("white", "correlated", "scaled", "outliers"))
    p_cal.add_argument("--half-pixel-offsets", action="store_true",
                       help="place every true offset at a half-pixel shift (worst case)")
    p_cal.add_argument("--out-name", default=None,
                       help="calibration subdirectory name (default: calibration)")
    p_cal.set_defaults(func=_cmd_calibrate)

    p_lim = sub.add_parser(
        "limits", help="calibrated limits, p-value, epoch table and figures"
    )
    p_lim.add_argument("--run", type=Path, required=True)
    p_lim.add_argument("--calibration", default=None,
                       help="calibration subdirectory (default: calibration)")
    p_lim.set_defaults(func=_cmd_limits)

    p_exp = sub.add_parser(
        "explore", help="optional UltraNest map of the profiled objective (diagnostic)"
    )
    _add_common(p_exp)
    p_exp.set_defaults(func=_cmd_explore)

    p_ver = sub.add_parser("verify", help="numerical verification suite")
    _add_common(p_ver)
    p_ver.add_argument("--on-data", action="store_true",
                       help="also run the checks on the real data (needs FITS files)")
    p_ver.add_argument("--out", type=Path, default=None)
    p_ver.set_defaults(func=_cmd_verify)

    p_rt = sub.add_parser("roundtrip", help="inject-and-recover on the real geometry")
    _add_common(p_rt)
    p_rt.add_argument("--tau", type=float, default=1.0e-3)
    p_rt.add_argument("--no-noise", action="store_true",
                      help="noise-free injection (must recover chi2 ~ 0 exactly)")
    p_rt.add_argument("--synthetic", action="store_true",
                      help="use standalone synthetic epochs (no FITS files needed)")
    p_rt.add_argument("--out-dir", type=Path, default=Path("outputs/dev"))
    p_rt.set_defaults(func=_cmd_roundtrip)

    p_insp = sub.add_parser("inspect-data", help="field geometry and initial forward model")
    _add_common(p_insp)
    p_insp.add_argument("--out-dir", type=Path, default=Path("outputs/dev"))
    p_insp.set_defaults(func=_cmd_inspect_data)

    p_shift = sub.add_parser(
        "inspect-shifts", help="offset landscapes and identifiability at the MLE"
    )
    p_shift.add_argument("--run", type=Path, required=True)
    p_shift.add_argument("--out-dir", type=Path, default=Path("outputs/dev"))
    p_shift.set_defaults(func=_cmd_inspect_shifts)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
