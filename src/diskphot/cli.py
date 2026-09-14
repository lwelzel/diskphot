"""Command-line interface: ``python -m diskphot <command>`` or ``diskphot``."""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

import numpy as np


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--device", type=str, default=None, help="auto | cuda | cpu")
    p.add_argument("--dtype", type=str, default=None, choices=["float64", "float32"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--variant", type=str, default=None,
                   choices=["common", "per-epoch", "grey", "null"])
    p.add_argument("--sigma-bg", type=float, default=None)
    p.add_argument("--gain", type=float, default=None)
    p.add_argument("--variance-source", type=str, default=None,
                   choices=["model", "plugin", "background"])
    p.add_argument("--mask-pixel", type=int, nargs=3, action="append", default=None,
                   metavar=("EPOCH", "X", "Y"), help="exclude a cutout pixel (repeatable)")
    p.add_argument("--offset-halfwidth", type=float, default=None)
    p.add_argument("--tau-hi", type=float, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--resolve-every", type=int, default=None)
    p.add_argument("--config", type=Path, default=None,
                   help="start from an existing config.json")


def _build_config(args: argparse.Namespace):
    from .config import DataConfig, FitConfig, Variant

    cfg = FitConfig.from_json(args.config) if args.config else FitConfig()
    if args.variant:
        cfg.variant = Variant(args.variant)
    for name in ("log_dir", "run_name", "device", "dtype", "seed"):
        v = getattr(args, name)
        if v is not None:
            setattr(cfg, name, v)
    d = cfg.data
    if args.data_dir is not None:
        d.data_dir = args.data_dir
    if args.sigma_bg is not None:
        d.sigma_bg = args.sigma_bg
    if args.gain is not None:
        d.gain = args.gain
    if args.variance_source is not None:
        d.variance_source = args.variance_source
    if args.mask_pixel:
        d.mask_pixels = tuple(tuple(px) for px in args.mask_pixel)
    if args.offset_halfwidth is not None:
        cfg.profile.offset_halfwidth_px = args.offset_halfwidth
    if args.tau_hi is not None:
        cfg.profile.tau_hi = args.tau_hi
    if args.steps is not None:
        cfg.profile.steps = args.steps
    if args.resolve_every is not None:
        cfg.solve.resolve_every = args.resolve_every
    return cfg


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="diskphot")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, doc in (
        ("fit", "observed fit of one variant"),
        ("calibrate", "injection-recovery calibration campaign"),
        ("limits", "statistics from an existing calibration"),
        ("explore", "UltraNest map of the profiled objective (diagnostic)"),
        ("verify", "numerical self-checks (no data needed)"),
        ("roundtrip", "end-to-end synthetic injection/recovery"),
        ("inspect-data", "field overview figures"),
        ("inspect-shifts", "1-D offset slices of the objective"),
    ):
        p = sub.add_parser(name, help=doc)
        _add_common(p)
        if name == "calibrate":
            p.add_argument("--run", type=Path, default=None,
                           help="run dir whose config to reuse")
            p.add_argument("--taus", type=float, nargs="+", default=None)
            p.add_argument("--repeats", type=int, default=None)
            p.add_argument("--n-null", type=int, default=None)
            p.add_argument("--noise-model", type=str, default=None,
                           choices=["white", "correlated", "scaled", "outliers"])
            p.add_argument("--half-pixel-offsets", action="store_true")
            p.add_argument("--out-name", type=str, default="calibration")
        if name == "limits":
            p.add_argument("--run", type=Path, required=True)
            p.add_argument("--calibration", type=Path, default=None,
                           help="calibration dir (default: <run>/calibration)")
        if name == "roundtrip":
            p.add_argument("--synthetic", action="store_true",
                           help="(default) synthetic epochs; kept for symmetry")
            p.add_argument("--tau", type=float, default=5.0e-2)
            p.add_argument("--no-noise", action="store_true")

    args = parser.parse_args(argv)
    cfg = _build_config(args)

    if args.command == "fit":
        from .inference import run_fit

        payload = run_fit(cfg)
        for key in ("logl_null", "logl_mle", "q0"):
            if key in payload:
                print(f"{key} = {payload[key]:.4f}")

    elif args.command == "calibrate":
        from .calibrate import run_calibration
        from .config import FitConfig
        from .inference import prepare_data

        if args.run is not None:
            cfg = FitConfig.from_json(Path(args.run) / "config.json")
        c = cfg.calib
        if args.taus:
            c.taus = tuple(args.taus)
        if args.repeats is not None:
            c.n_repeats = args.repeats
        if args.n_null is not None:
            c.n_null = args.n_null
        if args.noise_model is not None:
            c.noise_model = args.noise_model
        if args.half_pixel_offsets:
            c.half_pixel_offsets = True
        data = prepare_data(cfg)
        out = cfg.run_dir / args.out_name
        run_calibration(cfg, data, out)
        print(f"calibration written to {out}")

    elif args.command == "limits":
        from astropy.table import Table

        from .calibrate import analyze_calibration
        from .config import FitConfig
        from .inference import prepare_data
        from .plotting import detection_efficiency_plot, profile_curve_plot
        from .results import epoch_table, load_fit, summary_lines

        run = Path(args.run)
        cfg = FitConfig.from_json(run / "config.json")
        cal_dir = args.calibration or (run / "calibration")
        trials = Table.read(cal_dir / "trials.ecsv", format="ascii.ecsv")
        nulls = Table.read(cal_dir / "nulls.ecsv", format="ascii.ecsv")
        observed = load_fit(run).as_payload()
        summary = analyze_calibration(trials, nulls, cfg, observed,
                                      out_path=run / "limits_summary.json")
        detection_efficiency_plot(summary, run / "detection_efficiency.png")
        if "logl_curve" in observed:
            profile_curve_plot(observed, summary, run / "profile_curve.png")
        data = prepare_data(cfg)
        epoch_table(summary, data, run / "epoch_table.ecsv")
        for line in summary_lines(summary):
            print(line)

    elif args.command == "explore":
        from .inference import explore

        explore(cfg)

    elif args.command == "verify":
        from .diagnostics import run_verify

        out = cfg.run_dir / "verify.json"
        result = run_verify(out_path=out, seed=cfg.seed)
        for name, c in result["checks"].items():
            print(f"{'PASS' if c['pass'] else 'FAIL'}  {name}: "
                  f"{c['value']:.3e} (tol {c['tol']:.1e})")
        print("all pass" if result["all_pass"] else "FAILURES present")

    elif args.command == "roundtrip":
        from .diagnostics import run_roundtrip

        out = cfg.run_dir / "roundtrip"
        summary = run_roundtrip(cfg, tau_true=args.tau,
                                add_noise=not args.no_noise, out_dir=out, seed=cfg.seed)
        for k, v in summary.items():
            print(f"{k} = {v}")

    elif args.command == "inspect-data":
        from .inference import prepare_data
        from .plotting import plot_field_overview

        data = prepare_data(cfg)
        out = cfg.run_dir / "field_overview.png"
        plot_field_overview(data, out)
        print(f"wrote {out}")

    elif args.command == "inspect-shifts":
        from .diagnostics import theta_initial
        from .inference import prepare_data
        from .model import JointLogLikelihood
        from .plotting import plot_shift_scan

        data = prepare_data(cfg)
        model = JointLogLikelihood(data, cfg.variant, cfg.solve)
        theta = theta_initial(model)
        out_dir = cfg.run_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, data.n_epochs):
            for axis in (0, 1):
                plot_shift_scan(model, theta, epoch=epoch, axis=axis,
                                out_path=out_dir / f"shift_scan_ep{epoch}_{'xy'[axis]}.png")
        print(f"wrote shift scans to {out_dir}")


if __name__ == "__main__":
    main()
