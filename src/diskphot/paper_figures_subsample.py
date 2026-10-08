"""Regenerate the posterior figures of the paper from the stored equal-weight
posterior subsample (``<run_dir>/paper/posterior_subsample.csv``, written by
``diskphot.subsample_posterior``) in the style of ``diskphot.paper_figures``
-- no inference is rerun and no ``chains/`` directory is needed.  Used for
the submitted manuscript; ``diskphot.paper_figures --figs 1`` reproduces
the corner plot from the full weighted chain.

Outputs (pdf + png, in ``out_dir``):
  corner            offsets + tau' (15 parameters), replaces the 23-parameter corner
  corner_noise      ln f_0..ln f_7 + tau' (9 parameters), for the appendix
  tau_marginal      marginal posterior of tau' with the 16/50/84 % quantiles and
                    the 95 % upper limit of the full chain

Usage:
    python -m diskphot.paper_figures_subsample <out_dir> [<run_dir>]

``run_dir`` defaults to ``ultranest_run/marginal_common_tau_noisescale_v16``
under the working directory.  It must hold ``paper/posterior_subsample.csv``,
``paper/posterior_subsample.json`` and ``extra/paper_values.json`` (the
official numbers of the full chain, drawn as the quantile lines).
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from diskphot import paper_figures as pf  # noqa: E402  style + tex_label
from diskphot.paper_values import PCT16, PCT84, reflected_kde  # noqa: E402

DEFAULT_RUN = Path("ultranest_run/marginal_common_tau_noisescale_v16")


def cornerfig(df, cols, raw_names, out_dir, name):
    from ultranest.plot import cornerplot

    pts = df[cols].to_numpy(dtype=float)
    names = [r"$\tau^{\prime}$" if c == "tau" else raw_names[c] for c in cols]
    results = {"paramnames": [pf.tex_label(n) for n in names],
               "weighted_samples": {"points": pts,
                                    "weights": np.full(len(pts), 1.0 / len(pts))}}
    c_main, c_hi = pf.palette(2, 0.25, 0.80)
    pf.apply_style()
    fig = cornerplot(results, color=c_main, with_legend=False,
                     contour_kwargs={"linestyles": ["-", "-.", ":", "--"],
                                     "colors": [c_main, c_main, c_main, c_hi]},
                     label_kwargs={"fontsize": 11}, title_fmt=".2g",
                     hist_kwargs={"color": c_main}, max_n_ticks=3)
    n = len(cols)
    fig.set_size_inches(1.15 * n, 1.15 * n)
    pf.save(fig, out_dir, name)


def fig_tau_marginal(tau, ts, tau_hi, out_dir):
    """Marginal posterior of tau' from the subsample (histogram + reflected
    KDE, peak-normalised) with the quantile lines of the full chain."""
    grid = np.linspace(0.0, 0.02, 1500)
    dens = reflected_kde(tau, grid)
    c_post = pf.palette(2, 0.25, 0.85)[0]
    pf.apply_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    norm = dens.max()                                    # peak-normalised density
    bins = np.linspace(0, 0.02, 61)
    ax.hist(tau, bins=bins,
            weights=np.full(tau.size, 1.0 / (tau.size * (bins[1] - bins[0]) * norm)),
            color=c_post, alpha=0.25, lw=0, label=None)
    ax.plot(grid, dens / norm, color=c_post, lw=1.6,
            label=r"flat prior on $[0, %g]$" % tau_hi)
    ymax = 1.12
    for q, ls in ((ts["p16"], ":"), (ts["median"], "-"), (ts["p84"], ":")):
        ax.axvline(q, color="0.35", lw=0.8, ls=ls)
    ax.axvline(ts["ul95"], color="crimson", lw=1.1, ls="-.")
    ax.text(ts["ul95"] + 2.0e-4, 0.55 * ymax, "95% upper limit", color="crimson",
            fontsize=7.5, ha="left", va="top", rotation=90)
    ax.text(ts["median"] + 1.5e-4, 0.60 * ymax, "median", color="0.35",
            fontsize=7.5, ha="left", va="top", rotation=90)
    ax.text(ts["p16"] - 1.5e-4, 0.60 * ymax, "16%", color="0.35", fontsize=7,
            ha="right", va="top", rotation=90)
    ax.text(ts["p84"] + 1.5e-4, 0.60 * ymax, "84%", color="0.35", fontsize=7,
            ha="left", va="top", rotation=90)
    ax.set_xlim(0, 0.02)
    ax.set_xticks([0, 0.005, 0.010, 0.015, 0.020])
    ax.set_xticklabels(["0", "0.005", "0.010", "0.015", "0.020"])
    ax.set_ylim(0, ymax)
    ax.set_xlabel(r"$\tau^{\prime}$ (optical depth at the surface-brightness peak)")
    ax.set_ylabel("posterior density (peak = 1)")
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)
    pf.save(fig, out_dir, "tau_marginal")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="directory for the figures (pdf + png)")
    ap.add_argument("run_dir", nargs="?", default=str(DEFAULT_RUN),
                    help=f"run directory with the subsample (default {DEFAULT_RUN})")
    args = ap.parse_args(argv)
    run = Path(args.run_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(run / "paper" / "posterior_subsample.csv")
    side = json.loads((run / "paper" / "posterior_subsample.json").read_text())
    pv = json.loads((run / "extra" / "paper_values.json").read_text())
    raw_names = side["paramname_map"]           # cleaned -> TeX
    cols_off = [c for c in df.columns if c.startswith(("dx_", "dy_"))]
    cols_lnf = [c for c in df.columns if c.startswith("lnf_")]
    print("rows", len(df), "offset cols", len(cols_off), "lnf cols", len(cols_lnf))

    cornerfig(df, cols_off + ["tau"], raw_names, out, "corner")       # offsets + tau'
    cornerfig(df, cols_lnf + ["tau"], raw_names, out, "corner_noise")  # noise scales + tau'

    tau = df["tau"].to_numpy(dtype=float)
    ts = pv["tau"]                                       # official numbers (full chain)
    fig_tau_marginal(tau, ts, pv["extras"]["tau_hi"], out)

    # sanity: subsample quantiles vs official
    q = np.quantile(tau, [PCT16, 0.5, PCT84, 0.95, 0.9973])
    print("subsample tau quantiles 16/50/84/95/99.73:", ["%.3e" % v for v in q])
    print("official                              :",
          ["%.3e" % ts[k] for k in ("p16", "median", "p84", "ul95", "ul997")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
