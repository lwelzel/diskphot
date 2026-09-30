"""Regenerate the posterior figures of the paper from the stored equal-weight
posterior subsample (paper/posterior_subsample.csv, written by
subsample_posterior.py) in the style of paper_figures.py -- no inference is
rerun and no chains/ directory is needed.  Used for the submitted manuscript;
paper_figures.py --figs 1 reproduces figure 1 from the full weighted chain.

Outputs (in the output directory given as the first argument; the run
directory may be given as the second argument):
  corner.pdf          offsets + tau' (15 parameters), replaces the 23-parameter corner
  corner_noise.pdf    ln f_0..ln f_7 + tau' (9 parameters), for the appendix
  tau_marginal.pdf    marginal posterior of tau' with quantiles, the 95% limit and the
"""
import json, sys, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_figures as pf              # style + tex_label
from paper_values import reflected_kde, PCT16, PCT84

RUN = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[2] / "ultranest_run/marginal_common_tau_noisescale_v16"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RUN / "paper/posterior_subsample.csv")
side = json.loads((RUN / "paper/posterior_subsample.json").read_text())
pv = json.loads((RUN / "extra/paper_values.json").read_text())
raw_names = side["paramname_map"]           # cleaned -> TeX
cols_off = [c for c in df.columns if c.startswith(("dx_", "dy_"))]
cols_lnf = [c for c in df.columns if c.startswith("lnf_")]
print("rows", len(df), "offset cols", len(cols_off), "lnf cols", len(cols_lnf))

def cornerfig(cols, name):
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
    # tau' title in scientific notation (the default .2g prints 0.0031 fine; force
    # a compact power-of-ten form for the axis label)
    pf.save(fig, OUT, name)

# 1) offsets + tau'
cornerfig(cols_off + ["tau"], "corner")
# 2) noise scales + tau'
cornerfig(cols_lnf + ["tau"], "corner_noise")

# 3) tau' marginal
tau = df["tau"].to_numpy(dtype=float)
ts = pv["tau"]                                       # official numbers (full chain)
tau_hi = pv["extras"]["tau_hi"]
grid = np.linspace(0.0, 0.02, 1500)
dens = reflected_kde(tau, grid)
c_post = pf.palette(2, 0.25, 0.85)[0]
pf.apply_style()
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.hist(tau, bins=np.linspace(0, 0.02, 61), density=True, color=c_post, alpha=0.25,
        lw=0, label=None)
ax.plot(grid, dens, color=c_post, lw=1.6, label=r"flat prior on $[0, %g]$" % tau_hi)
ymax = 1.12 * dens.max()
for q, lab, ls in ((ts["p16"], "16", ":"), (ts["median"], "50", "-"), (ts["p84"], "84", ":")):
    ax.axvline(q, color="0.35", lw=0.8, ls=ls)
ax.axvline(ts["ul95"], color="crimson", lw=1.1, ls="-.")
ax.text(ts["ul95"] + 2.0e-4, 0.55 * ymax, "95% upper limit", color="crimson", fontsize=7.5, ha="left", va="top", rotation=90)
ax.text(ts["median"] + 1.5e-4, 0.60 * ymax, "median", color="0.35", fontsize=7.5, ha="left", va="top", rotation=90)
ax.text(ts["p16"] - 1.5e-4, 0.60 * ymax, "16%", color="0.35", fontsize=7, ha="right", va="top", rotation=90)
ax.text(ts["p84"] + 1.5e-4, 0.60 * ymax, "84%", color="0.35", fontsize=7, ha="left", va="top", rotation=90)
ax.set_xlim(0, 0.02)
ax.set_xticks([0, 0.005, 0.010, 0.015, 0.020])
ax.set_xticklabels(["0", "0.005", "0.010", "0.015", "0.020"])
ax.set_ylim(0, ymax)
ax.set_xlabel(r"$\tau^{\prime}$ (optical depth at the surface-brightness peak)")
ax.set_ylabel("posterior density")
ax.legend(loc="upper right", fontsize=7.5, frameon=False)
pf.save(fig, OUT, "tau_marginal")

# sanity: subsample quantiles vs official
q = np.quantile(tau, [PCT16, 0.5, PCT84, 0.95, 0.9973])
print("subsample tau quantiles 16/50/84/95/99.73:", ["%.3e" % v for v in q])
print("official                              :", ["%.3e" % ts[k] for k in ("p16", "median", "p84", "ul95", "ul997")])
