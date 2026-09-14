#!/usr/bin/env python3
"""Write a small, equal-weight subsample of an UltraNest posterior to CSV.

Reads the standard UltraNest run products (no ``ultranest`` import, no
``points.hdf5`` walk -- the same reason ``paper_figures.load_run`` avoids
``ultranest.read_file``):

    <log_dir>/info/results.json           paramnames, logz, ESS, ...
    <log_dir>/chains/weighted_post.txt    weight  logl  <params...>
    <log_dir>/chains/equal_weighted_post.txt   <params...>

and writes

    <out>.csv     one row per posterior draw: sample_id, logl, <params...>
    <out>.json    provenance sidecar (run dir, seed, draw method, Kish ESS,
                  logz, raw TeX parameter names -> CSV column names, ...)

Two draw methods:

  * ``systematic`` (default): systematic (stratified) resampling of the
    importance-weighted posterior in ``weighted_post.txt``.  Unbiased,
    lower variance than multinomial resampling, keeps ``logl`` for every
    row.  Points with weight > 1/n are duplicated (reported in the
    sidecar); choose n at or below the run's ESS to make that rare.
  * ``equal``: uniform draw WITHOUT replacement from
    ``equal_weighted_post.txt`` (no duplicates).  ``logl`` is recovered by
    matching each row to ``weighted_post.txt`` (exact float match; the two
    files are written from the same array), NaN where no match exists.

Column names are cleaned from the run's TeX ``paramnames``
(``$\\delta x_3$`` -> ``dx_3``, ``$\\ln f_2$`` -> ``lnf_2``, ``$\\tau$`` ->
``tau``); the mapping is stored in the sidecar.  ``--add-f`` appends
``f_i = exp(lnf_i)`` columns for v07 runs.  ``--params`` restricts the
output to a subset of (cleaned) column names, e.g. ``--params tau dx_1``.

If ``<log_dir>`` has no ``chains/`` but contains ``run*/`` subfolders
(``resume='subfolder'`` runs), the most recent one is used.

Usage:
    python subsample_posterior.py <log_dir> [-n 2000] [--seed 0]
        [--method systematic|equal] [--add-f] [--params tau lnf_0 ...]
        [--out path/without/extension] [--float-fmt %.10g]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# run-directory discovery and loading
# --------------------------------------------------------------------------- #
def resolve_log_dir(root: Path) -> Path:
    """The directory that actually holds ``chains/``: ``root`` itself, or
    the newest ``root/run*/`` subfolder."""
    root = Path(root)
    if (root / "chains").is_dir():
        return root
    runs = sorted((p for p in root.glob("run*") if (p / "chains").is_dir()),
                  key=lambda p: (int(re.sub(r"\D", "", p.name) or 0),
                                 p.stat().st_mtime))
    if runs:
        return runs[-1]
    sys.exit(f"no chains/ under {root} (or its run*/ subfolders)")


def load_paramnames(log_dir: Path, fallback_header: list[str] | None):
    p = log_dir / "info" / "results.json"
    info = json.loads(p.read_text()) if p.exists() else {}
    names = info.get("paramnames") or fallback_header
    if not names:
        sys.exit("cannot determine parameter names (no info/results.json "
                 "and no chain header)")
    return list(names), info


def read_chain(path: Path):
    """(header tokens, float array) of an UltraNest chain text file."""
    with open(path) as fh:
        header = fh.readline().split()
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    return header, data


def load_weighted(log_dir: Path):
    """weights (normalized), logl, points from chains/weighted_post.txt."""
    header, wp = read_chain(log_dir / "chains" / "weighted_post.txt")
    if header[:2] != ["weight", "logl"]:
        raise ValueError(f"unexpected weighted_post.txt header: {header[:3]}")
    w = wp[:, 0]
    w = w / w.sum()
    return w, wp[:, 1], wp[:, 2:], header[2:]


def load_equal(log_dir: Path):
    header, eq = read_chain(log_dir / "chains" / "equal_weighted_post.txt")
    return eq, header


# --------------------------------------------------------------------------- #
# names
# --------------------------------------------------------------------------- #
_CLEAN_RULES = (
    (re.compile(r"\\delta\s*x[_\s]*\{?(\d+)\}?"), r"dx_\1"),
    (re.compile(r"\\delta\s*y[_\s]*\{?(\d+)\}?"), r"dy_\1"),
    (re.compile(r"\\ln\s*f[_\s]*\{?(\d+)\}?"), r"lnf_\1"),
    (re.compile(r"\\tau\^?\{?\\?prime\}?|\\tau'"), "tau"),
    (re.compile(r"\\tau"), "tau"),
)


def clean_name(name: str) -> str:
    s = name.strip().strip("$")
    for pat, rep in _CLEAN_RULES:
        s = pat.sub(rep, s)
    s = s.replace("\\", "").replace("{", "").replace("}", "")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_")
    return s or "p"


def clean_names(names):
    out, seen = [], {}
    for n in names:
        c = clean_name(n)
        if c in seen:                       # keep columns unique
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        out.append(c)
    return out


# --------------------------------------------------------------------------- #
# resampling
# --------------------------------------------------------------------------- #
def kish_ess(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w ** 2).sum())


def systematic_resample(w: np.ndarray, n: int, rng: np.random.Generator):
    """Systematic resampling: one uniform offset, n equally spaced
    positions on the cumulative weight.  Returns indices (sorted)."""
    cw = np.cumsum(w)
    cw[-1] = 1.0
    u = (rng.random() + np.arange(n)) / n
    return np.searchsorted(cw, u, side="right").clip(0, len(w) - 1)


def match_logl(points: np.ndarray, wpoints: np.ndarray, wlogl: np.ndarray):
    """logl for rows of ``points`` by exact match against the weighted
    chain (both files are written from the same array).  NaN if unmatched."""
    lut = {}
    for row, l in zip(wpoints, wlogl):
        lut.setdefault(row.tobytes(), l)
    out = np.full(points.shape[0], np.nan)
    for k, row in enumerate(points):
        v = lut.get(np.ascontiguousarray(row).tobytes())
        if v is not None:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_dir", help="UltraNest run root (has chains/ and info/)")
    ap.add_argument("-n", "--n-samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=("systematic", "equal"),
                    default="systematic")
    ap.add_argument("--add-f", action="store_true",
                    help="append f_i = exp(lnf_i) columns (v07 runs)")
    ap.add_argument("--params", nargs="*", default=None,
                    help="cleaned column names to keep (default: all)")
    ap.add_argument("--out", default=None,
                    help="output path without extension "
                         "(default <log_dir>/paper/posterior_subsample)")
    ap.add_argument("--float-fmt", default="%.10g")
    args = ap.parse_args(argv)

    log_dir = resolve_log_dir(Path(args.log_dir))
    rng = np.random.default_rng(args.seed)

    w, wlogl, wpts, wheader = load_weighted(log_dir)
    raw_names, info = load_paramnames(log_dir, wheader)
    if len(raw_names) != wpts.shape[1]:
        sys.exit(f"{len(raw_names)} paramnames but {wpts.shape[1]} columns "
                 f"in weighted_post.txt")
    cols = clean_names(raw_names)
    ess = kish_ess(w)
    n = int(args.n_samples)

    notes = []
    if args.method == "systematic":
        idx = systematic_resample(w, n, rng)
        pts, logl = wpts[idx], wlogl[idx]
        n_dup = int(n - len(np.unique(idx)))
        if n_dup:
            notes.append(f"{n_dup} duplicated rows (n={n} vs Kish ESS "
                         f"{ess:.0f}); lower -n or use --method equal")
    else:
        eq, eq_header = load_equal(log_dir)
        if eq.shape[1] != len(raw_names):
            sys.exit("equal_weighted_post.txt column count does not match "
                     "paramnames")
        if n > eq.shape[0]:
            notes.append(f"requested {n} > {eq.shape[0]} equal-weight "
                         "samples available; writing all of them")
            n = eq.shape[0]
        idx = np.sort(rng.choice(eq.shape[0], size=n, replace=False))
        pts = eq[idx]
        logl = match_logl(pts, wpts, wlogl)
        n_miss = int(np.isnan(logl).sum())
        if n_miss:
            notes.append(f"logl not recovered for {n_miss} rows "
                         "(no exact match in weighted_post.txt)")

    # order rows by logl (descending) is arbitrary; keep the draw order but
    # shuffle once so that a head() of the CSV is itself a fair subsample
    perm = rng.permutation(len(pts))
    pts, logl = pts[perm], logl[perm]

    df = pd.DataFrame(pts, columns=cols)
    df.insert(0, "logl", logl)
    df.insert(0, "sample_id", np.arange(len(df)))

    if args.add_f:
        for c in [c for c in cols if c.startswith("lnf_")]:
            df["f_" + c[4:]] = np.exp(df[c])

    if args.params:
        missing = [p for p in args.params if p not in df.columns]
        if missing:
            sys.exit(f"unknown column(s) {missing}; available: "
                     f"{[c for c in df.columns if c not in ('sample_id', 'logl')]}")
        df = df[["sample_id", "logl"] + list(args.params)]

    out = Path(args.out) if args.out else log_dir / "paper" / "posterior_subsample"
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out.with_suffix(".csv")
    df.to_csv(csv_path, index=False, float_format=args.float_fmt)

    # quick summary of tau (if present) for the log and the sidecar
    summary = {}
    if "tau" in df.columns:
        q = np.quantile(df["tau"].to_numpy(), [0.158655, 0.5, 0.841345, 0.95])
        summary["tau"] = {"p16": float(q[0]), "median": float(q[1]),
                          "p84": float(q[2]), "ul95": float(q[3])}

    side = {
        "source_log_dir": str(log_dir.resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": args.method, "seed": args.seed,
        "n_rows": int(len(df)),
        "n_weighted_points": int(len(w)), "kish_ess_weighted": ess,
        "run_ess": info.get("ess"), "logz": info.get("logz"),
        "logzerr": info.get("logzerr"), "niter": info.get("niter"),
        "ncall": info.get("ncall"),
        "columns": list(df.columns),
        "paramname_map": dict(zip(cols, raw_names)),
        "subsample_summary": summary,
        "notes": notes,
    }
    out.with_suffix(".json").write_text(json.dumps(side, indent=2))

    print(f"wrote {csv_path}  ({len(df)} rows x {df.shape[1]} cols; "
          f"run ESS {info.get('ess', 'n/a')}, Kish ESS of weights {ess:.0f})")
    if summary:
        t = summary["tau"]
        print(f"tau: median {t['median']:.3e}  (16-84: {t['p16']:.3e} .. "
              f"{t['p84']:.3e}),  95% UL {t['ul95']:.3e}")
    for msg in notes:
        print("note:", msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
