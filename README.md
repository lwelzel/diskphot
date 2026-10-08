# diskphot

Transit photometry of the HD 107146 debris disk with HST/STIS: a joint
forward model of the background galaxy seen through the disk in eight
epochs, with the galaxy template marginalised in closed form, and
nested-sampling inference of the disk optical depth.

This is the analysis code of

> L. Welzel et al., *Measuring the dust opacity of the debris disk around
> HD 107146 with transit photometry*, A&A (2026).

It is research software: the model, the constants of the observations and
the figure layouts are specific to this data set and this paper.  The
repository contains everything needed to re-run the inference on the
reduced data, to regenerate every number, table and figure of the paper
from a run directory, and the record of the production run on which the
paper is based.

## Contents

| Path | What it is |
| --- | --- |
| `src/diskphot/likelihood.py` | The statistical model: native-grid data loading, the marginal log-likelihood (template integrated out, per-epoch registration offsets and noise scales, common optical depth τ′), the float64 reference evaluation, the `verify()` audit, the single-run prior-sensitivity reweighting and the `MODEL_CARD.txt` writer. The long module docstring documents the model and the numerical policy. |
| `src/diskphot/run_inference.py` | Nested-sampling driver (UltraNest) with the settings of the production run. |
| `src/diskphot/paper_values.py` | Turns a run directory into the LaTeX macros (`paper_values.tex`), the tables and `comparison_values.json` used by the paper. |
| `src/diskphot/paper_figures.py` | Figures: posterior corner plot, disk model with the galaxy track, residual grid, template eigenvalue spectrum, surface-density comparison, per-epoch noise scales. |
| `src/diskphot/paper_figures_schneider.py` | Epoch gallery, disk-model/cross-section figure and the galaxy-free stack with ALMA contours. |
| `src/diskphot/subsample_posterior.py` | Writes an equal-weight posterior subsample (CSV + provenance JSON) from the UltraNest chains. |
| `src/diskphot/paper_figures_subsample.py` | Corner plots and the τ′ marginal drawn from the stored subsample (no chains needed). |
| `ultranest_run/marginal_common_tau_noisescale_v16/` | Record of the production run: priors (`MODEL_CARD.txt`), audit numbers (`verify.json`), prior sensitivity (`sensitivity.json`), the 4096-draw posterior subsample (`paper/`), the generated macros, tables and comparison values (`extra/`), and the exact commands used (`plot_commands.txt`). The full chains are not versioned. |
| `tests/` | Audit of the likelihood implementation on synthetic epochs (runs on CPU in a few seconds). |

## Method in one paragraph

Each epoch is modelled as `t_i(τ′) · S_{s_i} G + D_i` on its own pixel grid:
the galaxy template `G` (free image), shifted by the epoch registration
`s_i` with a bicubic (Keys, a = −3/4) kernel, attenuated by
`t_i = exp(−τ′ D̂_i)` where `D̂_i` is the normalised disk-model surface
brightness, plus the disk model `D_i`.  The observations are never
resampled.  Under a Gaussian prior `G ~ N(0, λ⁻¹ I)` the template is
integrated out analytically; the normal matrix `H` is assembled exactly from
its 49 non-zero diagonals (products of kernel taps and integer-shifted weight
images) and one batched Cholesky factorisation of `H + λI` yields the
template posterior mean, χ², and `ln det(H + λI)`.  The sampled parameters
are the per-epoch offsets, per-epoch noise scales `f_i` (weights `w_i / f_i²`)
and τ′.  The hot path runs in float32 on the GPU; every reported value is
re-evaluated in float64 (`reference_logl`).  `verify()` checks the
implementation against independent calculations (adjoint identity, column
assembly of `H`, eigendecomposition log-determinant, finite-difference
gradients, the noise-scale identity) and `single_run_sensitivity` obtains
the template-prior ladder and the alternative τ′ prior from the same run by
importance reweighting.

## Installation

Python ≥ 3.12.  With [uv](https://docs.astral.sh/uv/):

```
uv sync            # creates .venv from uv.lock (add --group dev for pytest/ruff)
uv run python -m diskphot.paper_values --help
```

or with pip in any environment: `pip install -e .`

`torch` resolves to the default PyPI build.  The production run used the
CUDA 13.0 wheels (`torch 2.13.0+cu130`); see the commented block in
`pyproject.toml` to pin a CUDA or CPU-only build.  The likelihood selects
the first GPU when one is available; set `DISKPHOT_DEVICE=cpu` (or
`cuda:1`, ...) to override.  A CPU run of the full inference is slow but
correct.

## Data

The reduced data are not part of the repository (the HST data are public
in MAST under GO 12228, 14714, 15221 and 15501; the reduced epoch images
and the disk model are available from the authors on request).  Point
`DISKPHOT_DATA` at a directory with

```
<DISKPHOT_DATA>/hd107146_reduced/*.fits                                  reduced per-epoch frames; sorted file names = epoch order EP0..EP7
<DISKPHOT_DATA>/hd107146_disk_model/MODEL20_HD107146_16.5_47_h=0.3_NUP.fits   scattered-light disk model
<DISKPHOT_DATA>/galaxy_positions.csv                                     columns x, y: galaxy position per epoch in full-frame pixels ('#' comments allowed)
```

The default is `./data` relative to the working directory.  The
observation constants (exposure times, gain, background RMS, disk-model
alignment, cutout size, aperture ellipse) are module-level constants at the
top of `likelihood.py`.

The surface-density comparison figure (figure 5 of `paper_figures`) and the
ALMA overlay of `paper_figures_schneider` additionally need digitised
literature curves (`--lit-dir`, `--alma`); their expected files are
described in the respective functions.  They are not distributed here.

## Reproducing the paper

All commands run from the repository root; `RUN=ultranest_run/marginal_common_tau_noisescale_v16`.

1. **Audit and inference** (needs the data and, realistically, a GPU; the
   production run took ~6 × 10⁶ likelihood evaluations):

   ```
   python -m diskphot.run_inference --verify-only          # verify.json only
   python -m diskphot.run_inference --log-dir $RUN         # full run
   ```

   The run directory receives UltraNest's products plus `verify.json`,
   `sensitivity.json` and `MODEL_CARD.txt`.

2. **Values and tables** (`extra/paper_values.tex`, `table_*.tex`,
   `comparison_values.json`, `paper_values.json`):

   ```
   python -m diskphot.paper_values $RUN --out-dir $RUN/extra --evaluate
   ```

   `--evaluate` re-evaluates the likelihood on the data in float64 for the
   goodness-of-fit numbers and the per-epoch sightline limits; without it
   the macros that need the data are written as `\textbf{??}` and listed.

3. **Figures**:

   ```
   python -m diskphot.paper_figures $RUN --out-dir $RUN/plots --max-samples 2000 --albedo 0.2 --lit-dir literature_data
   python -m diskphot.paper_figures_schneider $RUN --out-dir $RUN/plots --arc-halfwidth 24 --profile-sigma 0.5 --combine median --alma literature_data/arks_alma/extracted
   ```

   `--figs` selects individual figures (1-6, or `gs02`, `gs0304`,
   `diskalma`); figures 1 and 6 need only the run directory.

4. **Posterior subsample and the figures drawn from it**:

   ```
   python -m diskphot.subsample_posterior $RUN --n-samples 4096 --method systematic --out $RUN/paper/posterior_subsample
   python -m diskphot.paper_figures_subsample $RUN/extra/figures $RUN
   ```

   Step 4b works without the chains and reproduces, from the committed
   record alone, `corner.pdf`, `corner_noise.pdf` and `tau_marginal.pdf` of
   the paper.

`$RUN/plot_commands.txt` lists the commands exactly as used for the paper.

## The production-run record

`ultranest_run/marginal_common_tau_noisescale_v16/` is the run behind every
number in the paper (23 sampled parameters: 14 offsets, 8 noise scales, τ′;
521 live points; 40 974 iterations; ln Z = 4569.6 ± 0.8).

- `MODEL_CARD.txt` -- the priors, the template-prior precision λ, the
  parameter layout and the likelihood constants, in the fixed wording that
  `paper_values.load_model_card` parses.
- `verify.json` -- the `verify()` audit numbers on the data (quoted in the
  appendix of the paper).
- `sensitivity.json` -- τ′ quantiles, Kish ESS and Δln Z of the λ ladder and
  of the soft-log τ′ prior, from importance reweighting of this run.
- `paper/posterior_subsample.{csv,json}` -- 4096 systematic draws from the
  weighted posterior with `logl`, cleaned column names (`dx_1`, `lnf_0`,
  `tau`, `f_0`, ...) and a provenance sidecar.
- `extra/` -- `paper_values.tex` (one `\newcommand` per reported value),
  the four tables, `comparison_values.json` and `paper_values.json`.

## Tests

```
uv run --group dev pytest
```

`tests/test_likelihood.py` builds the likelihood on small synthetic epochs
and checks the `verify()` identities (adjoint, stencil-vs-column `H`,
Cholesky-vs-eigendecomposition log-determinant, noise-scale identity,
legacy parameter layout, finite-difference gradient), the common-grid
identity with the pairwise statistic, the model-card round trip, and the
post-processing helpers.

## Software versions used for the paper

Python 3.12, numpy 2.5.2, torch 2.13.0 (+cu130), ultranest 4.5.0,
astropy 8.0.1, scipy 1.18.0, matplotlib 3.11.1, cmasher 1.9.2,
pandas 3.0.5, h5py 3.16.0.  `uv.lock` pins a fresh resolution of the
trimmed dependency list made for this release (same versions except
torch, which it takes from PyPI); it is not a byte-for-byte record of
the environment of the production run.

## License and citation

MIT License (see `LICENSE`).  If you use this code, please cite the paper
and the software (`CITATION.cff`).
