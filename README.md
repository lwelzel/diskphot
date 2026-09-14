# diskphot v0.3.0

Frequentist transit photometry of the HD 107146 debris disk against the
background galaxy, on native pixel grids.  v0.4.0 is a performance and
simplification rewrite of the v0.3 statistics-revision release.

## What changed in v0.4.0

- **Shifting is torch-native.**  All sub-pixel translations go through
  `torch.nn.functional.grid_sample(mode="bicubic")` -- the separable Keys
  (1981) cubic-convolution kernel with a = -3/4, one fused CPU/CUDA kernel
  per call -- replacing the hand-rolled per-tap gathers that dominated the
  v0.3 profile.  For a pure translation with this symmetric kernel and zero
  padding the exact adjoint is a negated shift, `S_s^T = S_{-s}`, verified
  against autograd's transpose in `diskphot verify`.
- **One solver everywhere.**  The template normal equations are assembled
  from the same grid-sampler operator and solved by batched, ridge-
  regularized Cholesky in float64 (`SolveConfig.ridge`, relative to the
  largest diagonal; default 1e-12).  The CG path, its stopping heuristics
  and the tight/loose duality are gone; the estimator is a single
  deterministic function of the data, identical for observed and simulated
  epochs, so the injection calibration applies unchanged.
- **Block-coordinate optimizer.**  The template is re-solved exactly every
  `SolveConfig.resolve_every` Adam steps and held fixed in between (a
  rigorous lower bound of the profile); best-point tracking and every
  reported log-likelihood use a fresh exact solve.  Multi-starts and
  calibration realizations run as one GPU batch.
- **Lazy imports.**  `import diskphot` no longer pulls matplotlib (or any
  plotting) -- submodules resolve on first use (PEP 562).
- Breaking config change: `FitConfig.cg` (CGConfig) is now `FitConfig.solve`
  (SolveConfig: `ridge`, `chunk`, `resolve_every`).  Old run configs need
  that one key renamed.

## Layout

- `data.py` -- FITS loading, native-grid cutouts, masks, variance model
- `transforms.py` -- grid-sampler shift/translate (adjoint = negated shift)
- `model.py` -- profiled joint likelihood; ridge-Cholesky template solve
- `inference.py` -- prepare_data, block-descent fits, profile curve, run_fit
- `synthetic.py` / `calibrate.py` -- injection-recovery campaign and the
  calibrated statistics (threshold, power, Neyman limits, coverage)
- `results.py`, `plotting.py`, `diagnostics.py`, `cli.py`

## CLI

```
diskphot fit        --variant common
diskphot calibrate  --run outputs/runs/common
diskphot limits     --run outputs/runs/common
diskphot verify     # numerical self-checks, no data needed
diskphot roundtrip  --tau 5e-2 [--no-noise]
```

GPU: set `--device cuda`; memory is bounded by `SolveConfig.chunk`
(members per H-assembly chunk) and batch sizes (`CalibConfig.n_null`,
`n_repeats`).
