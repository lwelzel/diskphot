"""diskphot -- transit photometry of the HD 107146 debris disk.

Joint forward model of the background galaxy seen through the disk in eight
HST/STIS epochs, with the galaxy template marginalised in closed form, and
nested-sampling inference of the disk optical depth.  Code of the paper
"Measuring the dust opacity of the debris disk around HD 107146 with
transit photometry" (Welzel et al.).

Modules (each one is also a command-line entry point, ``python -m diskphot.<module>``):

    likelihood               marginal log-likelihood, data loading, audit (verify)
    run_inference            UltraNest driver of the production run
    paper_values             LaTeX macros, tables and comparison values from a run
    paper_figures            figures 1-6 of the paper
    paper_figures_schneider  epoch gallery, disk-model and ALMA-overlay figures
    subsample_posterior      equal-weight posterior subsample (CSV + provenance)
    paper_figures_subsample  corner plots and tau' marginal from the subsample

Nothing is imported here: ``diskphot.likelihood`` pulls in torch and selects
the compute device at import time, which the pure post-processing modules
do not need.
"""

__version__ = "1.0.0"
