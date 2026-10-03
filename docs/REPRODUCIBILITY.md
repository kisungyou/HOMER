# Reproducing the computational examples

Run commands from the repository root after installing `requirements.txt`. It pins the direct dependencies; `requirements-lock.txt` records the complete validation environment. No HOMER package installation or source-data download is needed.

## Workflows

| Workflow | Command | Destination |
| --- | --- | --- |
| Rebuild saved manuscript figures | `python code/replot_figures.py` | `runs/paper/figures/` |
| Rebuild experimental LaTeX tables | `python code/summarize_results.py` | `runs/paper/tables/` |
| Short numerical run | `python code/reproduce.py --smoke` | `runs/smoke/` |
| Full numerical run | `python code/reproduce.py` | `runs/full/` |

The reporting commands read the compact reference material in `results/`. They do not rerun simulations. Figure export produces PGF, PDF, and PNG files using shared Matplotlib `rcParams`, Computer Modern fonts, and the manuscript's display sizes. It requires pdfLaTeX with PGF and Poppler's `pdftoppm` on the executable path. Saved PNG previews and the notebooks need neither tool.

The numerical driver also accepts `--mode smoke` or `--mode full`. Settings are stored in `code/configs/`.

| Replications | Smoke | Full |
| --- | ---: | ---: |
| Scalar, per cell | 32 | 2,000 |
| Functional, per cell | 8 | 500 |
| Covariance, per cell | 8 | 500 |
| Bootstrap resamples | 64 | 2,000 |
| Wearable participants | 30 | 30 |

Both modes retain the study grids, generating laws, and numerical tolerances. Only full mode uses the manuscript replication counts. Floating-point results can differ across numerical libraries, and timings depend on hardware.

## Select or resume a run

```bash
python code/reproduce.py --smoke --study covariance
python code/reproduce.py --study scalar functional --dry-run
python code/reproduce.py --resume
python code/reproduce.py --study scalar --audit-targets --resume
python code/reproduce.py --figures --resume
```

`--study` accepts one or more of `scalar`, `functional`, `wearable`, and `covariance`; `sim` selects the three simulations and `all` selects every study. `--dry-run` prints the plan. `--resume` reuses completed work only when settings, source code, dependencies, input hashes, and output hashes agree; an incomplete study is rerun.

`--audit-targets` runs two independent high-precision checks of the scalar population targets. `--figures` exports figures from the selected fresh results: Figure 1 requires scalar and functional results, Figure 2 requires functional and wearable results, and Figure 3 requires covariance results.

Fresh run folders contain `results/`, `tables/`, `figures/`, derived data where needed, and execution records. Replicate arrays and logs are generated locally and excluded from Git. Reference summaries remain in `results/`; their manifests describe the original manuscript calculations rather than the refactored directory layout.

## Interpretation and checks

Scalar and functional inference uses independent block means and fixed deterministic pseudo-Huber thresholds. Coverage for the population block target and the mean is reported separately. Few-block settings and target displacement are included in the comparisons.

Adaptive estimation studies share observations and perturbations across methods. Dispersed contamination may affect more blocks than concentrated contamination. The covariance comparator averages the subset sample covariances; it is not the pooled sample covariance. Wearable results describe perturbations of participant summaries and do not measure prediction or coverage.

Coverage summaries use pointwise Wilson intervals for Monte Carlo uncertainty. Bootstrap intervals for errors, quantiles, and paired differences also describe simulation uncertainty. Normal Monte Carlo intervals for Student-t4 MSE are omitted because a finite fourth moment is unavailable. Numerical failures remain recorded in the outputs.

Run `python -m pytest -q` for numerical and workflow checks. See [the validation record](VALIDATION.md) for the checks executed for this repository and [data provenance](DATA_SOURCES.md) for seeds and inputs.
