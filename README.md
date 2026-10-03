# HOMER: Robust Estimation and Mean Inference via Smooth Aggregation

Code, notebooks, and saved summaries for the computational examples in the HOMER manuscript.

HOMER aggregates estimates from disjoint subsets through radial Huber losses. Adaptive thresholds provide stable point estimates, while fixed thresholds support the paper's analysis of inference for block means. The examples examine target displacement, functional contrasts, covariance matrices, and participant summaries from activity data.

## Start here

Use Python 3.11 or later. From the repository root:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python code/replot_figures.py
python code/summarize_results.py
```

The last two commands rebuild Figures 1–3 and the experimental LaTeX tables from the supplied summaries. New files go to `runs/paper/`; the reference material in `results/` is preserved. Figure export requires pdfLaTeX with PGF and Poppler's `pdftoppm`. Tests, inline notebook plots, numerical runs, and table generation do not require TeX.

For an interactive introduction, run `jupyter lab` and open [scalar.ipynb](notebooks/scalar.ipynb), [functional.ipynb](notebooks/functional.ipynb), or [covariance_wearable.ipynb](notebooks/covariance_wearable.ipynb). Every mode plots directly from its selected results inside the notebook, without saving external figure files. The default `paper` mode reads the saved results, and its embedded outputs are visible on GitHub. Set `MODE` to `smoke` or `full` and run all cells to display fresh results; no TeX installation is needed.

## Reproduce the manuscript

| Manuscript component | Code and supplied material |
| --- | --- |
| Figure 1: scalar coverage and Gaussian efficiency | `code/run_scalar.py`, `code/run_functional.py`; scalar summaries, targets, and efficiency ratios in `results/` |
| Figure 2: functional contrasts and wearable stability | `code/run_functional.py`, `code/run_wearable.py`; functional summaries and wearable results |
| Figure 3: covariance heatmaps | `code/run_covariance.py`; covariance summaries and compact display matrices |
| Experimental tables and numerical checks | `code/summarize_results.py`, `code/check_targets.py`, `code/verify_population_targets.py`; manifests and target audits |

```bash
# Short run through every study.
python code/reproduce.py --smoke

# Full manuscript designs, seeds, and replication counts.
python code/reproduce.py

# Rebuild figures from a full run and independently audit scalar targets.
python code/reproduce.py --figures --audit-targets --resume
```

Fresh experiments write to `runs/smoke/` or `runs/full/`. Smoke runs reduce replication counts and check execution; they do not reproduce the paper's Monte Carlo conclusions. Full runs regenerate the replicate records and arrays omitted from this compact repository. The supplied summaries and small plot inputs suffice to regenerate the manuscript figures without a full run.

GitHub Actions runs the tests, smoke examples, and all three notebooks in smoke mode, checking that every figure appears inline and no external figure files are written.

See [Reproducibility](docs/REPRODUCIBILITY.md) for study selection, output locations, and numerical scope; [Data sources](docs/DATA_SOURCES.md) for provenance; and [Validation](docs/VALIDATION.md) for executed checks.

## Scope and license

`code/core.py` contains the shared aggregation and sandwich routines. Inference examples use fixed deterministic thresholds and independent block means. Adaptive estimation, covariance heatmaps, and wearable perturbations illustrate stability without extending those inference guarantees.

Software is available under the [MIT License](LICENSE). The included UCI HAR participant summaries retain their [source terms and attribution](data/licenses/README.md). Please cite the accompanying manuscript when using this work; [CITATION.cff](CITATION.cff) records the software citation.
