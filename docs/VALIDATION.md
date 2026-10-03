# Validation record

Checked locally on October 3, 2026 with Python 3.12.14 on macOS arm64. Numerical versions match the manuscript: NumPy 2.4.6, SciPy 1.17.1, pandas 3.0.5, Matplotlib 3.11.2, and mpmath 1.3.0. `requirements.txt` pins the direct dependencies; `requirements-lock.txt` records the complete tested environment.

## Executed checks

- **113 tests passed.** Coverage includes estimator identities and regression cases, input validation, singular covariance, independent target audits, protected output paths, verified resume, and plotting from compact reference inputs.
- **All four smoke and full studies completed**, with both target audits and all three figures. Full output contains 52,000 scalar records, 18,000 functional contrast records, 70,000 functional point-estimator records, 4,500 covariance records, and 210 wearable fits. All recorded solver, covariance, certificate, and positive-semidefiniteness checks passed.
- **Full results reproduce the archived numerical results:** 16 CSV files and six array archives agree with relative tolerance `1e-10` and absolute tolerance `1e-12`, excluding runtime fields. See [the comparison record](reference_comparison.json).
- **Compact saved-data reporting reproduces all three PNG figure previews and all five experimental LaTeX tables byte for byte.** PGF and PDF figures also regenerate, using the manuscript font settings and layout checks.
- **All three notebooks executed in saved-result mode.** Published notebooks have no embedded outputs; executed validation copies are retained outside this compact repository.
- **Resume reused all four studies in both modes**, after checking code, settings, environment, inputs, and output hashes.

The initial smoke and full workflows, including figures and target audits, took about 48 and 79 seconds in this environment. These are observed runtimes, not hardware-independent guarantees. GitHub Actions is configured for tests and smoke examples; hosted execution has not yet been checked.

The [machine-readable record](verification.json) summarizes these checks. This validates software execution and numerical reproduction, not the mathematical proofs. Smoke results are too small for scientific conclusions.

## Repeat

```bash
python -m pytest -q
python code/reproduce.py --smoke --figures --audit-targets
python code/reproduce.py --figures --audit-targets
python code/reproduce.py --resume
python code/replot_figures.py
python code/summarize_results.py
```

Figure commands need pdfLaTeX with PGF and Poppler. Generated records, arrays, figures, and logs remain under the ignored `runs/` directory. The distributed `results/` folder contains only small summaries, targets, manifests, and the prespecified covariance display replication. Reference input hashes are listed in [reference_sha256.json](reference_sha256.json).
