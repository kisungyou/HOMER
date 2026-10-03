# Data and result provenance

The examples use simulated estimates and participant summaries derived from the UCI Human Activity Recognition Using Smartphones dataset. The compact inputs needed for the default workflows are included; original wearable sensor recordings are not redistributed.

## Reference results and fresh runs

`results/` contains manuscript summary CSVs, scalar targets and independent target audits, study manifests, precomputed efficiency intervals, and the small set of covariance matrices used in Figure 3. The covariance display is prespecified replication zero at the stronger perturbation, with a shared color scale. It is separate from the summaries over all replications.

Replicate-level records, full covariance arrays, functional pairing arrays, and executed notebook outputs are omitted to keep the repository small. Full runs regenerate the numerical records and arrays beneath `runs/full/`; smoke runs generate reduced versions beneath `runs/smoke/`. The reporting commands need only the supplied summaries and display inputs.

| Study | Manuscript design | Principal seed |
| --- | --- | ---: |
| Scalar | 26 cells; 2,000 replications per cell; Bernoulli and Gaussian block laws | 20260916 |
| Functional | 12 inference cells and paired estimation experiments; 500 replications; 12 Fourier coefficients | 20260917 |
| Covariance | 32 blocks of 64 observations in 12 dimensions; 500 paired replications | 20260922 |
| Wearable | 30 participants in five fixed blocks, across six activities | 20260918 |

Study manifests retain formulas, parameter choices, tolerances, and bootstrap seeds. Their original source hashes and environments describe the manuscript execution and may differ from this reorganized code. Fresh runs record their own provenance.

Comparators share clean observations, affected indices, and perturbations as specified by each design. Scalar and functional inference sample exact block laws, so their timings do not measure processing every nominal raw observation. Historical result keys such as `PH(1)` and `Adaptive PH` are retained; figures and notebooks map them to the manuscript's HOMER labels.

## Wearable inputs

Source: [Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), by Jorge L. Reyes-Ortiz, Davide Anguita, Alessandro Ghio, Luca Oneto, and Xavier Parra; [DOI: 10.24432/C54S4K](https://doi.org/10.24432/C54S4K).

Preprocessing averages squared body-acceleration magnitude within each window, then averages window energies for each participant and activity. The analysis weights participants equally; overlapping windows are not treated as independent participants.

| Included file | Contents |
| --- | --- |
| `data/derived/subject_window_energy.csv` | Participant/activity energies and original window counts |
| `data/derived/subject_partition.csv` | Five blocks of six participants shared across activities |
| `data/derived/preprocessing_manifest.json` | Processing formula, window description, source link, and hashes of the six acceleration files |

The saved summaries support the complete wearable experiment without downloading raw data. To reconstruct them from an extracted official archive, inspect `python code/run_wearable.py --help` and its optional `--data-path` argument. New derived files are written to the run directory.

The source record preserves two licensing statements: the current UCI portal lists CC BY 4.0, while the historical archive README prohibits commercial use. See [the attribution and source-term notes](../data/licenses/README.md); the software's MIT license does not replace these terms.
