# Dataset attribution and source terms

The MIT license at the repository root applies to the software and its documentation. It does not replace the terms of third-party data.

## UCI Human Activity Recognition Using Smartphones

The participant summaries in `data/derived/subject_window_energy.csv` were derived from the [UCI Human Activity Recognition Using Smartphones dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). The original sensor files are not included.

Credit: Jorge L. Reyes-Ortiz, Davide Anguita, Alessandro Ghio, Luca Oneto, and Xavier Parra. The derived summaries average squared body-acceleration magnitude within windows, then within each participant/activity group. The processing formula and original input hashes are preserved in `data/derived/preprocessing_manifest.json`.

That manifest records both of these source statements:

- The current UCI portal identifies the dataset license as [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
- The README in the historical dataset archive prohibits commercial use.

These notes preserve the provenance of both statements; they do not grant additional rights or resolve the difference between them. Consult the source and its terms for the intended use, retaining attribution and identifying any changes to the data.

The block partition in `subject_partition.csv` is part of this study's experimental design. The wearable result tables are derived analyses of the participant summaries and should carry the same source attribution.
