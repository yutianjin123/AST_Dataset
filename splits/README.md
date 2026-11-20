# Data splits

The complete split manifests (training, simulated validation, unlabelled real
target domain, and the independent real-domain test set) are **not** included in
this public repository. They are available from the corresponding author upon
reasonable request.

What *is* visible here:

- `../data_manifest.csv` lists all 90 released example FITS files. Its `split`
  column marks the ten simulated cutouts as belonging to the training or the
  simulated validation split, and carries their true `image_multiplicity_k`
  and `theta_E_arcsec` labels taken from the project's original manifests.
- For real cutouts the `split` column is intentionally blank: the real samples
  are provided to illustrate image morphology only and are not a release of the
  evaluation split.

Independence matters for this study. The unlabelled real images used for
adversarial adaptation do not overlap the independent real-domain test set, and
the simulated validation set is disjoint from training. Any reproduction should
preserve these separations.
