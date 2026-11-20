# Results

Numerical results reproduced in the paper. All numbers are computed from the
independent real-domain test set (166 confirmed lenses + 500 real non-lenses)
and the 3,000-lens simulated validation subset, unless stated otherwise.

| File | Paper table / section | Content |
|---|---|---|
| `ablation_516_paper.csv` | Table 6 | ROC-AUC, AP, recall, precision, F1, TP/FP for all eight configurations |
| `ablation_516_full.csv`, `ablation_516_table.md` | Table 6 (extended) | Full metric set and a readable Markdown version |
| `domain_gap_table.csv`, `domain_gap_table.md` | Table 5 | Simulated-vs-real recall and the gap dR for each configuration |
| `physics_metrics.csv` | Table 7 | Einstein-radius MAE/RMSE/bias/Pearson r and image-multiplicity accuracy |
| `domain_distance_summary.csv`, `domain_distance_table.md` | Table 9 | CORAL, MMD^2 and Proxy A-distance |
| `feature_health.csv` | Table 9 | Low-variance feature dimensions and per-dimension standard deviation |
| `delong_test.csv` | 4.3 | Pairwise DeLong tests with Holm correction |
| `leakage_check.csv`, `robustness_leakage.csv`, `domain_confounding_metrics.csv`, `domain_confounding_report.txt` | 4.2 / 5 | Data-leakage and domain-confounding diagnostics |
| `power_analysis.csv` | 4.3 | Statistical power of the comparisons |
| `case_study_*.csv`, `case_study_summary.txt`, `case_study_realpos_summary.txt`, `common_failures.csv`, `error_analysis_summary.txt` | 4.6 / Table 10 | Per-sample hits, unique wins, common misses and error analysis |
| `example_predictions.csv` | — | Output of `code/inference.py` on the 30 released example cutouts |

`figures/` contains the corresponding plots (ROC curves, recall-gap bars,
model comparison, Einstein-radius scatter, error analysis, probability
distributions and power analysis). These are the source figures of the paper's
result section.

> The per-sample prediction CSVs of the full test set and the complete split
> manifests are not public. Contact the corresponding author for them.
