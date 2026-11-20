# Code — best-performing checkpoint only

Only **one** model is released here: the best configuration of the ablation
matrix, **ResNet18 + physics-aware multi-task learning + adversarial domain
adaptation (DANN)**. It achieved the highest ROC-AUC (0.9938) and average
precision (0.9889) on the independent real-domain test set, with no false
positives among the 500 real non-lenses at the fixed 0.5 threshold.

| File | Purpose |
|---|---|
| `best_task_model.pth` | The single released checkpoint (13.1M parameters, ~50 MB). Saved on the best validation F1 during training, and used by `predict_516.py` to produce the paper's Table 6. |
| `model.py` | Self-contained definition of `MultiTaskResNet18DANN` and the gradient-reversal layer. |
| `fits_io.py` | Minimal dependency-free FITS reader (falls back to astropy when available). |
| `inference.py` | Runs the checkpoint on the example cutouts with the exact training-time preprocessing. |
| `config.yaml` | Hyperparameters used to train the released model (paths replaced by placeholders). |
| `requirements.txt` | Minimal runtime dependencies. |

## Run

From the repository root:

```bash
pip install -r code/requirements.txt
python code/inference.py
```

This writes `results/example_predictions.csv` with the lens probability,
predicted image multiplicity and predicted Einstein radius for each of the
30 example cutouts. Useful options:

```bash
python code/inference.py --groups simulated_lenses real_negative_lenses
python code/inference.py --no-tta            # single view instead of 8-view TTA
python code/inference.py --device cpu
```

`inference.py` averages over the same eight dihedral views (four rotations and
their horizontal flips) used for the paper's evaluation. The GRL is disabled at
inference (`alpha=0.0`), so the domain head has no effect on the predictions.

## What is *not* included

The other seven checkpoints (Baseline / MTL / DA across both backbones), the
training scripts for the full ablation, and the complete dataset are **not**
released here. They are available from the corresponding author upon reasonable
request — please open an issue in this repository or contact the author directly.
