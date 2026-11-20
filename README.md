# PhyMTL-ADA

**Physics-aware multi-task learning with adversarial domain adaptation for
simulation-to-real strong gravitational lens detection.**

Training a lens finder on simulated positives and real non-lenses couples the
*class* with the *origin* of an image: a classifier can reach near-perfect
performance on simulated validation data by exploiting survey/imaging artefacts
instead of lensing morphology. This repository accompanies a study of that gap.
We combine a shared feature extractor with three prediction heads (lens
classification, image multiplicity, Einstein radius), two soft physics
consistency constraints, and a gradient-reversal domain discriminator trained
on unlabelled real cutouts. A 2 x 4 ablation (ResNet18 / Swin-T x base, +MTL,
+DA, +MTL+DA) is evaluated on an **independent real-domain test set** of 166
confirmed lenses and 500 real non-lenses.

This repository contains **example data, the single best-performing checkpoint,
configurations and numerical results**. It is intentionally small: the full
dataset, the complete split manifests, the training scripts for all eight runs
and the remaining seven checkpoints are available from the corresponding
author on request (see [Availability](#availability--contact)).

---

## Repository layout

```
AST_Dataset/
├── README.md                 # this file
├── data_manifest.csv         # index of all 90 example FITS files
├── simulated_lenses/         # 10 simulated lenses, g / r / i
├── real_positive_lenses/     # 10 confirmed real lenses, g / r / i
├── real_negative_lenses/     # 10 real non-lenses, g / r / i
├── code/
│   ├── best_task_model.pth   # ONLY released checkpoint (~50 MB)
│   ├── model.py              # MultiTaskResNet18DANN + gradient reversal layer
│   ├── fits_io.py            # minimal FITS reader (numpy; astropy optional)
│   ├── inference.py          # run the checkpoint on the examples
│   ├── config.yaml           # released model's training hyperparameters
│   ├── requirements.txt
│   └── README.md
├── configs/
│   ├── best_model.yaml
│   ├── ablation_matrix.csv   # all eight experiments
│   └── README.md
├── results/
│   ├── *.csv / *.md / *.txt  # numeric results behind the paper's tables
│   ├── figures/              # ROC, recall gap, physics, error analysis plots
│   ├── example_predictions.csv
│   └── README.md
├── splits/README.md          # why full split lists are not public
├── CITATION.cff
├── LICENSE
├── .gitignore
└── .gitattributes
```

---

## Example data

Each of the three groups contains **10 cutouts**, each supplied in three
independent KiDS-like bands (`g`, `r`, `i`) as separate FITS files:

| Group | Cutouts | Files | Label | FITS type |
|---|---:|---:|---|---|
| `simulated_lenses` | 10 | 30 | lens (1) | 101 x 101, BITPIX = -64 |
| `real_positive_lenses` | 10 | 30 | confirmed lens (1) | 101 x 101, BITPIX = -32 |
| `real_negative_lenses` | 10 | 30 | non-lens (0) | 101 x 101, BITPIX = -32 |

Total: **90 FITS files, ~6 MB**. One cutout is identified by the same filename
in all three band folders; the model input is the stacked `(3, 224, 224)`
tensor after preprocessing. Real filenames encode RA/Dec in degrees; simulated
filenames carry a numeric simulation tag.

`data_manifest.csv` indexes every file and, for the simulated examples, also
provides the **true** `image_multiplicity_k` and `theta_E_arcsec` labels and the
train/validation split they came from. Those labels are taken from the
project's original manifests, not inferred from filenames.

> These 30 cutouts are illustrative samples, not the dataset. They must not be
> used to re-estimate the paper's performance metrics.

---

## Model — one checkpoint only

Only the best configuration is released:

| | |
|---|---|
| Architecture | ResNet18 backbone + three task heads + domain head (DANN / GRL) |
| Variant | `+MTL +DA` (experiment `resnet18_gri_mtl_da`) |
| ROC-AUC / AP | **0.9938 / 0.9889** (highest of the eight runs) |
| At threshold 0.5 | 109 / 166 real lenses recovered, **0 false positives** among 500 non-lenses |
| Checkpoint | `code/best_task_model.pth` (13.1M parameters, ~50 MB) |
| Selection | Saved at the best validation F1; used by `predict_516.py` for the paper's Table 6 |

The forward pass returns four branches: binary lens logits, 5-class image
multiplicity logits, a scalar Einstein-radius estimate, and 2-class domain
logits. The gradient-reversal layer is disabled at inference (`alpha = 0.0`).

The other seven checkpoints (Baseline / MTL / DA for both backbones) are **not
included**. If you need them, please [get in touch](#availability--contact).

---

## Quick start

```bash
git clone https://github.com/yutianjin123/AST_Dataset.git
cd AST_Dataset
pip install -r code/requirements.txt
python code/inference.py --device cpu
```

`code/inference.py` reproduces the exact training-time preprocessing
(asinh stretch -> 99.9-percentile clip -> min-max normalisation -> centre pad
to 224 x 224) and averages the same eight dihedral views used in the paper's
evaluation. It writes `results/example_predictions.csv` with the lens
probability, predicted image multiplicity and predicted Einstein radius for
all 30 examples.

Useful options:

```bash
python code/inference.py --groups simulated_lenses real_negative_lenses
python code/inference.py --no-tta            # single view
python code/inference.py --threshold 0.5
```

On the released examples the checkpoint gives mean lens probabilities of
~1.00 (simulated lenses), ~0.65 (real lenses) and ~0.0004 (real non-lenses).
That ordering is a compact illustration of the simulation-to-real gap studied
in the paper; it is **not** a benchmark. For the simulated cutout
`0_2.87.fits` the predicted Einstein radius is 0.8384" against a true value of
0.8285" recorded in `data_manifest.csv`.

> The checkpoint is stored with Git LFS (`.gitattributes`). Run
> `git lfs install` before cloning if the file appears as a small pointer.

---

## Results

`results/` holds the numeric outputs behind the paper; `results/README.md`
maps each file to its table. Highlights from `ablation_516_paper.csv` (ROC-AUC
on the independent real-domain test set):

| Variant | ResNet18 | Swin-T |
|---|---:|---:|
| base | 0.9804 | 0.9711 |
| +MTL | 0.9815 | 0.9554 |
| +DA | 0.9822 | 0.9749 |
| **+MTL+DA** | **0.9938** | **0.9807** |

Adding domain adaptation to the multi-task model improves ROC-AUC on **both**
backbones (0.9815 -> 0.9938 and 0.9554 -> 0.9807), while the best real-lens
recall (0.7952) is obtained by Swin-T + DA. Note that no pairwise AUC
difference survives Holm correction in the DeLong tests
(`results/delong_test.csv`), so these point estimates should be read as
configuration trends rather than significant rankings.

---

## Training protocol

All eight experiments use random initialisation (no pretrained weights), no
early stopping, 300 epochs, batch size 32, learning rate 1e-4, 224 x 224
three-band inputs, `pos_weight = 3.0`, and the same photometric preprocessing.
The DA runs use DANN with a progressive GRL coefficient
`alpha(p) = 2 / (1 + exp(-10 p)) - 1`. Full per-run settings are in
`configs/ablation_matrix.csv`; the released model's configuration is
`configs/best_model.yaml`.

Augmentation is limited to horizontal/vertical flips and rotations by
multiples of 90 degrees, which preserve image multiplicity and Einstein
radius. No arbitrary scaling, shearing or non-rigid warping is applied.

---

## Availability & contact

The following are **not** public in this repository:

- the full training / validation / unlabelled-target / test datasets,
- the complete split manifests,
- the remaining seven model checkpoints,
- the full training and evaluation pipelines for all eight runs,
- the per-sample prediction files of the complete test set.

They are available from the corresponding author upon reasonable request.
Please **open an issue in this repository** or contact the author directly;
this is also the preferred route if you need a different backbone or variant.

---

## Citation

See `CITATION.cff`. BibTeX:

```bibtex
@misc{yu2026phymtlada,
  author       = {Yu, Tianjin},
  title        = {{PhyMTL-ADA}: Physics-aware Multi-task Learning with
                  Adversarial Domain Adaptation for Simulation-to-Real
                  Strong Gravitational Lens Detection},
  year         = {2026},
  note         = {Manuscript; code, example data and results available at
                  https://github.com/yutianjin123/AST_Dataset}
}
```

---

## License

MIT — see `LICENSE`. The example FITS cutouts are released for
reproducibility and illustration only; their use remains subject to the
original survey data policies.
