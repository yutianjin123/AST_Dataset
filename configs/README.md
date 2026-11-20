# Configurations

| File | Content |
|---|---|
| `best_model.yaml` | Full hyperparameters of the single released model (ResNet18 + MTL + DA). Absolute server paths are replaced by placeholders. |
| `ablation_matrix.csv` | All eight experiments of the 2 x 4 ablation: two backbones (ResNet18, Swin-T) x four variants (base, +MTL, +DA, +MTL+DA). |

All eight runs share the same protocol: random initialisation (no ImageNet
pretraining), no early stopping, 300 epochs, batch size 32, learning rate
1e-4, 224 x 224 three-band input, asinh stretch with 99.9-percentile clipping,
and `pos_weight = 3.0`. The optimiser differs by backbone (Adam/AdamW for
ResNet18, AdamW for Swin-T); see the `optimizer` and `weight_decay` columns.

Only the `resnet18_gri_mtl_da` checkpoint is released (`code/best_task_model.pth`,
`checkpoint_released = yes`). The other seven are available on request.
