"""PhyMTL-ADA best-performing model: ResNet18 + physics-aware MTL + DANN.

This file is a self-contained copy of the architecture used by the released
checkpoint (``best_task_model.pth``). It reproduces ``src/model_dann.py`` and
``src/utils.py`` from the training code so that the checkpoint can be loaded
without the rest of the training repository.

Outputs are raw logits for four branches:
    out_bin    : lens / non-lens classification (2 logits)
    out_num    : image-multiplicity classification (5 logits)
    out_rein   : Einstein-radius regression (1 value)
    out_domain : domain discrimination (2 logits), fed through a GRL
"""

import torch
import torch.nn as nn
from torchvision import models


class GradientReversalFn(torch.autograd.Function):
    """Identity in the forward pass, negates and scales the gradient by alpha."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradientReversalFn.apply(x, alpha)


class MultiTaskResNet18DANN(nn.Module):
    def __init__(self, num_classes_count=5, pretrained=False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        self.head_bin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 2),
        )
        self.head_num = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes_count),
        )
        self.head_rein = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.head_domain = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, 2),
        )

    def forward(self, x, alpha=0.0):
        # x: [B, 3, 224, 224] built from the g, r, i bands
        feat = self.features(x)                       # [B, 512, 1, 1]
        out_bin = self.head_bin(feat)
        out_num = self.head_num(feat)
        out_rein = self.head_rein(feat)
        out_domain = self.head_domain(grad_reverse(feat, alpha))
        return out_bin, out_num, out_rein, out_domain
