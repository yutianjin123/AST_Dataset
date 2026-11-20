"""Run the released PhyMTL-ADA checkpoint on the example cutouts.

Example
-------
    python inference.py                       # all 30 example cutouts, 8-view TTA
    python inference.py --groups simulated_lenses real_negative_lenses
    python inference.py --no-tta --out predictions.csv

The preprocessing below mirrors the training pipeline (asinh stretch, 99.9
percentile clipping, min-max normalisation, centre padding to 224x224).
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fits_io import read_fits_image
from model import MultiTaskResNet18DANN

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
BANDS = ("g", "r", "i")
GROUPS = ("simulated_lenses", "real_positive_lenses", "real_negative_lenses")


def preprocess(root: Path, group: str, filename: str, size: int = 224) -> torch.Tensor:
    """Load the three bands and reproduce the training-time normalisation."""
    stack = np.stack(
        [read_fits_image(root / group / band / filename) for band in BANDS], axis=0
    )
    stack = np.nan_to_num(stack)
    stack = np.arcsinh(stack)

    lo = float(np.min(stack))
    hi = float(np.percentile(stack, 99.9))
    stack = (stack - lo) / (hi - lo) if hi > lo else np.zeros_like(stack)
    stack = np.clip(stack, 0.0, 1.0)

    img = torch.from_numpy(stack.astype(np.float32))
    _, h, w = img.shape
    if h < size or w < size:
        pad_h, pad_w = size - h, size - w
        img = F.pad(
            img,
            (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            value=0,
        )
    elif h > size or w > size:
        img = transforms_center_crop(img, size)
    return img


def transforms_center_crop(img: torch.Tensor, size: int) -> torch.Tensor:
    _, h, w = img.shape
    top, left = (h - size) // 2, (w - size) // 2
    return img[:, top:top + size, left:left + size]


def tta_views(x: torch.Tensor) -> torch.Tensor:
    """Eight views: four rotations, then the horizontal flip rotated four times."""
    views = [torch.rot90(x, k=k, dims=[2, 3]) for k in range(4)]
    flipped = torch.flip(x, dims=[3])
    views += [torch.rot90(flipped, k=k, dims=[2, 3]) for k in range(4)]
    return torch.cat(views, dim=0)


def load_model(weights: Path, device: torch.device) -> MultiTaskResNet18DANN:
    model = MultiTaskResNet18DANN(num_classes_count=5, pretrained=False)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT,
                    help="Repository root containing the three sample folders")
    ap.add_argument("--groups", nargs="+", default=list(GROUPS))
    ap.add_argument("--weights", type=Path, default=CODE_DIR / "best_task_model.pth")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "example_predictions.csv")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-tta", action="store_true", help="Disable 8-view test-time augmentation")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device} | weights: {args.weights}")
    model = load_model(args.weights, device)

    rows = []
    with torch.no_grad():
        for group in args.groups:
            band_dir = args.data_root / group / "g"
            if not band_dir.is_dir():
                print(f"  skip {group}: {band_dir} not found")
                continue
            filenames = sorted(p.name for p in band_dir.glob("*.fits"))
            for fname in filenames:
                x = preprocess(args.data_root, group, fname).unsqueeze(0).to(device)
                x = x if args.no_tta else tta_views(x)
                out_bin, out_num, out_rein, _ = model(x, alpha=0.0)
                p_lens = torch.softmax(out_bin, dim=1)[:, 1].mean().item()
                k = int(out_num.argmax(dim=1).mode().values.item())
                theta = float(out_rein.mean().item())
                rows.append({
                    "group": group,
                    "filename": fname,
                    "p_lens": f"{p_lens:.6f}",
                    "pred_label": int(p_lens >= args.threshold),
                    "pred_image_multiplicity_k": k,
                    "theta_E_pred_arcsec": f"{theta:.4f}",
                })
            print(f"  {group}: {len(filenames)} cutouts")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
