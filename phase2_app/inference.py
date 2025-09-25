from pathlib import Path
import sys
import torch
import numpy as np

# Add project root (…/lv_capstone) to sys.path so we can import src/*
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.model import UNet2D
from src.dataset import LVH5

# Paths to data & metadata created in Phase 1
H5     = str(ROOT / "data" / "synthetic_lv_dataset_small.h5")
SPL    = str(ROOT / "splits.npz")
STATS  = str(ROOT / "norm_stats.json")

# Accept either filename depending on what you saved
CANDIDATE_WEIGHTS = [
    ROOT / "phase1_results" / "best_unet_b64.pt",
    ROOT / "phase1_results" / "best_unet.pt",
]

def _pick_weights() -> str:
    for p in CANDIDATE_WEIGHTS:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "No weights found in phase1_results/. "
        "Expected best_unet_b64.pt or best_unet.pt"
    )

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset(split: str = "test"):
    return LVH5(H5, SPL, split, STATS)

def load_model():
    device = get_device()
    wpath = _pick_weights()

    # Try base=64 first; if weight shapes mismatch, fall back to base=32.
    for base in (64, 32):
        try:
            model = UNet2D(in_ch=6, out_ch=3, base=base).to(device)
            state = torch.load(wpath, map_location=device)
            model.load_state_dict(state)
            model.eval()
            return model, device, base, wpath
        except RuntimeError:
            continue
    raise RuntimeError("Could not load weights with base=64 or base=32.")

@torch.no_grad()
def predict_index(model, device, ds, idx: int):
    """
    Returns:
      x       : torch.Tensor [6, H, W]
      y_true  : torch.Tensor [3, H, W]
      y_pred  : torch.Tensor [3, H, W]
    """
    x, y_true = ds[idx]
    x_in = x.unsqueeze(0).to(device)           # [1, 6, H, W]
    y_pred = model(x_in).cpu().squeeze(0)      # [3, H, W]
    return x, y_true, y_pred

