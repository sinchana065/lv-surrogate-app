import torch
import matplotlib.pyplot as plt
import numpy as np
from src.dataset import LVH5
from src.model import UNet2D

H5       = "data/synthetic_lv_dataset_small.h5"
SPL      = "splits.npz"
STATS    = "norm_stats.json"
WEIGHTS  = "best_unet.pt"

# device
device = torch.device("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")

# dataset + model
ds = LVH5(H5, SPL, "test", STATS)
x, y_true = ds[0]  # one sample
x = x.unsqueeze(0).to(device)

model = UNet2D(in_ch=6, out_ch=3, base=32).to(device)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval()

with torch.no_grad():
    y_pred = model(x).cpu().squeeze(0)

names = ["u","v","p"]

# --- make a combined figure (3 columns: true, pred, error) ---
fig, axes = plt.subplots(len(names), 3, figsize=(12, 10))
for i, nm in enumerate(names):
    # true
    im1 = axes[i,0].imshow(y_true[i], origin="lower", cmap="jet")
    axes[i,0].set_title(f"{nm} (true)")
    fig.colorbar(im1, ax=axes[i,0], fraction=0.046, pad=0.04)

    # pred
    im2 = axes[i,1].imshow(y_pred[i], origin="lower", cmap="jet")
    axes[i,1].set_title(f"{nm} (pred)")
    fig.colorbar(im2, ax=axes[i,1], fraction=0.046, pad=0.04)

    # error
    err = np.abs(y_true[i] - y_pred[i])
    im3 = axes[i,2].imshow(err, origin="lower", cmap="hot")
    axes[i,2].set_title(f"{nm} (error)")
    fig.colorbar(im3, ax=axes[i,2], fraction=0.046, pad=0.04)

plt.suptitle("CFD vs Surrogate Prediction + Error Maps", fontsize=14)
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig("viz_results_error.png", dpi=150)
plt.close()

print("✅ Saved visualization with error maps as viz_results_error.png")

