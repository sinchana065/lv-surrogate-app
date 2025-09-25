import torch, numpy as np
from torch.utils.data import DataLoader
from src.dataset import LVH5
from src.model import UNet2D
import piq  # pip install piq

H5       = "data/synthetic_lv_dataset_small.h5"
SPL      = "splits.npz"
STATS    = "norm_stats.json"
WEIGHTS  = "best_unet.pt"

# device
device = torch.device("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")

# dataset + loader
test_ds = LVH5(H5, SPL, "test", STATS)
test_dl = DataLoader(test_ds, batch_size=4, shuffle=False)

# model
model = UNet2D(in_ch=6, out_ch=3, base=32).to(device)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval()

mae_list, rmse_list, ssim_list = [], [], []

with torch.no_grad():
    for x, y in test_dl:
        x, y = x.to(device), y.to(device)
        pred = model(x)

        # --- MAE and RMSE ---
        mae = torch.mean(torch.abs(pred - y)).item()
        rmse = torch.sqrt(torch.mean((pred - y)**2)).item()

        # --- SSIM (channel-wise, rescaled to [0,1], then average) ---
        ssim_vals = []
        for i in range(3):  # u, v, p
            pred_rescaled = (pred[:, i:i+1] - pred[:, i:i+1].min()) / (
                pred[:, i:i+1].max() - pred[:, i:i+1].min() + 1e-8
            )
            y_rescaled = (y[:, i:i+1] - y[:, i:i+1].min()) / (
                y[:, i:i+1].max() - y[:, i:i+1].min() + 1e-8
            )
            ssim_val = piq.ssim(pred_rescaled, y_rescaled, data_range=1.0)
            ssim_vals.append(ssim_val.item())
        ssim = np.mean(ssim_vals)

        mae_list.append(mae)
        rmse_list.append(rmse)
        ssim_list.append(ssim)

print("==== Test Metrics ====")
print(f"MAE : {np.mean(mae_list):.4f}")
print(f"RMSE: {np.mean(rmse_list):.4f}")
print(f"SSIM: {np.mean(ssim_list):.4f}")

