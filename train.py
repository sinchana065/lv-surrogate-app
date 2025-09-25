import torch, json, numpy as np
from torch.utils.data import DataLoader
from src.dataset import LVH5
from src.model import UNet2D
from src.losses import surrogate_loss

# ----------------------------
# Config
# ----------------------------
H5     = "data/synthetic_lv_dataset_small.h5"
SPL    = "splits.npz"
STATS  = "norm_stats.json"
WEIGHTS = "best_unet_b64.pt"

BATCH  = 4
EPOCHS = 200
PATIENCE = 20

device = torch.device("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ----------------------------
# Dataset + Loader
# ----------------------------
train_ds = LVH5(H5, SPL, "train", STATS)
val_ds   = LVH5(H5, SPL, "val",   STATS)

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH)

# ----------------------------
# Model + Optimizer + Scheduler
# ----------------------------
model = UNet2D(in_ch=6, out_ch=3, base=64).to(device)

opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-5)

# ----------------------------
# Training Loop
# ----------------------------
best_val, patience_counter = 1e9, 0
log = {"train": [], "val": [], "mse": [], "grad": [], "div": []}

for ep in range(EPOCHS):
    # ----- train -----
    model.train()
    tr_losses = []
    for x,y in train_dl:
        x,y = x.to(device), y.to(device)
        pred = model(x)
        loss, _ = surrogate_loss(pred, y, w_mse=1.0, w_grad=0.2, w_div=0.2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tr_losses.append(loss.item())
    tr_loss = np.mean(tr_losses)

    # ----- val -----
    model.eval()
    va_losses, mse_list, grad_list, div_list = [], [], [], []
    with torch.no_grad():
        for x,y in val_dl:
            x,y = x.to(device), y.to(device)
            pred = model(x)
            loss, comps = surrogate_loss(pred, y, w_mse=1.0, w_grad=0.2, w_div=0.2)
            va_losses.append(loss.item())
            mse_list.append(comps["mse"])
            grad_list.append(comps["grad"])
            div_list.append(comps["div"])
    va_loss = np.mean(va_losses)
    mse, grad, div = np.mean(mse_list), np.mean(grad_list), np.mean(div_list)

    # log
    log["train"].append(tr_loss)
    log["val"].append(va_loss)
    log["mse"].append(mse)
    log["grad"].append(grad)
    log["div"].append(div)

    print(f"Epoch {ep:03d} | train {tr_loss:.4f} | val {va_loss:.4f} | "
          f"mse {mse:.4f} grad {grad:.4f} div {div:.4f} | lr {opt.param_groups[0]['lr']:.2e}")

    # scheduler step
    sched.step()

    # early stopping + save best
    if va_loss < best_val:
        best_val = va_loss
        torch.save(model.state_dict(), WEIGHTS)
        print(f"Saved: {WEIGHTS}")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping.")
            break

# ----------------------------
# Save logs
# ----------------------------
np.savez("training_log.npz", **log)
print("📊 Saved training log -> training_log.npz")
print("Best val loss:", best_val)


