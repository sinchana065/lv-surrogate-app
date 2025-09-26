import os, io, time, sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import streamlit as st
import h5py
import pandas as pd

# --------------------------------------------------------
# Ensure "src" folder (from lv_capstone) is importable
# --------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from fno import FNO2d  # src/fno.py must define FNO2d

# --------------------------------------------------------
# App configuration
# --------------------------------------------------------
st.set_page_config(page_title="LV Digital Twin", layout="wide")
st.title("🫀 LV Digital Twin — Real-time Hemodynamics ")

DEVICE = torch.device("cpu")  # Streamlit Cloud runs CPU
IN_CH, OUT_CH = 1, 3
GRID_H, GRID_W = 128, 128
H5_PATH = os.path.join(ROOT, "data", "synthetic_lv_dataset_small.h5")
CKPT_PATH = os.path.join(ROOT, "fno_mcwilliams.pth")

# --------------------------------------------------------
# Cached loaders
# --------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model() -> nn.Module:
    model = FNO2d(IN_CH, OUT_CH, modes1=16, modes2=16, width=32).to(DEVICE)
    if not os.path.exists(CKPT_PATH):
        st.error(f"❌ Model weights not found at {CKPT_PATH}")
        st.stop()
    sd = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()
    return model

@st.cache_data(show_spinner=False)
def load_h5():
    if not os.path.exists(H5_PATH):
        st.error(f"❌ Dataset not found at {H5_PATH}")
        st.stop()
    with h5py.File(H5_PATH, "r") as f:
        sdf = f["inputs/sdf"][:]           # (N,128,128)
        u = f["labels/u"][:]
        v = f["labels/v"][:]
        p = f["labels/p"][:]
        scalars = {
            "hr": f["inputs/hr"][:],
            "sbp": f["inputs/sbp"][:],
            "dbp": f["inputs/dbp"][:],
            "phase": f["inputs/phase"][:],
        }
    X = torch.tensor(sdf, dtype=torch.float32).unsqueeze(1)   # (N,1,H,W)
    Y = torch.stack([
        torch.tensor(u, dtype=torch.float32),
        torch.tensor(v, dtype=torch.float32),
        torch.tensor(p, dtype=torch.float32)
    ], dim=1)  # (N,3,H,W)
    return X, Y, scalars

model = load_model()
X, Y, scalars = load_h5()
N = X.shape[0]

# --------------------------------------------------------
# Sidebar Controls
# --------------------------------------------------------
st.sidebar.header("Controls")

presets = {
    "Healthy (example)": 0,
    "Hypertensive (example)": min(5, N-1),
    "Tachycardia (example)": min(10, N-1)
}
preset_choice = st.sidebar.selectbox("Scenario Preset", list(presets.keys()))
default_idx = presets[preset_choice]

idx = st.sidebar.slider("Validation Sample", 0, N-1, value=default_idx, step=1)

# Display dataset parameters
st.sidebar.markdown("### Sample Parameters")
st.sidebar.write(f"HR: **{float(scalars['hr'][idx]):.1f} bpm**")
st.sidebar.write(f"SBP: **{float(scalars['sbp'][idx]):.1f} mmHg**")
st.sidebar.write(f"DBP: **{float(scalars['dbp'][idx]):.1f} mmHg**")
st.sidebar.write(f"Phase: **{int(scalars['phase'][idx])}**")

show_diff = st.sidebar.checkbox("Show difference maps (Pred − True)", value=True)

# --------------------------------------------------------
# Inference
# --------------------------------------------------------
xb = X[idx:idx+1].to(DEVICE)
yb = Y[idx:idx+1].to(DEVICE)

with torch.no_grad():
    t0 = time.time()
    pred = model(xb)
    infer_ms = (time.time() - t0) * 1000.0

# Metrics
mse_overall = torch.mean((pred - yb) ** 2).item()
mse_ch = torch.mean((pred - yb) ** 2, dim=(0,2,3)).squeeze(0).cpu().numpy()

colA, colB, colC, colD, colE = st.columns(5)

colA.metric("MSE (overall)", f"{mse_overall:.4f}")
colB.metric("MSE u", f"{mse_ch[0]:.4f}")
colC.metric("MSE v", f"{mse_ch[1]:.4f}")
colD.metric("MSE p", f"{mse_ch[2]:.4f}")
colE.metric("Inference time", f"{infer_ms:.1f} ms")


# --------------------------------------------------------
# Visualization
# --------------------------------------------------------
fields = ["u", "v", "p"]
pred_np = pred[0].cpu().numpy()
true_np = yb[0].cpu().numpy()

fig, axes = plt.subplots(2, 3, figsize=(10, 6))
for j in range(3):
    im = axes[0, j].imshow(pred_np[j], cmap="jet")
    axes[0, j].set_title(f"Pred {fields[j]}")
    axes[0, j].axis("off")
    plt.colorbar(im, ax=axes[0, j], fraction=0.046, pad=0.04)
for j in range(3):
    im = axes[1, j].imshow(true_np[j], cmap="jet")
    axes[1, j].set_title(f"True {fields[j]}")
    axes[1, j].axis("off")
    plt.colorbar(im, ax=axes[1, j], fraction=0.046, pad=0.04)
plt.tight_layout()
st.pyplot(fig, clear_figure=True)

# Difference maps
if show_diff:
    diff = pred_np - true_np
    fig2, ax2 = plt.subplots(1, 3, figsize=(10, 3))
    for j in range(3):
        im = ax2[j].imshow(diff[j], cmap="bwr")
        ax2[j].set_title(f"Diff {fields[j]} (Pred−True)")
        ax2[j].axis("off")
        plt.colorbar(im, ax=ax2[j], fraction=0.046, pad=0.04)
    plt.tight_layout()
    st.pyplot(fig2, clear_figure=True)

# --------------------------------------------------------
# Downloads (fixed PNG export)
# --------------------------------------------------------
# Re-generate the Pred/True plot for saving
fig_dl, axes_dl = plt.subplots(2, 3, figsize=(10, 6))
for j in range(3):
    im = axes_dl[0, j].imshow(pred_np[j], cmap="jet")
    axes_dl[0, j].set_title(f"Pred {fields[j]}")
    axes_dl[0, j].axis("off")
    plt.colorbar(im, ax=axes_dl[0, j], fraction=0.046, pad=0.04)
for j in range(3):
    im = axes_dl[1, j].imshow(true_np[j], cmap="jet")
    axes_dl[1, j].set_title(f"True {fields[j]}")
    axes_dl[1, j].axis("off")
    plt.colorbar(im, ax=axes_dl[1, j], fraction=0.046, pad=0.04)
plt.tight_layout()

buf = io.BytesIO()
fig_dl.savefig(buf, format="png", dpi=300, bbox_inches="tight")
buf.seek(0)
st.download_button("⬇️ Download Pred/True PNG", data=buf,
                   file_name=f"lv_pred_true_{idx}.png", mime="image/png")
plt.close(fig_dl)

# CSV
df = pd.DataFrame({
    "sample_idx":[idx],
    "mse_overall":[mse_overall],
    "mse_u":[float(mse_ch[0])],
    "mse_v":[float(mse_ch[1])],
    "mse_p":[float(mse_ch[2])],
    "hr":[float(scalars['hr'][idx])],
    "sbp":[float(scalars['sbp'][idx])],
    "dbp":[float(scalars['dbp'][idx])],
    "phase":[int(scalars['phase'][idx])]
})
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download metrics CSV", data=csv_bytes,
                   file_name=f"lv_metrics_{idx}.csv", mime="text/csv")

st.caption("Prototype app: FNO2d surrogate for LV hemodynamics. "
           "Features: sample slider, presets, metrics, difference maps, downloads.")

