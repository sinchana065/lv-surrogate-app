import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from inference import load_dataset, load_model, predict_index

st.set_page_config(page_title="LV Surrogate App", layout="wide")
st.title("🫀 Left Ventricle Hemodynamics — Real-time Surrogate")

@st.cache_resource
def _cached_model():
    model, device, base, wpath = load_model()
    return model, device, base, wpath

@st.cache_resource
def _cached_dataset():
    return load_dataset(split="test")

def _imshow(ax, arr, title, vmin=None, vmax=None, cmap="jet"):
    im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    return im

def _metrics(a, b):
    a = np.asarray(a); b = np.asarray(b)
    mae  = float(np.mean(np.abs(a-b)))
    rmse = float(np.sqrt(np.mean((a-b)**2)))
    return mae, rmse

# Load once (cached)
try:
    model, device, base, wpath = _cached_model()
    ds = _cached_dataset()
    st.sidebar.success(f"Model loaded (base={base})")
    st.sidebar.caption(f"Weights: {wpath.split('/')[-1]}")
    st.sidebar.caption(f"Device: {device}")
except Exception as e:
    st.error(str(e))
    st.stop()

# Sidebar controls
st.sidebar.header("Controls")
idx = st.sidebar.slider("Sample index (test split)", 0, len(ds)-1, 0)
run = st.sidebar.button("Run Inference", type="primary")

if run:
    with st.spinner("Running model..."):
        x, y_true, y_pred = predict_index(model, device, ds, idx)
        x_np     = x.numpy()             # [6, H, W] (not used visually yet)
        y_true_n = y_true.numpy()        # [3, H, W]
        y_pred_n = y_pred.numpy()        # [3, H, W]

        # Per-channel metrics
        names = ["u","v","p"]
        rows = []
        for i, nm in enumerate(names):
            mae, rmse = _metrics(y_true_n[i], y_pred_n[i])
            rows.append((nm, mae, rmse))

        # 3x3 grid: true / pred / |error| for u,v,p
        fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
        for r, nm in enumerate(names):
            vmin = min(y_true_n[r].min(), y_pred_n[r].min())
            vmax = max(y_true_n[r].max(), y_pred_n[r].max())
            _imshow(axes[r,0], y_true_n[r], f"{nm} (true)", vmin=vmin, vmax=vmax)
            _imshow(axes[r,1], y_pred_n[r], f"{nm} (pred)", vmin=vmin, vmax=vmax)
            _imshow(axes[r,2], np.abs(y_true_n[r]-y_pred_n[r]), f"{nm} (error)", cmap="inferno")
        st.pyplot(fig)

        # Table of metrics
        st.subheader("Per-channel error")
        st.write(
            "| Channel | MAE | RMSE |\n|---|---:|---:|\n" +
            "\n".join([f"| {n} | {mae:.4f} | {rmse:.4f} |" for (n,mae,rmse) in rows])
        )

        # Velocity magnitude (pred)
        vel_mag = np.sqrt(y_pred_n[0]**2 + y_pred_n[1]**2)
        fig2, ax2 = plt.subplots(1,1, figsize=(5,4))
        _imshow(ax2, vel_mag, "Velocity magnitude (pred)")
        st.pyplot(fig2)
else:
    st.info("Pick a **sample index** on the left and click **Run Inference** to visualize predictions.")

