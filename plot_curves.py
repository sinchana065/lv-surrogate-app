import numpy as np
import matplotlib.pyplot as plt

# Load logs
logs = np.load("training_log.npz")
train_losses = logs["train_losses"]
val_losses   = logs["val_losses"]
mse_losses   = logs["mse_losses"]
grad_losses  = logs["grad_losses"]
div_losses   = logs["div_losses"]

# --- Plot train vs val loss ---
plt.figure(figsize=(6,4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
print("✅ Saved loss_curve.png")

# --- Plot components ---
plt.figure(figsize=(6,4))
plt.plot(mse_losses, label="MSE Loss")
plt.plot(grad_losses, label="Grad Loss")
plt.plot(div_losses, label="Div Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss Component")
plt.title("Validation Loss Components")
plt.legend()
plt.tight_layout()
plt.savefig("loss_components.png", dpi=150)
print("✅ Saved loss_components.png")

