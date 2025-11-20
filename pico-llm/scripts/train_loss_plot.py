import json
import os
import matplotlib.pyplot as plt

dir_name = "trained_outputs/outputs_tiny_overfit"
LOSS_LOG_PATH = os.path.join(dir_name, "loss_logs.json")

with open(LOSS_LOG_PATH, "r", encoding="utf-8") as f:
    logs = json.load(f)

model_name = "kvcache_transformer"
data = logs[model_name]

train_ll = data["train"]   # list of list (per-step)
test_ll  = data["test"]    # list of list (per-batch)


# ---- 1) Compute mean train loss per epoch ----
train_means = [
    sum(epoch_losses) / len(epoch_losses)
    if len(epoch_losses) > 0 else float("nan")
    for epoch_losses in train_ll
]

# ---- 2) Compute mean test loss per epoch ----
test_means = [
    sum(epoch_losses) / len(epoch_losses)
    if len(epoch_losses) > 0 else float("nan")
    for epoch_losses in test_ll
]

epochs = list(range(1, len(train_means) + 1))


# ---- 3) Plot the clean curve ----
plt.figure(figsize=(8, 5))

plt.plot(epochs, train_means, marker="o", label="Train (mean per epoch)")
plt.plot(epochs, test_means, marker="s", label="Test (mean per epoch)")

plt.title("Mean Train & Test Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(dir_name, "loss_means_epoch_kv.png"), dpi=200)
plt.show()
