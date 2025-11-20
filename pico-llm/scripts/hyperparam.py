import json
import os
import matplotlib.pyplot as plt

embed_sizes = [512, 1024]
base_dir = "trained_outputs"

plt.figure(figsize=(8, 5))
plt.title("Mean Training Loss (First 3 Epochs) for Different Embedding Sizes")
plt.xlabel("Epoch")
plt.ylabel("Mean Train Loss")

max_epochs_to_plot = 3  # stop after epoch 3 since thats the min number of epochs run in both

for emb in embed_sizes:
    dir_name = f"{base_dir}/outputs_wiki_{emb}"
    loss_log_path = os.path.join(dir_name, "loss_logs.json")
    if not os.path.exists(loss_log_path):
        print(f"⚠️ Skipping {emb} — no loss_logs.json found.")
        continue

    with open(loss_log_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    model_name = list(logs.keys())[0]
    data = logs[model_name]
    train_ll = data["train"]

    # Compute mean loss per epoch
    epoch_means = [sum(epoch_losses)/len(epoch_losses) for epoch_losses in train_ll if len(epoch_losses) > 0]

    # Keep only first N epochs
    epoch_means = epoch_means[:max_epochs_to_plot]
    epochs = list(range(1, len(epoch_means) + 1))

    plt.plot(epochs, epoch_means, marker="o", label=f"embed={emb}")

plt.xticks(range(1, max_epochs_to_plot + 1))
plt.grid(True, alpha=0.3)
plt.legend(title="Embedding Size")
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "compare_embed_loss_first3epochs.png"), dpi=200)
plt.show()
