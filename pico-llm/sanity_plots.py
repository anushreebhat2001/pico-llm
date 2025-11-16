import json
import os
import matplotlib.pyplot as plt

dir_name = "outputs_embedding_tiny"
LOSS_LOG_PATH = os.path.join(dir_name, "loss_logs.json")

with open(LOSS_LOG_PATH, "r", encoding="utf-8") as f:
    logs = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
axes = axes.flatten()
model_names = ["kgram_mlp_seq", "lstm_seq", "kvcache_transformer"]

fig.suptitle("Train loss per step + test loss per epoch", fontsize=14)

for ax, model_name in zip(axes, model_names):
    data = logs[model_name]
    train_ll = data["train"]
    test_ll  = data["test"]

    # Flatten train
    flat_train = []
    step_indices = []
    step_counter = 0
    epoch_start_indices = []

    for epoch_idx, epoch_losses in enumerate(train_ll, start=1):
        epoch_start_indices.append(step_counter)
        for loss in epoch_losses:
            flat_train.append(loss)
            step_indices.append(step_counter)
            step_counter += 1

    ax.plot(step_indices, flat_train, label="train (per step)")

    # Test: show epoch-mean test loss as points at epoch starts
    test_means = [sum(e)/len(e) if e else float("nan") for e in test_ll]
    ax.plot(epoch_start_indices, test_means, "rx--", label="test (per epoch)")

    ax.set_title(model_name)
    ax.set_xlabel("Global step per epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.92])
plt.savefig(dir_name+".png", dpi=200)
plt.show()
