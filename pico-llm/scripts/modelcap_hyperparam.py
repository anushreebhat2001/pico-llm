import json, os, matplotlib.pyplot as plt

configs = {
    "TinyStories (16H,8B)": "trained_outputs/outputs_tiny_overfit/loss_logs.json",
    "Wiki (16H,8B)": "trained_outputs/outputs_wiki_512/loss_logs.json"
}

plt.figure(figsize=(8,5))
plt.title("Overfitting Comparison: TinyStories vs Wiki (16 Heads, 8 Blocks)")
plt.xlabel("Epoch")
plt.ylabel("Loss")

for name, path in configs.items():
    if not os.path.exists(path):
        print(f"Missing {path}")
        continue

    with open(path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    model_name = list(logs.keys())[0]
    data = logs[model_name]
    train_ll, test_ll = data["train"], data["test"]

    train_means = [sum(e)/len(e) for e in train_ll if e]
    test_means  = [sum(e)/len(e) for e in test_ll  if e]
    epochs = range(1, min(len(train_means), len(test_means)) + 1)

    plt.plot(epochs, train_means, marker="o", label=f"{name} – train")
    plt.plot(epochs, test_means, marker="x", linestyle="--", label=f"{name} – test")

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("trained_outputs/overfitting_tiny_vs_wiki_16H8B.png", dpi=200)
plt.show()
