import json
import matplotlib.pyplot as plt

def load_points(path):
    with open(path, "r") as f:
        data = json.load(f)
    seq_lens = [p["seq_len"] for p in data["points"]]
    tps = [p["tokens_per_sec"] for p in data["points"]]
    return seq_lens, tps, data["model"]

def main():
    kv_path = "outputs/timing_sweep_kvcache_transformer.json"
    lin_path = "outputs/timing_sweep_linear_transformer.json"

    kv_L, kv_tps, kv_name = load_points(kv_path)

    # Linear may crash early — handle gracefully
    try:
        lin_L, lin_tps, lin_name = load_points(lin_path)
        has_linear = True
    except FileNotFoundError:
        has_linear = False

    plt.figure(figsize=(7, 5))

    plt.plot(
        kv_L, kv_tps,
        marker="o",
        linewidth=2,
        label="Softmax + KV cache"
    )

    if has_linear:
        plt.plot(
            lin_L, lin_tps,
            marker="o",
            linestyle="--",
            linewidth=2,
            label="Linear attention (Mamba-ish)"
        )

    plt.xlabel("Sequence length")
    plt.ylabel("Tokens / second")
    plt.title("Training Throughput vs Sequence Length")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
