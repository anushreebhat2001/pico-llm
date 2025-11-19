import os
# exam_interpretability.py
import torch
import matplotlib.pyplot as plt
import tiktoken

from pico_llm import TransformerModel   # your Transformer class

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Base dir = folder this file lives in
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs_tinystories_full")

WEIGHTS_PATH = os.path.join(OUT_DIR, "kvcache_transformer_final_weights.pt")

# Prompt you want to analyze (<= ~15 tokens is perfect)
PROMPT = "Once upon a time, there was a little girl named Lily who loved to play outside."

enc = tiktoken.get_encoding("gpt2")
prompt_tokens = enc.encode(PROMPT)
prompt_token_strs = [enc.decode([t]) for t in prompt_tokens]
print("Prompt token count:", len(prompt_tokens))
print("Token index mapping:")
for i, tok in enumerate(prompt_token_strs):
    print(f"{i:2d} -> {repr(tok)}")


# ---- LOAD MODEL + WEIGHTS ----
print("Loading Transformer model...")
vocab_size = enc.n_vocab

# These MUST match the config used during training
D_MODEL = 512       # from checkpoint: token_emb weight [50257, 512]
N_HEADS = 8         # from attention tensor shape [1, 8, T, T]
N_BLOCKS = 6        # blocks.0..blocks.5 exist, 6 and 7 are missing
BLOCK_SIZE = 512    # from pos_emb weight [512, 512]

model = TransformerModel(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_blocks=N_BLOCKS,
    block_size=BLOCK_SIZE,
).to(DEVICE)

state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
model.load_state_dict(state)   # should now load cleanly
model.eval()
print("Loaded weights from:", WEIGHTS_PATH)

# ---- RUN MODEL ON THE NEW PROMPT AND COLLECT ATTENTION ----
tokens_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=DEVICE).unsqueeze(1)  # (T, 1)

with torch.no_grad():
    _ = model(tokens_tensor, collect_attn=True)

# Now attention_mats is a list: len = n_blocks, each (B, H, T, T)
attention_mats = model.attention_matrices
print("Num layers with attention:", len(attention_mats))
print("Layer 0 attention shape:", attention_mats[0].shape)  # expect (1, 8, T, T)


def plot_attention_head(block_idx=0, head_idx=0):
    """
    block_idx: which transformer layer (0 .. num_layers-1)
    head_idx: which attention head (0 .. num_heads-1)
    """
    attn = attention_mats[block_idx]      # (B, H, T, T)
    attn = attn[0, head_idx]              # (T, T)

        # Show top 4 keys for a few interesting query positions
    interesting_qs = [0, 5, 9, 11, 13, 16]  # tweak these as you like

    for q in interesting_qs:
        if q >= attn.shape[0]:
            continue
        topk = torch.topk(attn[q], k=4)
        q_tok = prompt_token_strs[q] if q < len(prompt_token_strs) else "<unk>"
        print(f"\nQuery pos {q:2d} ({repr(q_tok)} ) attends to:")
        for score, idx in zip(topk.values, topk.indices):
            idx = idx.item()
            tok = prompt_token_strs[idx] if idx < len(prompt_token_strs) else "<unk>"
            print(f"   key {idx:2d} ({repr(tok)}), weight={score.item():.3f}")


    T = attn.shape[-1]
    plt.figure(figsize=(6, 5))
    plt.imshow(attn.cpu().numpy(), aspect="auto")
    plt.colorbar(label="attention weight")
    plt.xlabel("Key position")
    plt.ylabel("Query position")
    plt.title(f"Layer {block_idx}, Head {head_idx}")
    plt.tight_layout()
    plt.show()

    # Optional: print top-attended tokens for a few query positions
    for q in [0, min(5, T - 1), T - 1]:
        topk = torch.topk(attn[q], k=5)
        token_name = prompt_token_strs[q] if q < len(prompt_token_strs) else "<unk>"
        print(f"\nQuery token {q} ('{token_name}') attends to:")
        for score, idx in zip(topk.values, topk.indices):
            idx = idx.item()
            tok = prompt_token_strs[idx] if idx < len(prompt_token_strs) else "<unk>"
            print(f"  -> pos {idx:2d} '{tok}' (w={score.item():.3f})")


if __name__ == "__main__":
    # Start with one head; then try others once this works.
    plot_attention_head(block_idx=0, head_idx=0)

