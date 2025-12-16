import os, sys, argparse, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from datasets import load_dataset

# Import your classes (adjust path if needed)
sys.path.append("../")
from pico_llm import TransformerModel, LSTMSeqModel  # KGram omitted intentionally

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pretty_tok(enc, tid: int):
    s = enc.decode([tid])
    return s.replace("\n", "\\n")

def decode_window(enc, toks, pos, window=25):
    """
    Decode a window around pos and HIGHLIGHT the exact token at pos as <<<token>>>.
    """
    lo = max(0, pos - window)
    hi = min(len(toks), pos + window + 1)

    pre = enc.decode(toks[lo:pos])
    mid = enc.decode([toks[pos]])
    post = enc.decode(toks[pos + 1:hi])

    s = f"{pre}<<<{mid}>>>{post}"
    return s.replace("\n", "\\n")

def infer_transformer_config_from_state_dict(sd):
    # token_emb.weight: (V, d_model)
    vocab_size, d_model = sd["token_emb.weight"].shape
    use_pos = ("pos_emb.weight" in sd)
    block_size = int(sd["pos_emb.weight"].shape[0]) if use_pos else 1024

    # infer n_blocks by scanning keys
    block_ids = []
    for k in sd.keys():
        if k.startswith("blocks."):
            try:
                block_ids.append(int(k.split(".")[1]))
            except:
                pass
    n_blocks = (max(block_ids) + 1) if block_ids else 0
    return vocab_size, d_model, n_blocks, block_size, use_pos

def maybe_infer_n_heads_from_attn(attn_pt_path):
    """
    If you saved attention matrices via your training script, we can infer heads from that.
    attention_matrices is a list of tensors of shape (B,H,T,T).
    """
    if attn_pt_path is None or not os.path.exists(attn_pt_path):
        return None
    mats = torch.load(attn_pt_path, map_location="cpu")
    if isinstance(mats, list) and len(mats) > 0:
        return int(mats[0].shape[1])
    return None

# ----------------------------
# Sparse Autoencoder
# ----------------------------
class SparseAutoencoder(nn.Module):
    """
    x in R^d  ->  z in R^m (sparse)  ->  x_hat in R^d
    loss = MSE(x_hat, x) + l1_coeff * mean(|z|)
    """
    def __init__(self, d_in, d_hidden, l1_coeff=1e-3):
        super().__init__()
        self.enc = nn.Linear(d_in, d_hidden)
        self.dec = nn.Linear(d_hidden, d_in)
        self.l1_coeff = l1_coeff

    def forward(self, x):
        z = F.relu(self.enc(x))
        x_hat = self.dec(z)
        mse = F.mse_loss(x_hat, x)
        l1 = z.abs().mean()
        loss = mse + self.l1_coeff * l1
        return loss, mse.detach(), l1.detach(), z

def train_sae(X, d_hidden, steps, batch_size, lr, l1_coeff, device):
    sae = SparseAutoencoder(d_in=X.size(1), d_hidden=d_hidden, l1_coeff=l1_coeff).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    Xd = X.to(device)
    N = Xd.size(0)

    sae.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, N, (batch_size,), device=device)
        xb = Xd[idx]
        loss, mse, l1, _ = sae(xb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 200 == 0:
            print(f"[SAE] step {step:4d}: loss={loss.item():.4f} mse={mse.item():.4f} l1={l1.item():.4f}")

    return sae

# ----------------------------
# Text loading
# ----------------------------
def load_text_corpus(text_source, num_texts,
                     dataset_name="wikitext", dataset_config="wikitext-103-v1", split="train[:5000]",
                     text_file=None):
    """
    Returns list[str] length ~= num_texts.

    text_source:
      - "wikitext": uses datasets non-streaming
      - "tinystories": uses streaming so you DON'T download everything
      - "file": reads lines from a file
    """
    out = []

    if text_source == "tinystories":
        # TinyStories dataset on HF
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        for ex in ds:
            txt = (ex.get("text", "") or "").strip()
            if txt:
                out.append(txt)
            if len(out) >= num_texts:
                break
        return out

    if text_source == "wikitext":
        ds = load_dataset(dataset_name, dataset_config, split=split)
        ds = ds.filter(lambda x: len(x["text"].strip()) > 0)
        ds = ds.select(range(min(num_texts, len(ds))))
        return [x["text"] for x in ds]

    if text_source == "file":
        if text_file is None:
            raise ValueError("--text_source file requires --text_file PATH")
        with open(text_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(line)
                if len(out) >= num_texts:
                    break
        return out

    raise ValueError(f"Unknown text_source={text_source}")

def tokenize_texts(enc, texts, max_len):
    out = []
    for t in texts:
        toks = enc.encode(t)[:max_len]
        if len(toks) >= 8:
            out.append(toks)
    return out

# ----------------------------
# Activation collection
# ----------------------------
def collect_transformer_mlp_out(model, token_seqs, layer_idx, device, positions_per_seq=32):
    """
    Uses your built-in logging: model(seq, collect_attn=True) fills model.activation_outputs
    and we read activation_outputs[layer_idx] which is mlp_out with shape (B,T,d_model).
    Returns:
      X: (N, d_model)
      meta: list of (tokens_list, pos)
    """
    model.eval()
    X_list = []
    meta = []

    with torch.no_grad():
        for toks in token_seqs:
            seq = torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(1)  # (T,1)
            _ = model(seq, collect_attn=True)

            # model.activation_outputs[layer_idx] is stored on CPU already by your model code
            acts = model.activation_outputs[layer_idx]  # (B,T,d_model) on CPU
            acts = acts.squeeze(0)                      # (T,d_model)

            T = acts.size(0)
            k = min(positions_per_seq, T)
            pos_sample = random.sample(range(T), k=k)

            for p in pos_sample:
                X_list.append(acts[p])
                meta.append((toks, p))

    X = torch.stack(X_list, dim=0)  # (N,d_model)
    return X, meta

def collect_lstm_hidden(model, token_seqs, device, positions_per_seq=32):
    """
    Collect LSTM outputs (hidden states) from model.lstm output, shape (T,1,H).
    Returns:
      X: (N,H)
      meta: list of (tokens_list, pos)
    """
    model.eval()
    X_list = []
    meta = []

    with torch.no_grad():
        for toks in token_seqs:
            seq = torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(1)  # (T,1)

            emb = model.embedding(seq)               # (T,1,E)
            model.lstm.flatten_parameters()
            out, _ = model.lstm(emb)                 # (T,1,H)
            out = out.squeeze(1).detach().cpu()      # (T,H) on CPU

            T = out.size(0)
            k = min(positions_per_seq, T)
            pos_sample = random.sample(range(T), k=k)

            for p in pos_sample:
                X_list.append(out[p])
                meta.append((toks, p))

    X = torch.stack(X_list, dim=0)
    return X, meta

# ----------------------------
# Feature interpretation
# ----------------------------
def pick_features_to_show(sae, X, num_features=20, device="cpu"):
    """
    Choose “interesting” features by average activation on X.
    """
    sae.eval()
    with torch.no_grad():
        z = F.relu(sae.enc(X.to(device))).cpu()  # (N, d_hidden)
        means = z.mean(dim=0)                    # (d_hidden,)
        _, idxs = torch.topk(means, k=min(num_features, means.numel()))
    return idxs.tolist()

def top_contexts_for_feature(sae, X, meta, feat_idx, enc, device, top_k=10, window=25):
    sae.eval()
    with torch.no_grad():
        z = F.relu(sae.enc(X.to(device))).cpu()
    vals = z[:, feat_idx]
    v, idx = torch.topk(vals, k=min(top_k, vals.numel()))
    out = []
    for score, i in zip(v.tolist(), idx.tolist()):
        toks, pos = meta[i]
        out.append((score, pos, decode_window(enc, toks, pos, window=window)))
    return out

def write_feature_report(path, title, features, contexts_by_feature):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        for feat in features:
            f.write(f"## Feature {feat}\n\n")
            for score, pos, snippet in contexts_by_feature[feat]:
                f.write(f"- **{score:.4f}** @pos {pos}: `{snippet}`\n")
            f.write("\n")

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", choices=["transformer", "lstm"], default="transformer")
    ap.add_argument("--weights", required=True, help="Path to *_final_weights.pt")
    ap.add_argument("--attn_pt", default=None, help="Optional: *_attention_matrices.pt to infer n_heads")
    ap.add_argument("--out_dir", default="interp_outputs")
    ap.add_argument("--seed", type=int, default=0)

    # Text source selection
    ap.add_argument("--text_source", choices=["wikitext", "tinystories", "file"], default="tinystories")
    ap.add_argument("--text_file", default=None, help="Used when --text_source file. One example per line.")

    # Data for activation collection
    ap.add_argument("--dataset", default="wikitext")
    ap.add_argument("--dataset_config", default="wikitext-103-v1")
    ap.add_argument("--split", default="train[:5000]")
    ap.add_argument("--num_texts", type=int, default=2000)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seqs_used", type=int, default=300)
    ap.add_argument("--positions_per_seq", type=int, default=32)

    # Transformer specifics
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=None, help="Only needed if not inferable from attn_pt")
    ap.add_argument("--use_post_norm", action="store_true", help="Match your training setting if you used post-norm")

    # SAE hyperparams
    ap.add_argument("--sae_mult", type=float, default=4.0, help="d_hidden = sae_mult * d_in")
    ap.add_argument("--l1", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--show_features", type=int, default=20)
    ap.add_argument("--top_contexts", type=int, default=10)
    ap.add_argument("--window", type=int, default=25)

    args = ap.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")

    # ---- Load corpus + tokenize ----
    texts = load_text_corpus(
        text_source=args.text_source,
        num_texts=args.num_texts,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        text_file=args.text_file,
    )

    token_seqs = tokenize_texts(enc, texts, max_len=args.max_len)
    random.shuffle(token_seqs)
    token_seqs = token_seqs[:args.seqs_used]
    print(f"[Data] source={args.text_source} | using {len(token_seqs)} sequences (max_len={args.max_len}).")

    # ---- Load model ----
    sd = torch.load(args.weights, map_location=device)

    if args.model_type == "transformer":
        vocab_size, d_model, n_blocks, block_size, use_pos = infer_transformer_config_from_state_dict(sd)

        inferred_heads = maybe_infer_n_heads_from_attn(args.attn_pt)
        n_heads = args.n_heads or inferred_heads or 8
        print(f"[Transformer] inferred d_model={d_model}, n_blocks={n_blocks}, block_size={block_size}, use_pos={use_pos}, n_heads={n_heads}")

        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_blocks=n_blocks,
            block_size=block_size,
            use_position_emb=use_pos,
            use_post_norm=args.use_post_norm,
        ).to(device)
        model.load_state_dict(sd, strict=False)
        model.eval()

        layer_idx = max(0, min(args.layer, n_blocks - 1))
        X, meta = collect_transformer_mlp_out(
            model, token_seqs, layer_idx=layer_idx, device=device, positions_per_seq=args.positions_per_seq
        )
        print(f"[Collect] Transformer layer {layer_idx} mlp_out -> X shape: {tuple(X.shape)}")

    elif args.model_type == "lstm":
        vocab_size = sd["embedding.weight"].shape[0]
        embed_size = sd["embedding.weight"].shape[1]
        hidden_size = sd["lstm.weight_ih_l0"].shape[0] // 4
        print(f"[LSTM] inferred vocab={vocab_size}, embed={embed_size}, hidden={hidden_size}")

        model = LSTMSeqModel(vocab_size=vocab_size, embed_size=embed_size, hidden_size=hidden_size).to(device)
        model.load_state_dict(sd, strict=False)
        model.eval()

        X, meta = collect_lstm_hidden(model, token_seqs, device, positions_per_seq=args.positions_per_seq)
        print(f"[Collect] LSTM hidden states -> X shape: {tuple(X.shape)}")

    else:
        raise ValueError("Unsupported model_type")

    # ---- Train SAE ----
    d_in = X.size(1)
    d_hidden = int(args.sae_mult * d_in)
    print(f"[SAE] Training SAE: d_in={d_in}, d_hidden={d_hidden}, l1={args.l1}, steps={args.steps}")

    sae = train_sae(
        X=X,
        d_hidden=d_hidden,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        l1_coeff=args.l1,
        device=device,
    )

    # ---- Interpret features ----
    features = pick_features_to_show(sae, X, num_features=args.show_features, device=device)
    contexts_by_feature = {}
    for feat in features:
        contexts_by_feature[feat] = top_contexts_for_feature(
            sae, X, meta, feat, enc, device=device, top_k=args.top_contexts, window=args.window
        )

    # ---- Write report ----
    tag = f"{args.model_type}"
    if args.model_type == "transformer":
        tag += f"_layer{args.layer}"
    report_path = os.path.join(args.out_dir, f"sae_report_{tag}.md")
    title = f"SAE Monosemantic Feature Report ({tag})"
    write_feature_report(report_path, title, features, contexts_by_feature)
    print(f"[Done] Wrote report to: {report_path}")

if __name__ == "__main__":
    main()
