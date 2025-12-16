# pico_llm.py
# starter code by matus & o1-pro
# MODS:
#  - timing sweep: KV-cache softmax vs DeltaKet "linear" attention
#  - FIX: robust causal mask sizing for softmax KV-cache (no more 2086 vs 2087)

import argparse
import time
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json

import tiktoken

################################################################################
# Args
################################################################################

def _parse_int_list_csv(s: str):
    if s is None or len(s.strip()) == 0:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_args():
    p = argparse.ArgumentParser("pico-llm: timing sweep softmax KV-cache vs DeltaKet")

    p.add_argument("--device_id", type=str, default="cuda:0")
    p.add_argument("--output_dir", type=str, default="outputs")

    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--embed_size", type=int, default=1024)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_blocks", type=int, default=6)

    p.add_argument("--use_position_emb", action="store_true")
    p.set_defaults(use_position_emb=False)

    p.add_argument("--use_post_norm", action="store_true")
    p.set_defaults(use_post_norm=False)

    p.add_argument("--run_timing_sweep", action="store_true")
    p.set_defaults(run_timing_sweep=False)

    p.add_argument("--transformer_attention", type=str, default="softmax",
                   choices=["softmax", "linear"])

    p.add_argument("--sweep_seq_lens", type=str, default="64,128,256,512,1024")
    p.add_argument("--sweep_warmup_steps", type=int, default=10)
    p.add_argument("--sweep_measured_steps", type=int, default=30)

    p.add_argument("--deltaket_eps", type=float, default=1e-6)

    return p.parse_args()

################################################################################
# Transformer bits
################################################################################

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * norm_x

# ---------------------------
# Softmax attention (KV-history)
# ---------------------------
class MultiHeadSelfAttentionSoftmax(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None, past_k=None, past_v=None):
        B, T, C = x.shape
        H = self.n_heads
        D = self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        if past_k is not None:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights, k, v

# ---------------------------
# DeltaKet attention (state-cache S,Z)
# ---------------------------
class MultiHeadSelfAttentionDeltaKet(nn.Module):
    def __init__(self, d_model, n_heads, eps=1e-6):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.eps = eps

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _phi(self, x):
        return F.elu(x) + 1.0

    def forward(self, x, mask=None, past_k=None, past_v=None):
        # mask ignored (causal by recurrence)
        B, T, C = x.shape
        H = self.n_heads
        D = self.head_dim
        device, dtype = x.device, x.dtype

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        q = self._phi(q)
        k = self._phi(k)

        if past_k is not None and past_v is not None:
            S = past_k  # (B,H,D,D)
            Z = past_v  # (B,H,D)
        else:
            S = torch.zeros((B, H, D, D), device=device, dtype=dtype)
            Z = torch.zeros((B, H, D), device=device, dtype=dtype)

        out = torch.empty((B, H, T, D), device=device, dtype=dtype)

        for t in range(T):
            kt = k[:, :, t, :]
            vt = v[:, :, t, :]

            num_hat = torch.einsum("bhd,bhde->bhe", kt, S)
            den_hat = torch.einsum("bhd,bhd->bh", kt, Z).unsqueeze(-1)
            v_hat = num_hat / (den_hat + self.eps)

            dv = vt - v_hat
            S = S + kt.unsqueeze(-1) * dv.unsqueeze(-2)
            Z = Z + kt

            qt = q[:, :, t, :]
            num = torch.einsum("bhd,bhde->bhe", qt, S)
            den = torch.einsum("bhd,bhd->bh", qt, Z).unsqueeze(-1)
            out[:, :, t, :] = num / (den + self.eps)

        attn_output = out.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.out_proj(attn_output)
        attn_weights = torch.empty(0, device=device)
        return attn_output, attn_weights, S, Z

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, use_post_norm=False, attention_impl="softmax", deltaket_eps=1e-6):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.mlp_norm = RMSNorm(d_model)
        self.use_post_norm = use_post_norm

        if attention_impl == "softmax":
            self.attn = MultiHeadSelfAttentionSoftmax(d_model, n_heads)
        else:
            self.attn = MultiHeadSelfAttentionDeltaKet(d_model, n_heads, eps=deltaket_eps)

        hidden_dim = int(d_model * 4.0)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x, mask=None, past_k=None, past_v=None):
        if self.use_post_norm:
            attn_out, _, new_k, new_v = self.attn(x, mask=mask, past_k=past_k, past_v=past_v)
            x = self.attn_norm(x + attn_out)
            mlp_out = self.mlp(x)
            x = self.mlp_norm(x + mlp_out)
        else:
            h = self.attn_norm(x)
            attn_out, _, new_k, new_v = self.attn(h, mask=mask, past_k=past_k, past_v=past_v)
            x = x + attn_out
            m = self.mlp_norm(x)
            mlp_out = self.mlp(m)
            x = x + mlp_out

        return x, new_k, new_v

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_blocks, block_size,
                 use_position_emb=False, use_post_norm=False,
                 attention_impl="softmax", deltaket_eps=1e-6):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.use_position_emb = use_position_emb
        self.use_post_norm = use_post_norm
        self.attention_impl = attention_impl

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model) if use_position_emb else None

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, use_post_norm=use_post_norm,
                             attention_impl=attention_impl, deltaket_eps=deltaket_eps)
            for _ in range(n_blocks)
        ])

        self.final_norm = RMSNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

        # small default mask buffer; we will dynamically grow if needed
        mask = torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("causal_mask", mask, persistent=False)

    def _get_causal_mask(self, q_len: int, k_len: int, device, dtype):
        """
        Returns a (1,1,q_len,k_len) lower-triangular causal mask.
        Uses buffer if possible; otherwise builds dynamically.
        """
        if k_len <= self.causal_mask.size(-1) and q_len <= self.causal_mask.size(-2):
            return self.causal_mask[:, :, :q_len, :k_len].to(device=device, dtype=dtype)
        # dynamic build
        m = torch.tril(torch.ones((q_len, k_len), device=device, dtype=dtype))
        return m.unsqueeze(0).unsqueeze(0)

    def forward(self, tokens_seq, past_kv=None, return_kv=False, past_len: int = 0):
        """
        tokens_seq: (T, B)
        past_kv:
          softmax: [(K,V)] where K,V: (B,H,past_T,D)
          linear : [(S,Z)] where S:(B,H,D,D), Z:(B,H,D)
        past_len:
          softmax: ignored (we infer from K)
          linear : tracked externally for pos_emb correctness
        """
        T, B = tokens_seq.shape
        device = tokens_seq.device

        # ---- softmax past_len must be inferred from cache (source of truth) ----
        if self.attention_impl == "softmax":
            if past_kv is not None and len(past_kv) > 0 and past_kv[0][0] is not None:
                past_len = past_kv[0][0].size(2)
            else:
                past_len = 0

        # positions for pos_emb
        if self.pos_emb is not None:
            positions = torch.arange(past_len, past_len + T, device=device)
            # prevent out-of-range indexing
            positions = torch.clamp(positions, 0, self.block_size - 1)
            positions = positions.unsqueeze(0).expand(B, T)  # (B,T)
        else:
            positions = None

        x = self.token_emb(tokens_seq.t())  # (B,T,C)
        if self.pos_emb is not None:
            x = x + self.pos_emb(positions)

        mask = None
        if self.attention_impl == "softmax":
            total_k = past_len + T
            mask = self._get_causal_mask(q_len=T, k_len=total_k, device=device, dtype=x.dtype)

        new_kv = []
        for i, block in enumerate(self.blocks):
            pk, pv = (past_kv[i] if past_kv is not None else (None, None))
            x, nk, nv = block(x, mask=mask, past_k=pk, past_v=pv)
            new_kv.append((nk, nv))

        x = self.final_norm(x)
        logits = self.unembed(x).transpose(0, 1)  # (T,B,V)

        new_past_len = past_len + T
        if return_kv:
            return logits, new_kv, new_past_len
        return logits

################################################################################
# Timing sweep
################################################################################

def _sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def run_timing_sweep(args, device, vocab_size):
    os.makedirs(args.output_dir, exist_ok=True)

    seq_lens = _parse_int_list_csv(args.sweep_seq_lens)
    warmup = args.sweep_warmup_steps
    measured = args.sweep_measured_steps

    # ensure internal block_size big enough for pos_emb indexing
    internal_block_size = max(args.block_size, max(seq_lens) + warmup + measured + 16)

    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=args.embed_size,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        block_size=internal_block_size,
        use_position_emb=args.use_position_emb,
        use_post_norm=args.use_post_norm,
        attention_impl=args.transformer_attention,
        deltaket_eps=args.deltaket_eps,
    ).to(device).eval()

    print("\n=== TIMING SWEEP ===")
    print(f"attention={args.transformer_attention}, device={device}, pos_emb={args.use_position_emb}")
    print("Columns: L, prefill_ms, decode_ms_per_token, cache_type")

    rows = []

    for L in seq_lens:
        ctx = torch.randint(0, vocab_size, (L, 1), device=device, dtype=torch.long)
        past_kv = None
        past_len = 0

        # Prefill
        _sync_if_cuda(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            _, past_kv, past_len = model(ctx, past_kv=past_kv, return_kv=True, past_len=past_len)
        _sync_if_cuda(device)
        prefill_ms = (time.perf_counter() - t0) * 1000.0

        # Warmup decode
        for _ in range(warmup):
            tok = torch.randint(0, vocab_size, (1, 1), device=device, dtype=torch.long)
            with torch.no_grad():
                _, past_kv, past_len = model(tok, past_kv=past_kv, return_kv=True, past_len=past_len)

        # Measured decode
        _sync_if_cuda(device)
        t1 = time.perf_counter()
        for _ in range(measured):
            tok = torch.randint(0, vocab_size, (1, 1), device=device, dtype=torch.long)
            with torch.no_grad():
                _, past_kv, past_len = model(tok, past_kv=past_kv, return_kv=True, past_len=past_len)
        _sync_if_cuda(device)
        dt = time.perf_counter() - t1

        decode_ms_per_tok = (dt / measured) * 1000.0
        cache_type = "KV-history" if args.transformer_attention == "softmax" else "DeltaKet-state(S,Z)"

        print(f"{L:5d} | {prefill_ms:10.3f} | {decode_ms_per_tok:16.6f} | {cache_type}")

        rows.append({
            "L": L,
            "prefill_ms": prefill_ms,
            "decode_ms_per_token": decode_ms_per_tok,
            "cache": cache_type
        })

    out = {
        "attention": args.transformer_attention,
        "use_position_emb": args.use_position_emb,
        "embed_size": args.embed_size,
        "n_heads": args.n_heads,
        "n_blocks": args.n_blocks,
        "warmup_steps": warmup,
        "measured_steps": measured,
        "seq_lens": seq_lens,
        "rows": rows
    }

    out_json = os.path.join(args.output_dir, f"timing_{args.transformer_attention}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved timing JSON to: {out_json}")

    # Plot
    try:
        import matplotlib.pyplot as plt

        xs = [r["L"] for r in rows]
        ys = [r["decode_ms_per_token"] for r in rows]

        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xscale("log", base=2)
        plt.xlabel("Context length L (log2)")
        plt.ylabel("Decode time (ms/token)")
        plt.title(f"Decode scaling ({args.transformer_attention})")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        out_png = os.path.join(args.output_dir, f"timing_{args.transformer_attention}.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"Saved plot to: {out_png}")

    except Exception as e:
        print(f"Plotting failed: {e}")

################################################################################
# Main
################################################################################

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device_id.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested {args.device_id} but CUDA not available; using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device_id)

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab

    if args.run_timing_sweep:
        run_timing_sweep(args, device=device, vocab_size=vocab_size)
        return

    print("Pass --run_timing_sweep to run the benchmark.")

if __name__ == "__main__":
    main()
