# starter code by matus & o1-pro
import argparse
import time
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import json

from datasets import load_dataset
import tiktoken

################################################################################
# 1. Command-line arg parsing
################################################################################

def _parse_int_list_csv(s: str):
    if s is None or len(s.strip()) == 0:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_args():
    parser = argparse.ArgumentParser(
        description="pico-llm: models + timing sweep for KV-cache softmax vs DeltaKet."
    )

    # Original args kept (even if you won't use training)
    parser.add_argument("--input_files", nargs="*", default=None,
                        help="Optional list of text files to mix in as data sources.")
    parser.add_argument("--tinystories_weight", type=float, default=0.5,
                        help="Probability of sampling from TinyStories if present.")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None)
    parser.add_argument("--num_inner_mlp_layers", type=int, default=1)
    parser.add_argument("--monosemantic_enabled", action="store_true")
    parser.set_defaults(monosemantic_enabled=False)

    parser.add_argument("--kgram_k", type=int, default=3)
    parser.add_argument("--kgram_chunk_size", type=int, default=1)

    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--embed_size", type=int, default=1024)
    parser.add_argument("--prompt", type=str, default="Once upon a")

    parser.add_argument("--device_id", type=str, default="cuda:0")
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="outputs")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--use_position_emb", action="store_true")
    parser.set_defaults(use_position_emb=False)

    parser.add_argument("--use_post_norm", action="store_true")
    parser.set_defaults(use_post_norm=False)

    # =========================
    # NEW: timing sweep switches
    # =========================
    parser.add_argument("--run_timing_sweep", action="store_true",
                        help="If set, run timing sweep and exit (no training).")
    parser.set_defaults(run_timing_sweep=False)

    parser.add_argument("--transformer_attention", type=str, default="softmax",
                        choices=["softmax", "linear"],
                        help="softmax = standard attention with KV-cache. linear = DeltaKet attention with state-cache.")

    parser.add_argument("--sweep_seq_lens", type=str, default="64,128,256,512,1024",
                        help="Comma-separated list of context lengths to benchmark.")

    parser.add_argument("--sweep_warmup_steps", type=int, default=10,
                        help="Warmup decode steps per sequence length.")
    parser.add_argument("--sweep_measured_steps", type=int, default=30,
                        help="Measured decode steps per sequence length.")

    # Model shape knobs for sweep (so you can keep embed_size=1024 but adjust heads/blocks if needed)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=6)

    # DeltaKet stability
    parser.add_argument("--deltaket_eps", type=float, default=1e-6)

    return parser.parse_args()

################################################################################
# 2. Data handling (kept)
################################################################################

class MixedSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, tinystories_seqs, other_seqs, p_tiny: float):
        super().__init__()
        self.tinystories_seqs = tinystories_seqs
        self.other_seqs = other_seqs
        self.p_tiny = p_tiny

        self.has_tinystories = (len(self.tinystories_seqs) > 0)
        self.has_other = (len(self.other_seqs) > 0)

        self.total_length = len(self.tinystories_seqs) + len(self.other_seqs)
        if self.total_length == 0:
            raise ValueError("No data found! Both TinyStories and other sets are empty.")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        r = random.random()
        if self.has_tinystories and self.has_other:
            if r < self.p_tiny:
                i = random.randint(0, len(self.tinystories_seqs) - 1)
                seq = self.tinystories_seqs[i]
            else:
                i = random.randint(0, len(self.other_seqs) - 1)
                seq = self.other_seqs[i]
        elif self.has_tinystories:
            i = random.randint(0, len(self.tinystories_seqs) - 1)
            seq = self.tinystories_seqs[i]
        else:
            i = random.randint(0, len(self.other_seqs) - 1)
            seq = self.other_seqs[i]

        return torch.tensor(seq, dtype=torch.long)

def seq_collate_fn(batch):
    max_len = max(len(seq) for seq in batch)
    batch_size = len(batch)

    padded = torch.zeros(max_len, batch_size, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded[:seq_len, i] = seq

    return padded

################################################################################
# 3. Loss (kept)
################################################################################

def compute_next_token_loss(logits, tokens):
    seq_len, batch_size, vocab_size = logits.shape
    if seq_len < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    preds = logits[:-1, :, :]
    gold = tokens[1:, :]

    preds = preds.reshape(-1, vocab_size)
    gold = gold.reshape(-1)
    return F.cross_entropy(preds, gold)

################################################################################
# 4. (Kept) KGram / LSTM
################################################################################

class KGramMLPSeqModel(nn.Module):
    def __init__(self, vocab_size, k=3, embed_size=1024, num_inner_layers=1, chunk_size=1):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size

        self.embedding = nn.Embedding(vocab_size, embed_size)

        in_dim = self.k * self.embed_size
        hidden_dim = self.embed_size

        layers = []
        if self.num_inner_layers <= 0:
            layers.append(nn.Linear(in_dim, self.vocab_size))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            for _ in range(self.num_inner_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.SiLU())
            layers.append(nn.Linear(hidden_dim, self.vocab_size))

        self.net = nn.Sequential(*layers)

    def forward(self, tokens_seq):
        seq_len, batch_size = tokens_seq.shape
        device = tokens_seq.device
        outputs = []

        start = 0
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            block_outputs = []
            for t in range(start, end):
                batch_logits = []
                for b in range(batch_size):
                    if t < self.k:
                        needed = self.k - t
                        context_ids = [0] * needed + tokens_seq[:t, b].tolist()
                    else:
                        context_ids = tokens_seq[t-self.k:t, b].tolist()

                    context_ids_tensor = torch.tensor(context_ids, dtype=torch.long, device=device)
                    context_emb = self.embedding(context_ids_tensor)
                    context_flat = context_emb.view(1, -1)
                    logits_b = self.net(context_flat)
                    batch_logits.append(logits_b)

                block_outputs.append(torch.cat(batch_logits, dim=0).unsqueeze(0))

            block_outputs = torch.cat(block_outputs, dim=0)
            outputs.append(block_outputs)
            start = end

        outputs = torch.cat(outputs, dim=0)
        return outputs

class LSTMSeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size=1024, hidden_size=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=False)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens_seq):
        emb = self.embedding(tokens_seq)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(emb)
        logits = self.linear(out)
        return logits

################################################################################
# 5. Transformer
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
# Softmax attention (original)
# ---------------------------
class MultiHeadSelfAttentionSoftmax(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
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

# ---------------------------------------------
# DeltaKet attention (linear) with state-cache
# cache per layer: (S, Z)
#   S: (B, H, D, D)
#   Z: (B, H, D)
# ---------------------------------------------
class MultiHeadSelfAttentionDeltaKet(nn.Module):
    def __init__(self, d_model, n_heads, eps=1e-6):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.eps = eps

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _phi(self, x):
        # positive feature map
        return F.elu(x) + 1.0

    def forward(self, x, mask=None, past_k=None, past_v=None):
        # mask ignored; causality is by recurrence
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
            kt = k[:, :, t, :]  # (B,H,D)
            vt = v[:, :, t, :]  # (B,H,D)

            # predict v_hat from current memory
            num_hat = torch.einsum("bhd,bhde->bhe", kt, S)                 # (B,H,D)
            den_hat = torch.einsum("bhd,bhd->bh", kt, Z).unsqueeze(-1)     # (B,H,1)
            v_hat = num_hat / (den_hat + self.eps)

            dv = vt - v_hat
            S = S + kt.unsqueeze(-1) * dv.unsqueeze(-2)                    # (B,H,D,D)
            Z = Z + kt                                                     # (B,H,D)

            qt = q[:, :, t, :]
            num = torch.einsum("bhd,bhde->bhe", qt, S)
            den = torch.einsum("bhd,bhd->bh", qt, Z).unsqueeze(-1)
            out[:, :, t, :] = num / (den + self.eps)

        attn_output = out.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.out_proj(attn_output)

        attn_weights = torch.empty(0, device=device)  # placeholder
        return attn_output, attn_weights, S, Z

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4.0, use_post_norm=False, attention_impl="softmax", deltaket_eps=1e-6):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.mlp_norm = RMSNorm(d_model)
        self.use_post_norm = use_post_norm

        if attention_impl == "softmax":
            self.attn = MultiHeadSelfAttentionSoftmax(d_model, n_heads)
        elif attention_impl == "linear":
            self.attn = MultiHeadSelfAttentionDeltaKet(d_model, n_heads, eps=deltaket_eps)
        else:
            raise ValueError(f"Unknown attention_impl={attention_impl}")

        hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x, mask=None, collect_attn=False, attn_list=None, act_list=None, past_k=None, past_v=None):
        if self.use_post_norm:
            attn_out, attn_weights, new_k, new_v = self.attn(x, mask=mask, past_k=past_k, past_v=past_v)
            x = self.attn_norm(x + attn_out)

            mlp_out = self.mlp(x)
            x = self.mlp_norm(x + mlp_out)
        else:
            h = self.attn_norm(x)
            attn_out, attn_weights, new_k, new_v = self.attn(h, mask=mask, past_k=past_k, past_v=past_v)
            x = x + attn_out

            m = self.mlp_norm(x)
            mlp_out = self.mlp(m)
            x = x + mlp_out

        if collect_attn and (attn_list is not None) and (act_list is not None):
            # For DeltaKet, attn_weights is an empty placeholder; still safe.
            attn_list.append(attn_weights.detach().cpu() if attn_weights.numel() > 0 else attn_weights.cpu())
            act_list.append(mlp_out.detach().cpu())

        return x, new_k, new_v

class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size=50257,
        d_model=1024,
        n_heads=2,
        n_blocks=4,
        block_size=1024,
        use_position_emb=False,
        use_post_norm=False,
        attention_impl="softmax",
        deltaket_eps=1e-6,
    ):
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
        self.pos_emb = nn.Embedding(block_size, d_model) if self.use_position_emb else None

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                mlp_ratio=4.0,
                use_post_norm=use_post_norm,
                attention_impl=attention_impl,
                deltaket_eps=deltaket_eps,
            ) for _ in range(n_blocks)
        ])

        self.final_norm = RMSNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

        # Softmax-only causal mask buffer
        mask = torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("causal_mask", mask, persistent=False)

        self.attention_matrices = []
        self.activation_outputs = []

    def forward(self, tokens_seq, collect_attn=False, past_kv=None, return_kv=False, past_len: int = 0):
        """
        tokens_seq: (seq_len, batch)
        past_kv:
          - softmax: list[(K,V)] where K,V are (B,H,past_T,D)
          - linear : list[(S,Z)] where S is (B,H,D,D) and Z is (B,H,D)
        past_len:
          - softmax: inferred from K.size(2)
          - linear : must be passed/maintained externally for pos_emb correctness
        """
        seq_len, batch_size = tokens_seq.shape
        device = tokens_seq.device

        # determine past_len for softmax from cached K
        if self.attention_impl == "softmax":
            if past_kv is not None and len(past_kv) > 0 and past_kv[0][0] is not None:
                past_len = past_kv[0][0].size(2)
            else:
                past_len = 0

        # positions (only if pos_emb)
        positions = None
        if self.pos_emb is not None:
            positions = torch.arange(past_len, past_len + seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_emb(tokens_seq.t())  # (B, T, C)
        if self.pos_emb is not None:
            # clamp positions in case someone passes seq_len > block_size
            pos_clamped = torch.clamp(positions, 0, self.block_size - 1)
            x = x + self.pos_emb(pos_clamped)

        if collect_attn:
            self.attention_matrices = []
            self.activation_outputs = []

        # mask used only for softmax attention
        mask = None
        if self.attention_impl == "softmax":
            total_k = past_len + seq_len
            mask = self.causal_mask[:, :, :seq_len, :total_k]

        new_kv = []
        for i, block in enumerate(self.blocks):
            past_k, past_v = (past_kv[i] if past_kv is not None else (None, None))
            x, k, v = block(
                x,
                mask=mask,
                collect_attn=collect_attn,
                attn_list=self.attention_matrices,
                act_list=self.activation_outputs,
                past_k=past_k,
                past_v=past_v,
            )
            new_kv.append((k, v))

        x = self.final_norm(x)
        logits = self.unembed(x)            # (B, T, V)
        logits = logits.transpose(0, 1)     # (T, B, V)

        new_past_len = past_len + seq_len

        if return_kv:
            return logits, new_kv, new_past_len
        else:
            return logits

################################################################################
# 6. Generation (kept; updated to support linear past_len)
################################################################################

def nucleus_sampling(logits, p=0.95):
    probs = torch.softmax(logits, dim=-1)

    if p >= 1.0:
        idx = torch.multinomial(probs, num_samples=1)
        return idx.item()

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    k = torch.searchsorted(cum_probs, torch.tensor(p, device=logits.device)).item() + 1
    k = max(1, min(k, sorted_probs.size(0)))

    truncated_probs = sorted_probs[:k]
    truncated_indices = sorted_indices[:k]
    truncated_probs = truncated_probs / truncated_probs.sum()

    sampled_idx = torch.multinomial(truncated_probs, num_samples=1).item()
    chosen_token = truncated_indices[sampled_idx].item()
    return chosen_token

def generate_text(model, enc, init_text, max_new_tokens=20, device="cpu", top_p=None):
    was_training = model.training
    model.eval()

    with torch.no_grad():
        context_tokens = enc.encode(init_text)

        past_kv = None
        past_len = 0

        for _ in range(max_new_tokens):
            if isinstance(model, TransformerModel):
                if past_kv is None:
                    seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)
                else:
                    seq_tensor = torch.tensor([context_tokens[-1]], dtype=torch.long, device=device).unsqueeze(1)

                logits_seq, past_kv, past_len = model(seq_tensor, past_kv=past_kv, return_kv=True, past_len=past_len)
                next_logits = logits_seq[-1, 0, :]
            else:
                seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)
                logits_seq = model(seq_tensor)
                next_logits = logits_seq[-1, 0, :]

            chosen = torch.argmax(next_logits).item() if top_p is None else nucleus_sampling(next_logits, p=top_p)
            context_tokens.append(chosen)

    model.train(was_training)
    return enc.decode(context_tokens)

################################################################################
# 7. Timing sweep
################################################################################

def _sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def run_timing_sweep(args, device, vocab_size):
    """
    Measures decode-time per token as a function of context length.

    Setup per seq_len L:
      1) "prefill": run model on random tokens of length L, get past_kv
      2) "decode": repeatedly feed 1 token using cache, measure ms/token

    For softmax KV-cache:
      - cache grows with L + decode steps, so decode time increases with L
    For DeltaKet:
      - cache is fixed-size state, decode time ~ constant with L
    """
    os.makedirs(args.output_dir, exist_ok=True)

    seq_lens = _parse_int_list_csv(args.sweep_seq_lens)
    warmup = args.sweep_warmup_steps
    measured = args.sweep_measured_steps

    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=args.embed_size,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        block_size=max(args.block_size, max(seq_lens) + measured + 8),
        use_position_emb=args.use_position_emb,
        use_post_norm=args.use_post_norm,
        attention_impl=args.transformer_attention,
        deltaket_eps=args.deltaket_eps,
    ).to(device)

    model.eval()

    results = {
        "attention": args.transformer_attention,
        "use_position_emb": args.use_position_emb,
        "embed_size": args.embed_size,
        "n_heads": args.n_heads,
        "n_blocks": args.n_blocks,
        "warmup_steps": warmup,
        "measured_steps": measured,
        "seq_lens": seq_lens,
        "rows": []
    }

    print("\n=== TIMING SWEEP ===")
    print(f"attention={args.transformer_attention}, device={device}, pos_emb={args.use_position_emb}")
    print("Columns: L, prefill_ms, decode_ms_per_token, cache_type")

    for L in seq_lens:
        # random context tokens
        ctx = torch.randint(low=0, high=vocab_size, size=(L, 1), dtype=torch.long, device=device)
        past_kv = None
        past_len = 0

        # prefill timing
        _sync_if_cuda(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            _, past_kv, past_len = model(ctx, past_kv=past_kv, return_kv=True, past_len=past_len)
        _sync_if_cuda(device)
        prefill_ms = (time.perf_counter() - t0) * 1000.0

        # warmup decode
        for _ in range(warmup):
            tok = torch.randint(low=0, high=vocab_size, size=(1, 1), dtype=torch.long, device=device)
            with torch.no_grad():
                _, past_kv, past_len = model(tok, past_kv=past_kv, return_kv=True, past_len=past_len)

        # measured decode
        _sync_if_cuda(device)
        t1 = time.perf_counter()
        for _ in range(measured):
            tok = torch.randint(low=0, high=vocab_size, size=(1, 1), dtype=torch.long, device=device)
            with torch.no_grad():
                _, past_kv, past_len = model(tok, past_kv=past_kv, return_kv=True, past_len=past_len)
        _sync_if_cuda(device)
        dt = time.perf_counter() - t1
        decode_ms_per_tok = (dt / measured) * 1000.0

        row = {
            "L": L,
            "prefill_ms": prefill_ms,
            "decode_ms_per_token": decode_ms_per_tok,
            "cache": "KV-history" if args.transformer_attention == "softmax" else "DeltaKet-state(S,Z)"
        }
        results["rows"].append(row)

        print(f"{L:5d} | {prefill_ms:10.3f} | {decode_ms_per_tok:16.6f} | {row['cache']}")

    # Save json
    out_json = os.path.join(args.output_dir, f"timing_{args.transformer_attention}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved timing JSON to: {out_json}")

    # Plot
    try:
        import matplotlib.pyplot as plt

        xs = [r["L"] for r in results["rows"]]
        ys_decode = [r["decode_ms_per_token"] for r in results["rows"]]
        ys_prefill = [r["prefill_ms"] for r in results["rows"]]

        plt.figure()
        plt.plot(xs, ys_decode, marker="o")
        plt.xscale("log", base=2)
        plt.xlabel("Context length L (log2 scale)")
        plt.ylabel("Decode time (ms/token)")
        plt.title(f"Decode scaling: {args.transformer_attention} (KV-cache vs state-cache)")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)

        out_png = os.path.join(args.output_dir, f"timing_{args.transformer_attention}.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"Saved plot to: {out_png}")

        # Also plot prefill
        plt.figure()
        plt.plot(xs, ys_prefill, marker="o")
        plt.xscale("log", base=2)
        plt.xlabel("Context length L (log2 scale)")
        plt.ylabel("Prefill time (ms)")
        plt.title(f"Prefill time vs L: {args.transformer_attention}")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        out_png2 = os.path.join(args.output_dir, f"prefill_{args.transformer_attention}.png")
        plt.tight_layout()
        plt.savefig(out_png2, dpi=200)
        plt.close()
        print(f"Saved prefill plot to: {out_png2}")

    except Exception as e:
        print(f"Plotting failed (matplotlib issue?): {e}")

################################################################################
# 8. Main
################################################################################

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # device
    requested_device_id = args.device_id
    if requested_device_id.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device_id}' but CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device_id)

    # tokenizer/vocab
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab

    if args.run_timing_sweep:
        run_timing_sweep(args, device=device, vocab_size=vocab_size)
        return

    # If you accidentally run without --run_timing_sweep:
    print("You didn't pass --run_timing_sweep. This file is set up for benchmarking.")
    print("Example:")
    print("  python pico_llm.py --run_timing_sweep --transformer_attention softmax --use_position_emb --sweep_seq_lens 64,128,256,512,1024")
    print("  python pico_llm.py --run_timing_sweep --transformer_attention linear  --use_position_emb --sweep_seq_lens 64,128,256,512,1024")

if __name__ == "__main__":
    main()
