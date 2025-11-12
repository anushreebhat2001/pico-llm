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


# We do not import numpy or scikit-learn, so we implement a naive k-means in pure PyTorch.
# If you prefer scikit-learn, you can adapt the code.

from datasets import load_dataset
import tiktoken

################################################################################
# 1. Command-line arg parsing
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple k-gram or sequence-based models on TinyStories and/or custom text files.")
    parser.add_argument("--input_files", nargs="*", default=None,
                        help="Optional list of text files to mix in as data sources. Each line is one example (up to block_size).")
    parser.add_argument("--tinystories_weight", type=float, default=0.5,
                        help="Probability of sampling from TinyStories if present. Default=0.5. (set to 0.0 to skip TinyStories).")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None,
                        help="If set, each epoch ends after this many steps (for quick tests).")
    parser.add_argument("--num_inner_mlp_layers", type=int, default=1,
                        help="Number of (Linear->SiLU) blocks inside the k-gram MLP. Default=1.")
    parser.add_argument("--monosemantic_enabled", action="store_true",
                        help="(DISABLED BY DEFAULT) If set, run the monosemantic analysis.")
    parser.set_defaults(monosemantic_enabled=False)  # disable by default

    # Additional hyperparams to mitigate slow k-gram
    parser.add_argument("--kgram_k", type=int, default=3,
                        help="Sliding window size for k-gram MLP. Smaller can reduce memory usage. Default=3.")
    parser.add_argument("--kgram_chunk_size", type=int, default=1,
                        help="Process k-gram timesteps in micro-batches. Default=1.")

    parser.add_argument("--block_size", type=int, default=1024,
                        help="Maximum sequence length for each example. Default=1024.")

    # New arguments:
    parser.add_argument("--embed_size", type=int, default=1024,
                        help="Dimension of the embedding layer for LSTM, MLP, etc. Default=1024.")
    parser.add_argument("--prompt", type=str, default="Once upon a",
                        help="Prompt used for generation. Default='Once upon a'.")

    # Newly added device argument:
    parser.add_argument("--device_id", type=str, default="cuda:0",
                        help="Torch device identifier (default='cuda:0'). If CUDA is unavailable, fallback to 'cpu'.")
    
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                    help="Directory to store final weights and the zip.")

    args = parser.parse_args()
    return args


################################################################################
# 2. Data handling: entire sequences up to block_size => (seq_len, batch)
################################################################################

class MixedSequenceDataset(torch.utils.data.Dataset):
    """
    We store two lists of entire token sequences:
      - tinystories_seqs
      - other_seqs
    Each sequence is length <= block_size.

    During __getitem__, we randomly pick from one list or the other with probability p_tiny.
    Return that entire sequence as a 1D LongTensor.
    """
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
    """
    batch: list of 1D LongTensors of various lengths [<= block_size].
    1) find max length
    2) pad with zeros
    3) shape => (max_len, batch_size)
    """
    max_len = max(len(seq) for seq in batch)
    batch_size = len(batch)

    padded = torch.zeros(max_len, batch_size, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded[:seq_len, i] = seq

    return padded


################################################################################
# 3. K-gram MLP in a sequence-to-sequence approach
################################################################################

def compute_next_token_loss(logits, tokens):
    """
    logits: (seq_len, batch, vocab_size)
    tokens: (seq_len, batch)
    Next-token prediction => we shift target by 1.
    """
    seq_len, batch_size, vocab_size = logits.shape
    if seq_len < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    preds = logits[:-1, :, :]  # (seq_len-1, batch, vocab_size)
    gold = tokens[1:, :]       # (seq_len-1, batch)

    preds = preds.reshape(-1, vocab_size)
    gold = gold.reshape(-1)
    return F.cross_entropy(preds, gold)

def sample_fast(model, enc, prompt, device, max_new_tokens=20, top_p=None, temperature=1.0, **_):
    """Use KV-cached generation if available; otherwise fall back to slow path."""
    if hasattr(model, "generate_kv"):
        idx = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)  # (1, T0)
        out = model.generate_kv(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        text = enc.decode(out[0].tolist())
        return text, text
    else:
        return generate_text(model, enc, prompt, max_new_tokens=max_new_tokens, device=device, top_p=top_p)

class KGramMLPSeqModel(nn.Module):
    """
    For each position t in [0..seq_len-1], gather the last k tokens => one-hot => MLP => logits.
    Return (seq_len, batch, vocab_size).

    Potentially very large memory usage for big vocab or seq_len. chunk_size helps mitigate overhead.
    """

    def __init__(self, vocab_size, k=3, embed_size=1024, num_inner_layers=1, chunk_size=1):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size

        # NEW: embeddings for token IDs
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)

        # NEW: MLP over concatenated k embeddings (k * embed_size)
        input_dim = self.k * self.embed_size
        hidden = max(256, min(1024, input_dim // 2))  # a bit wider is fine here

        layers = [nn.Linear(input_dim, hidden), nn.SiLU()]
        for _ in range(self.num_inner_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, self.vocab_size)]
        self.net = nn.Sequential(*layers)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        return: (seq_len, batch, vocab_size)
        We'll do a loop over time steps. chunk_size can reduce overhead.
        """
        seq_len, batch_size = tokens_seq.shape
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
                        context_ids = [0]*needed + tokens_seq[:t, b].tolist()
                    else:
                        context_ids = tokens_seq[t-self.k:t, b].tolist()

                    context_ids_t = torch.tensor(context_ids, dtype=torch.long, device=tokens_seq.device)  # (k,)
                    context_emb = self.embedding(context_ids_t)       # (k, embed_size)
                    context_flat = context_emb.flatten().unsqueeze(0) # (1, k*embed_size)

                    logits_b = self.net(context_flat)  # (1, vocab_size)
                    batch_logits.append(logits_b)
                block_outputs.append(torch.cat(batch_logits, dim=0).unsqueeze(0))  # (1, batch, vocab_size)

            block_outputs = torch.cat(block_outputs, dim=0)  # (chunk_size, batch, vocab_size)
            outputs.append(block_outputs)
            start = end

        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, vocab_size)
        return outputs


################################################################################
# 4. LSTM-based seq2seq
################################################################################

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
        """
        tokens_seq: (seq_len, batch)
        => (seq_len, batch, vocab_size)
        """
        emb = self.embedding(tokens_seq)   # (seq_len, batch, embed)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(emb)           # (seq_len, batch, hidden)
        logits = self.linear(out)         # (seq_len, batch, vocab_size)
        return logits


################################################################################
# 5. Our "stub" Transformer with KV-cache 
#    Very slow Python loop for training. Multi-head sums head outputs.
################################################################################

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., C)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * (x * rms)


class MLP(nn.Module):
    def __init__(self, d_model, mult=4.0, dropout=0.0, activation="gelu"):
        super().__init__()
        hidden = int(mult * d_model)
        act = nn.GELU() if activation.lower() == "gelu" else nn.SiLU()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden, bias=True),
            act,
            nn.Linear(hidden, d_model, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CausalSelfAttention(nn.Module):
    """
    KV-cache aware multi-head self-attention.
    Training: pass full x (T,B,C), past_kv=None -> returns (T,B,C), present_kv.
    Cached gen: pass x with T=1 and past_kv=(Kpast,Vpast) -> returns (1,B,C), new (K,V).
    """
    def __init__(self, d_model, n_heads, dropout=0.0, bias=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out = nn.Linear(d_model, d_model, bias=bias)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.register_buffer("tri_mask_cache", None, persistent=False)

    def _split(self, t):  # (T,B,C) -> (B,nH,T,Hd)
        T, B, C = t.shape
        return t.view(T, B, self.n_heads, self.head_dim).permute(1, 2, 0, 3)

    def forward(self, x, past_kv=None):
        T, B, C = x.shape
        qkv = self.qkv(x)             # (T,B,3C)
        q, k, v = qkv.chunk(3, dim=-1)

        q = self._split(q)            # (B,nH,T,Hd)
        k = self._split(k)
        v = self._split(v)

        if past_kv is not None:
            Kpast, Vpast = past_kv    # (B,nH,Tp,Hd)
            k = torch.cat([Kpast, k], dim=2)
            v = torch.cat([Vpast, v], dim=2)

        if hasattr(F, "scaled_dot_product_attention"):
            # Use causal mask only when computing a fresh block (no past)
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=(past_kv is None),
            )  # (B,nH,T,Hd)
        else:
            # Manual attention (fallback)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,nH,T,Tk)
            if past_kv is None:
                Tk = att.size(-1)
                if self.tri_mask_cache is None or self.tri_mask_cache.size(-1) < Tk or self.tri_mask_cache.size(-2) < T:
                    self.tri_mask_cache = torch.tril(torch.ones(T, Tk, device=x.device, dtype=torch.bool))
                att = att.masked_fill(~self.tri_mask_cache, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v

        # merge heads -> (T,B,C)
        y = y.permute(2, 0, 1, 3).contiguous().view(T, B, C)
        y = self.resid_drop(self.out(y))
        return y, (k, v)


class TransformerBlock(nn.Module):
    """
    Pre-norm block:
      x = x + Attn(RMSNorm(x))
      x = x + MLP(RMSNorm(x))
    """
    def __init__(self, d_model, n_heads, dropout=0.0, mlp_mult=4.0, activation="gelu", bias=True):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout, bias=bias)
        self.norm2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, mult=mlp_mult, dropout=dropout, activation=activation)

    def forward(self, x, past_kv=None):
        att_out, present_kv = self.attn(self.norm1(x), past_kv=past_kv)
        x = x + att_out
        x = x + self.mlp(self.norm2(x))
        return x, present_kv


class TransformerModel(nn.Module):
    """
    Decoder-only Transformer with:
      - token + learned positional embeddings
      - RMSNorm pre-norm blocks
      - KV-cache aware attention for fast generation
    Training forward: tokens_seq (T,B) -> logits (T,B,V)
    """
    def __init__(
        self,
        vocab_size=50257,
        d_model=1024,
        n_heads=8,
        n_blocks=6,
        block_size=1024,
        dropout=0.0,
        mlp_mult=4.0,
        activation="gelu",
        tie_weights=True,
        bias=True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.block_size = block_size

        # (a) embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

        # (c) transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                mlp_mult=mlp_mult,
                activation=activation,
                bias=bias,
            ) for _ in range(n_blocks)
        ])

        # final norm + (d) unembedding
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ---- Training / full forward (no cache) ----
    def forward(self, tokens_seq):
        """
        tokens_seq: (T,B)
        returns logits: (T,B,V)
        """
        T, B = tokens_seq.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.block_size}")

        pos = torch.arange(0, T, device=tokens_seq.device)  # (T,)
        x = self.tok_emb(tokens_seq) + self.pos_emb(pos).unsqueeze(1)  # (T,B,C)
        x = self.drop(x)

        present = []
        for blk in self.blocks:
            x, kv = blk(x, past_kv=None)  # training path
            present.append(kv)            # unused, but could be inspected

        x = self.final_norm(x)
        logits = self.lm_head(x)  # (T,B,V)
        return logits

    # ---- One-step cached forward for fast generation ----
    @torch.no_grad()
    def forward_one_step(self, tok_t, t_pos, cache):
        """
        tok_t: (1,B) last token ids
        t_pos: int (absolute position index)
        cache: list of (K,V) per block (or None) length == n_blocks
        returns: logits_t: (B,V), new_cache: list of (K,V)
        """
        B = tok_t.size(1)
        x = self.tok_emb(tok_t) + self.pos_emb(torch.tensor([t_pos], device=tok_t.device)).unsqueeze(1)  # (1,B,C)

        new_cache = []
        for i, blk in enumerate(self.blocks):
            past_kv = None if cache is None else cache[i]
            x, present_kv = blk(x, past_kv=past_kv)   # T=1 in cached path
            new_cache.append(present_kv)

        x = self.final_norm(x)          # (1,B,C)
        logits_t = self.lm_head(x)[0]   # (B,V)
        return logits_t, new_cache

    # ---- KV-cached generation loop ----
    @torch.no_grad()
    def generate_kv(self, idx, max_new_tokens=20, temperature=1.0, top_p=None, top_k=0):
        """
        KV-cached generation. idx: (B,T0) prefix on device.
        Returns: (B, T0 + max_new_tokens)
        """
        self.eval()
        B, T0 = idx.shape
        device = idx.device

        # Prime cache by running through prefix tokens (except we only need to step through)
        cache = [None] * len(self.blocks)
        # run all but last prefix token to fill cache
        for t in range(T0 - 1):
            tok_t = idx[:, t:t+1].transpose(0, 1)     # (1,B)
            _, cache = self.forward_one_step(tok_t, t_pos=t, cache=cache)

        cur = idx
        for t in range(T0 - 1, T0 - 1 + max_new_tokens):
            tok_t = cur[:, -1:].transpose(0, 1)       # (1,B)
            logits, cache = self.forward_one_step(tok_t, t_pos=t, cache=cache)  # (B,V)

            # sampling
            logits = logits / max(temperature, 1e-6)
            if top_k and top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)

            if top_p is not None and 0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                # keep smallest set with cum >= top_p
                cutoff = (cum <= top_p).sum(dim=-1, keepdim=True) + 1
                cutoff = torch.clamp(cutoff, max=probs.size(-1))
                mask = torch.ones_like(probs, dtype=torch.bool)
                for b in range(B):
                    mask[b, sorted_idx[b, :cutoff[b, 0]]] = False
                probs = probs.masked_fill(mask, 0)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            next_tok = torch.multinomial(probs, num_samples=1)  # (B,1)
            cur = torch.cat([cur, next_tok], dim=1)

        return cur


################################################################################
# 6. K-Means Monosemantic (DISABLED by default)
################################################################################


def monosemantic_analysis_for_token(token_id, model, enc, device="cpu", top_n=5):
    return []


################################################################################
# 7. Single code path for text generation
################################################################################

def nucleus_sampling(logits, p=0.95):
    """
    logits: 1D tensor of shape (vocab_size,)
    p: cumulative probability threshold (0 < p <= 1)
    returns: sampled token id (int)
    """
    # 1. Convert logits → probabilities
    probs = F.softmax(logits, dim=-1)

    # 2. Sort probabilities descending
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # 3. Compute cumulative probability
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 4. Find cutoff index where cumulative prob >= p
    cutoff = torch.sum(cumulative_probs <= p).item() + 1  # +1 to include boundary token

    # 5. Truncate to top-p tokens
    sorted_probs = sorted_probs[:cutoff]
    sorted_indices = sorted_indices[:cutoff]

    # 6. Renormalize truncated probs to sum = 1
    sorted_probs = sorted_probs / torch.sum(sorted_probs)

    # 7. Sample one token id from this distribution
    chosen_index = torch.multinomial(sorted_probs, num_samples=1).item()
    chosen_token = sorted_indices[chosen_index].item()

    return chosen_token



def generate_text(model, enc, init_text, max_new_tokens=20, device="cpu",
                  top_p=None,
                  monosemantic_info=None,
                  do_monosemantic=False):
    """
    A single code path for all models:
      - We keep a growing list 'context_tokens'.
      - At each step, we feed the entire context as (seq_len,1) to model(...).
      - We get model(...)->(seq_len,1,vocab_size). We take the final step's logits => logits[-1,0,:].
      - We pick next token (greedy or top-p), append to context_tokens.
      - Optionally do monosemantic analysis on that newly generated token.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        context_tokens = enc.encode(init_text)
        annotation_list = []

        for step_i in range(max_new_tokens):
            seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)
            logits_seq = model(seq_tensor)              # (seq_len,1,vocab_size)
            next_logits = logits_seq[-1, 0, :]         # shape (vocab_size,)

            if top_p is None:
                # greedy
                chosen_token = torch.argmax(next_logits).item()
            else:
                chosen_token = nucleus_sampling(next_logits, p=top_p)

            context_tokens.append(chosen_token)

            if do_monosemantic and monosemantic_info is not None:
                neighbors = monosemantic_analysis_for_token(
                    chosen_token, model, monosemantic_info, enc, device=device, top_n=5
                )
                annotation_list.append((chosen_token, neighbors))
            else:
                annotation_list.append((chosen_token, []))

    model.train(was_training)

    final_text = enc.decode(context_tokens)
    prefix_text = enc.decode(context_tokens[:-max_new_tokens])
    annotated_strs = [prefix_text]
    for (tid, neighs) in annotation_list:
        token_str = enc.decode([tid])
        if neighs:
            neighbor_strs = [f"{enc.decode([x[1]])}" for x in neighs]
            annotated = f"{token_str}[NN={neighbor_strs}]"
        else:
            annotated = token_str
        annotated_strs.append(annotated)

    annotated_text = "".join(annotated_strs)
    return final_text, annotated_text

################################################################################
# 8. Training
################################################################################

def train_one_model(model,
                    loader,
                    epochs,
                    model_name,
                    device,
                    lr=1e-3,
                    log_steps=100,
                    sample_interval=30,
                    max_steps_per_epoch=None,
                    enc=None,
                    monosemantic_info=None,
                    prompt="Once upon a",
                    log_file_path=None):
    """
    We add `prompt` as an explicit argument so we can pass it down from main().
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ce_losses_by_epoch = []
    generations_by_epoch = {}
    start_time = time.time()
    next_sample_time = start_time
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        partial_loss = 0.0
        partial_count = 0
        epoch_ce_losses = []
        step_in_epoch = 0
        
        for batch_idx, batch_tokens in enumerate(loader, start=1):
            step_in_epoch += 1
            global_step += 1

            batch_tokens = batch_tokens.to(device)  # (seq_len, batch)

            logits = model(batch_tokens)  # (seq_len, batch, vocab_size)
            loss = compute_next_token_loss(logits, batch_tokens)
            epoch_ce_losses.append(float(loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            partial_loss += loss.item()
            partial_count += 1

            if batch_idx % log_steps == 0:
                avg_part_loss = partial_loss / partial_count
                print(f"[{model_name}] Epoch {epoch}/{epochs}, "
                      f"Step {batch_idx}/{len(loader)} (global step: {global_step}) "
                      f"Partial Avg Loss: {avg_part_loss:.4f}")
                partial_loss = 0.0
                partial_count = 0

            current_time = time.time()
            if current_time >= next_sample_time and enc is not None:
                with torch.no_grad():
                    print(f"\n[{model_name}] Generating sample texts at epoch={epoch}, step={batch_idx}...")
                    gen_settings = [
                        ("greedy", None),
                        ("top-p=0.2", 0.2),
                        ("top-p=0.5", 0.5),
                        ("top-p=0.8", 0.8),
                        ("top-p=0.95", 0.95),
                        ("top-p=1.0", 1.0),
                    ]
                    for label, p in gen_settings:
                        text_out, ann_out = sample_fast(
                            model, enc, prompt,
                            max_new_tokens=20, device=device, top_p=p,
                            monosemantic_info=monosemantic_info,
                            do_monosemantic=(monosemantic_info is not None),
                        )
                        print(f"[{model_name}] {label} Sample: {text_out}")
                        print(f" Annotated: {ann_out}\n")

                next_sample_time = current_time + sample_interval

            if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
                print(f"[{model_name}] Reached max_steps_per_epoch={max_steps_per_epoch}, ending epoch {epoch} early.")
                break
            
        ce_losses_by_epoch.append(epoch_ce_losses)
        avg_loss = total_loss / step_in_epoch
        print(f"[{model_name}] *** End of Epoch {epoch} *** Avg Loss: {avg_loss:.4f}")
        gen_settings = [
            ("greedy", None),
            ("top-p=0.2", 0.2),
            ("top-p=0.5", 0.5),
            ("top-p=0.8", 0.8),
            ("top-p=0.95", 0.95),
            ("top-p=1.0", 1.0),
        ]
        epoch_map = {}
        with torch.no_grad():
            for label, p in gen_settings:
                text_out, ann_out = sample_fast(
                    model, enc, prompt,
                    max_new_tokens=20, device=device, top_p=p,
                    monosemantic_info=monosemantic_info,
                    do_monosemantic=(monosemantic_info is not None),
                )
                epoch_map[label] = {"text": text_out, "annotated": ann_out}
        generations_by_epoch[epoch] = epoch_map
    if log_file_path is not None:
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"## {model_name} CE_losses_by_epoch\n")
            f.write(repr(ce_losses_by_epoch) + "\n\n")
            f.write(f"## {model_name} Generations_by_epoch\n")
            f.write(json.dumps(generations_by_epoch, indent=2) + "\n\n")

################################################################################
# 9. Main
################################################################################

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    log_file_path = os.path.join(args.save_dir, "logs.txt")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("")  # start fresh

    saved_weight_paths = []

    # Additional local variables from arguments
    k = args.kgram_k
    chunk_size = args.kgram_chunk_size

    embed_size = args.embed_size
    batch_size = 16
    num_epochs = 5
    learning_rate = 1e-3

    block_size = args.block_size
    train_subset_size = 20000
    log_interval_steps = 100
    sample_interval_seconds = 30

    max_steps_per_epoch = args.max_steps_per_epoch
    num_inner_layers = args.num_inner_mlp_layers

    # NEW: pick device from args.device_id, fallback to cpu if needed
    requested_device_id = args.device_id
    if requested_device_id.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device_id}' but CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device_id)

    print(f"Using device: {device}, block_size={block_size}, kgram_k={k}, chunk_size={chunk_size}, embed_size={embed_size}")

    ############################################################################
    # Data
    ############################################################################
    tinystories_seqs = []
    other_seqs = []

    if args.tinystories_weight > 0.0:
        print(f"Loading TinyStories from huggingface with weight={args.tinystories_weight}...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        dataset = dataset.select(range(train_subset_size))
    else:
        print("TinyStories weight=0 => skipping TinyStories.")
        dataset = None

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Vocab size: {vocab_size}")

    if dataset is not None:
        for sample in dataset:
            text = sample['text']
            tokens = enc.encode(text)
            tokens = tokens[:block_size]
            if len(tokens) > 0:
                tinystories_seqs.append(tokens)
        print(f"TinyStories sequences: {len(tinystories_seqs)}")

    if args.input_files:
        for filepath in args.input_files:
            print(f"Reading custom text file: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                tokens = enc.encode(line)
                tokens = tokens[:block_size]
                if len(tokens) > 0:
                    other_seqs.append(tokens)
        print(f"Custom input files: {len(other_seqs)} sequences loaded.")
    else:
        print("No custom input files provided.")

    p_tiny = args.tinystories_weight
    if len(tinystories_seqs) == 0 and p_tiny>0:
        print("Warning: TinyStories is empty but tinystories_weight>0. That's okay, no data from it.")
    combined_dataset = MixedSequenceDataset(
        tinystories_seqs=tinystories_seqs,
        other_seqs=other_seqs,
        p_tiny=p_tiny
    )

    train_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=seq_collate_fn
    )

    ############################################################################
    # Models
    ############################################################################
    kgram_model = KGramMLPSeqModel(
        vocab_size=vocab_size,
        k=k,
        embed_size=embed_size,
        num_inner_layers=num_inner_layers,
        chunk_size=chunk_size
    ).to(device)

    lstm_model = LSTMSeqModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=embed_size
    ).to(device)

    transformer = TransformerModel(
    ).to(device)

    models = {
        "kgram_mlp_seq": kgram_model,
        "lstm_seq": lstm_model,
        "kvcache_transformer": transformer,
    }


    ############################################################################
    # Train each model
    ############################################################################
    for model_name, model in models.items():
        print(f"\n=== Training model: {model_name} ===")
        train_one_model(
            model=model,
            loader=train_loader,
            epochs=num_epochs,
            model_name=model_name,
            device=device,
            lr=learning_rate,
            log_steps=log_interval_steps,
            sample_interval=sample_interval_seconds,
            max_steps_per_epoch=max_steps_per_epoch,
            enc=enc,
            prompt=args.prompt,
            log_file_path=log_file_path,
        )

        # Final generation from the user-provided prompt (args.prompt).
        with torch.no_grad():
            # 1) Greedy
            text_greedy, ann_greedy = sample_fast(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=None,
            )
            text_topp2, ann_topp2 = sample_fast(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=0.2,
            )
            text_topp5, ann_topp5 = sample_fast(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=0.5,
            )
            
            text_topp8, ann_topp8 = sample_fast(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=0.8,
            )
            # 2) top-p=0.95
            text_topp, ann_topp = sample_fast(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=0.95,
            )
            # 3) top-p=1.0 => full distribution random sampling
            text_topp1, ann_topp1 = sample_fast(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=1.0,
            )
            # --- Save final weights-only for this model ---
            final_path = os.path.join(args.save_dir, f"{model_name}_final_weights.pth")
            torch.save(model.state_dict(), final_path)
            saved_weight_paths.append(final_path)

        # Final generation from the user-provided prompt (args.prompt).
            with torch.no_grad():
                gen_settings = [
                    ("greedy", None),
                    ("top-p=0.2", 0.2),
                    ("top-p=0.5", 0.5),
                    ("top-p=0.8", 0.8),
                    ("top-p=0.95", 0.95),
                    ("top-p=1.0", 1.0),
                ]

                # Collect for logging
                final_generations = {}  # {label: {"text":..., "annotated":...}}

                for label, p in gen_settings:
                    text_out, ann_out = sample_fast(
                        model, enc, args.prompt, max_new_tokens=20, device=device, top_p=p,
                    )
                    final_generations[label] = {"text": text_out, "annotated": ann_out}

                    print(f"[{model_name}] Final sample ({label}) from prompt: '{args.prompt}'")
                    print(text_out)
                    print(f"Annotated:\n{ann_out}\n")

                # --- Save final weights-only for this model ---
                final_path = os.path.join(args.save_dir, f"{model_name}_final_weights.pth")
                torch.save(model.state_dict(), final_path)
                saved_weight_paths.append(final_path)
                print(f"[{model_name}] Saved final weights to: {final_path}")
                print("--------------------------------------------------")

                # --- Append final generations to logs.txt ---
                with open(log_file_path, "a", encoding="utf-8") as f:
                    f.write(f"## {model_name} final_generations\n")
                    f.write(json.dumps(final_generations, ensure_ascii=False) + "\n\n")


if __name__ == "__main__":
    main()
