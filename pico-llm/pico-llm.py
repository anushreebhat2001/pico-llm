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

    parser.add_argument("--test_fraction", type=float, default=0.1,
                        help="Fraction of data to reserve for test split. Default=0.1.")

    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory where logs and model weights will be saved.")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

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


class KGramMLPSeqModel(nn.Module):
    """
    For each position t in [0..seq_len-1], gather the last k tokens => embeddings => MLP => logits.
    Return (seq_len, batch, vocab_size).
    """

    def __init__(self, vocab_size, k=3, embed_size=1024, num_inner_layers=1, chunk_size=1):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size

        # NEW: learn embeddings instead of one-hot
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Input is concatenated embeddings of k tokens: dimension = k * embed_size
        in_dim = self.k * self.embed_size
        hidden_dim = self.embed_size

        layers = []
        if self.num_inner_layers <= 0:
            # Degenerate case: direct projection to vocab
            layers.append(nn.Linear(in_dim, self.vocab_size))
        else:
            # First (Linear->SiLU) block
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            # Additional (Linear->SiLU) blocks
            for _ in range(self.num_inner_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.SiLU())
            # Final projection to vocabulary logits
            layers.append(nn.Linear(hidden_dim, self.vocab_size))

        self.net = nn.Sequential(*layers)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        return: (seq_len, batch, vocab_size)
        We'll do a loop over time steps. chunk_size can reduce overhead.
        """
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
                    # Collect k previous tokens, padding with 0 at the left if needed
                    if t < self.k:
                        needed = self.k - t
                        context_ids = [0] * needed + tokens_seq[:t, b].tolist()
                    else:
                        context_ids = tokens_seq[t-self.k:t, b].tolist()

                    # context_ids: list of length k
                    context_ids_tensor = torch.tensor(
                        context_ids,
                        dtype=torch.long,
                        device=device,
                    )  # (k,)

                    # NEW: lookup embeddings for the k tokens
                    context_emb = self.embedding(context_ids_tensor)  # (k, embed_size)

                    # Flatten k embeddings into a single vector
                    context_flat = context_emb.view(1, -1)  # (1, k * embed_size)

                    logits_b = self.net(context_flat)  # (1, vocab_size)
                    batch_logits.append(logits_b)

                # stack batch dimension
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
        # x: (..., dim)
        norm_x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * norm_x


class MultiHeadSelfAttention(nn.Module):
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

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, d_model)
        mask: (1, 1, seq_len, seq_len) causal mask
        """
        B, T, C = x.shape
        H = self.n_heads
        D = self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # (B, H, T, T)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, H, T, T)
        attn_output = torch.matmul(attn_weights, v)        # (B, H, T, D)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4.0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.mlp_norm = RMSNorm(d_model)

        hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x, mask=None, collect_attn=False, attn_list=None, act_list=None):
        # x: (batch, seq_len, d_model)
        # Attention block (pre-norm)
        h = self.attn_norm(x)
        attn_out, attn_weights = self.attn(h, mask=mask)
        x = x + attn_out

        # MLP block (pre-norm)
        m = self.mlp_norm(x)
        mlp_out = self.mlp(m)
        x = x + mlp_out

        if collect_attn and (attn_list is not None) and (act_list is not None):
            attn_list.append(attn_weights.detach().cpu())
            act_list.append(mlp_out.detach().cpu())

        return x


class TransformerModel(nn.Module):
    def __init__(self, vocab_size=50257, d_model=1024, n_heads=2, n_blocks=4, block_size=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.block_size = block_size

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model=d_model, n_heads=n_heads) for _ in range(n_blocks)]
        )
        self.final_norm = RMSNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

        # Causal mask buffer
        mask = torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("causal_mask", mask, persistent=False)

        # For logging attention & activations
        self.attention_matrices = []
        self.activation_outputs = []

    def forward(self, tokens_seq, collect_attn=False):
        """
        tokens_seq: (seq_len, batch)
        returns: (seq_len, batch, vocab_size)
        """
        seq_len, batch_size = tokens_seq.shape
        device = tokens_seq.device

        # positions: (batch, seq_len)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)

        # token_emb expects (batch, seq_len)
        x = self.token_emb(tokens_seq.t()) + self.pos_emb(positions)  # (batch, seq_len, d_model)

        if collect_attn:
            self.attention_matrices = []
            self.activation_outputs = []

        mask = self.causal_mask[:, :, :seq_len, :seq_len]

        for block in self.blocks:
            x = block(
                x,
                mask=mask,
                collect_attn=collect_attn,
                attn_list=self.attention_matrices,
                act_list=self.activation_outputs,
            )

        x = self.final_norm(x)
        logits = self.unembed(x)  # (batch, seq_len, vocab_size)
        return logits.transpose(0, 1)  # (seq_len, batch, vocab_size)


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
    logits: 1D tensor (vocab_size,)
    p: float in (0,1]; cumulative probability mass to keep.
    Implements top-p (nucleus) sampling.
    """
    probs = torch.softmax(logits, dim=-1)

    if p >= 1.0:
        # Pure sampling from full distribution
        idx = torch.multinomial(probs, num_samples=1)
        return idx.item()

    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    # Smallest k such that cumulative probability >= p
    k = torch.searchsorted(cum_probs, torch.tensor(p, device=logits.device)).item() + 1
    k = max(1, min(k, sorted_probs.size(0)))

    truncated_probs = sorted_probs[:k]
    truncated_indices = sorted_indices[:k]
    truncated_probs = truncated_probs / truncated_probs.sum()

    sampled_idx = torch.multinomial(truncated_probs, num_samples=1).item()
    chosen_token = truncated_indices[sampled_idx].item()
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
                    train_loader,
                    test_loader,
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
                    top_p_values=None):
    """
    We add `prompt` as an explicit argument so we can pass it down from main().
    """

    if top_p_values is None:
        top_p_values = [0.2, 0.5, 0.75, 0.95, 1.0]

    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    next_sample_time = start_time
    global_step = 0

    train_losses_per_epoch = []
    test_losses_per_epoch = []
    generations_per_epoch = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        partial_loss = 0.0
        partial_count = 0

        step_in_epoch = 0
        epoch_train_losses = []

        for batch_idx, batch_tokens in enumerate(train_loader, start=1):
            step_in_epoch += 1
            global_step += 1

            batch_tokens = batch_tokens.to(device)  # (seq_len, batch)

            logits = model(batch_tokens)  # (seq_len, batch, vocab_size)
            loss = compute_next_token_loss(logits, batch_tokens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            total_loss += loss_val
            partial_loss += loss_val
            partial_count += 1
            epoch_train_losses.append(loss_val)

            if batch_idx % log_steps == 0:
                avg_part_loss = partial_loss / partial_count
                print(f"[{model_name}] Epoch {epoch}/{epochs}, "
                      f"Step {batch_idx}/{len(train_loader)} (global step: {global_step}) "
                      f"Partial Avg Loss: {avg_part_loss:.4f}")
                partial_loss = 0.0
                partial_count = 0

            current_time = time.time()
            if current_time >= next_sample_time and enc is not None:
                with torch.no_grad():
                    print(f"\n[{model_name}] Generating sample text (greedy) at epoch={epoch}, step={batch_idx}...")
                    text_greedy, ann_greedy = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=None,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Greedy Sample: {text_greedy}")
                    print(f" Annotated: {ann_greedy}\n")

                    print(f"[{model_name}] Generating sample text (top-p=0.95) at epoch={epoch}, step={batch_idx}...")
                    text_topp, ann_topp = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=0.95,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=0.95) Sample: {text_topp}")
                    print(f" Annotated: {ann_topp}\n")

                    # third generation => top-p=1.0 => full distribution random sampling
                    print(f"[{model_name}] Generating sample text (top-p=1.0) at epoch={epoch}, step={batch_idx}...")
                    text_topp1, ann_topp1 = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=1.0,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=1.0) Sample: {text_topp1}")
                    print(f" Annotated: {ann_topp1}\n")

                next_sample_time = current_time + sample_interval

            if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
                print(f"[{model_name}] Reached max_steps_per_epoch={max_steps_per_epoch}, ending epoch {epoch} early.")
                break

        avg_loss = total_loss / step_in_epoch
        print(f"[{model_name}] *** End of Epoch {epoch} *** Avg Train Loss: {avg_loss:.4f}")
        train_losses_per_epoch.append(epoch_train_losses)

        # -------------------------------
        # Evaluation on test set
        # -------------------------------
        epoch_test_losses = []
        if test_loader is not None:
            model.eval()
            with torch.no_grad():
                for batch_tokens in test_loader:
                    batch_tokens = batch_tokens.to(device)
                    logits = model(batch_tokens)
                    loss = compute_next_token_loss(logits, batch_tokens)
                    epoch_test_losses.append(loss.item())
            if len(epoch_test_losses) > 0:
                avg_test_loss = sum(epoch_test_losses) / len(epoch_test_losses)
            else:
                avg_test_loss = float("nan")
            print(f"[{model_name}] Epoch {epoch}/{epochs} *** Avg Test Loss: {avg_test_loss:.4f}")
        test_losses_per_epoch.append(epoch_test_losses)

        # -------------------------------
        # Generations for different p-values for this epoch
        # -------------------------------
        epoch_generations = {}
        if enc is not None:
            with torch.no_grad():
                # Greedy
                text_greedy, _ = generate_text(
                    model, enc, prompt, max_new_tokens=20, device=device,
                    top_p=None
                )
                epoch_generations["greedy"] = text_greedy
                # Different nucleus sampling values
                for pval in top_p_values:
                    text_p, _ = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=pval
                    )
                    epoch_generations[str(pval)] = text_p

        generations_per_epoch.append(epoch_generations)

    return train_losses_per_epoch, test_losses_per_epoch, generations_per_epoch


################################################################################
# 9. Main
################################################################################

def split_sequences(seqs, test_fraction):
    train_seqs = []
    test_seqs = []
    for s in seqs:
        if random.random() < test_fraction:
            test_seqs.append(s)
        else:
            train_seqs.append(s)
    return train_seqs, test_seqs


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Additional local variables from arguments
    k = args.kgram_k
    chunk_size = args.kgram_chunk_size

    embed_size = args.embed_size
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.learning_rate

    block_size = args.block_size
    train_subset_size = 20000
    log_interval_steps = 100
    sample_interval_seconds = 30

    max_steps_per_epoch = args.max_steps_per_epoch
    num_inner_layers = args.num_inner_mlp_layers
    test_fraction = args.test_fraction

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
    if len(tinystories_seqs) == 0 and p_tiny > 0:
        print("Warning: TinyStories is empty but tinystories_weight>0. That's okay, no data from it.")

    # Train/test split on sequence lists
    tiny_train, tiny_test = split_sequences(tinystories_seqs, test_fraction)
    other_train, other_test = split_sequences(other_seqs, test_fraction)

    # Fallback if test split is empty
    if len(tiny_test) == 0 and len(other_test) == 0:
        print("Warning: test split is empty; using a small portion of training data as test.")
        if len(tiny_train) > 1:
            n_move = max(1, len(tiny_train) // 10)
            tiny_test = tiny_train[-n_move:]
            tiny_train = tiny_train[:-n_move]
        elif len(other_train) > 1:
            n_move = max(1, len(other_train) // 10)
            other_test = other_train[-n_move:]
            other_train = other_train[:-n_move]

    train_dataset = MixedSequenceDataset(
        tinystories_seqs=tiny_train,
        other_seqs=other_train,
        p_tiny=p_tiny
    )

    # It is possible that test set is empty; handle gracefully
    test_dataset = None
    if len(tiny_test) + len(other_test) > 0:
        test_dataset = MixedSequenceDataset(
            tinystories_seqs=tiny_test,
            other_seqs=other_test,
            p_tiny=p_tiny
        )
    else:
        print("No test data available; test_loss lists will be empty.")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=seq_collate_fn
    )

    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=seq_collate_fn
        )
    else:
        test_loader = None

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

    kv_transformer = TransformerModel(
        vocab_size=vocab_size,
        d_model=embed_size,
        n_heads=8,
        n_blocks=6,
        block_size=block_size
    ).to(device)

    models = {
        "kgram_mlp_seq": kgram_model,
        "lstm_seq": lstm_model,
        "kvcache_transformer": kv_transformer,
    }

    all_loss_logs = {}
    all_generation_logs = {}
    top_p_values = [0.2, 0.5, 0.75, 0.95, 1.0]

    ############################################################################
    # Train each model
    ############################################################################
    for model_name, model in models.items():
        print(f"\n=== Training model: {model_name} ===")
        train_losses, test_losses, gen_per_epoch = train_one_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=num_epochs,
            model_name=model_name,
            device=device,
            lr=learning_rate,
            log_steps=log_interval_steps,
            sample_interval=sample_interval_seconds,
            max_steps_per_epoch=max_steps_per_epoch,
            enc=enc,
            prompt=args.prompt,  # <--- Pass the user-specified prompt here
            top_p_values=top_p_values,
        )

        all_loss_logs[model_name] = {
            "train": train_losses,
            "test": test_losses,
        }
        all_generation_logs[model_name] = gen_per_epoch

        # Save model weights to load later
        weights_path = os.path.join(args.output_dir, f"{model_name}_final_weights.pt")
        torch.save(model.state_dict(), weights_path)
        print(f"[{model_name}] Saved final weights to {weights_path}")

        # Final generation from the user-provided prompt (args.prompt).
        with torch.no_grad():
            # 1) Greedy
            text_greedy, ann_greedy = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=None,
            )
            # 2) different p-values for final prompt output
            final_generations = {"greedy": text_greedy}
            for pval in top_p_values:
                text_p, ann_p = generate_text(
                    model, enc, args.prompt, max_new_tokens=20, device=device,
                    top_p=pval,
                )
                final_generations[str(pval)] = text_p

        print(f"[{model_name}] Final samples from prompt: '{args.prompt}'")
        for key, txt in final_generations.items():
            print(f"  Sampling mode {key}:")
            print(f"  {txt}\n")

        # Save attention matrices and activation outputs for Transformer
        if isinstance(model, TransformerModel):
            with torch.no_grad():
                sample_tokens = torch.tensor(
                    enc.encode(args.prompt),
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(1)  # (seq_len, 1)
                _ = model(sample_tokens, collect_attn=True)

                attn_file = os.path.join(args.output_dir, f"{model_name}_attention_matrices.pt")
                acts_file = os.path.join(args.output_dir, f"{model_name}_activations.pt")
                torch.save(model.attention_matrices, attn_file)
                torch.save(model.activation_outputs, acts_file)
                print(f"[{model_name}] Saved attention matrices to {attn_file}")
                print(f"[{model_name}] Saved MLP activation outputs to {acts_file}")

    # Save JSON logs for losses and generations
    loss_log_path = os.path.join(args.output_dir, "loss_logs.json")
    gen_log_path = os.path.join(args.output_dir, "generation_logs.json")

    with open(loss_log_path, "w", encoding="utf-8") as f:
        json.dump(all_loss_logs, f, indent=2)
    with open(gen_log_path, "w", encoding="utf-8") as f:
        json.dump(all_generation_logs, f, indent=2)

    print(f"\nSaved loss logs to {loss_log_path}")
    print(f"Saved generation logs to {gen_log_path}")

    # Finally, let's share how I'm feeling:
    print("\n*** I'm feeling great today! Hope you're well, too. ***")


if __name__ == "__main__":
    main()
