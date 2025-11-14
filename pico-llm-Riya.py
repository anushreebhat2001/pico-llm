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
    
    # For saving Transformer model:
    parser.add_argument("--save_dir", type=str, default="./saved_models",
                        help="Directory to save model weights. Default='./saved_models'")
    parser.add_argument("--save_name", type=str, default=None,
                        help="Custom name for saved model (e.g., 'my_transformer'). If not provided, uses model_name")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to saved model weights to load")


    args = parser.parse_args()
    return args

################################################################################
# Save and load model
################################################################################

def save_model_weights(model, model_name, save_dir="./saved_models"):
    """
    Save just the model weights.
    
    Args:
        model: The model to save
        model_name: Name for the file (e.g., "transformer")
        save_dir: Directory to save in
        
    Returns:
        Path to saved file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    filepath = os.path.join(save_dir, f"{model_name}_weights.pt")
    torch.save(model.state_dict(), filepath)
    
    print(f"✓ Model weights saved to: {filepath}")
    return filepath


def load_model_weights(model, filepath, device="cpu"):
    """
    Load model weights.
    
    Args:
        model: The model to load weights into
        filepath: Path to the weights file
        device: Device to load on
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model weights not found: {filepath}")
    
    print(f"Loading model weights from: {filepath}")
    model.load_state_dict(torch.load(filepath, map_location=device))
    print("✓ Model weights loaded successfully")
    
    return model


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

        layers = []
        input_dim = k * vocab_size
        hidden_dim = embed_size
        for i in range(num_inner_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, vocab_size))  # final logits layer
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

                    context_oh = F.one_hot(
                        torch.tensor(context_ids, dtype=torch.long, device=tokens_seq.device),
                        num_classes=self.vocab_size
                    )
                    context_flat = context_oh.flatten().float().unsqueeze(0)
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
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = x.pow(2).mean(dim=-1, keepdim=True)  
        denom = torch.sqrt(ms + self.eps)         
        return x / denom * self.weight

class CausalSelfAttention(nn.Module):
    """
    Multi-head masked self-attention layer with causal masking.
    Prevents positions from attending to future positions.
    
    Note: d_k (key dimension) and d_v (value dimension) can be different from d_model.
    For standard transformers, we use d_k = d_v = d_model for simplicity.
    """
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, dropout=0.1):
        super().__init__()
        
        # If not specified, use d_model for all dimensions (standard practice)
        self.d_model = d_model
        self.d_k = d_k if d_k is not None else d_model
        self.d_v = d_v if d_v is not None else d_model
        self.n_heads = n_heads
        
        # Each head gets a portion of the total dimension
        assert self.d_k % n_heads == 0, "d_k must be divisible by n_heads"
        assert self.d_v % n_heads == 0, "d_v must be divisible by n_heads"
        
        self.head_dim_k = self.d_k // n_heads  # Dimension per head for K and Q
        self.head_dim_v = self.d_v // n_heads  # Dimension per head for V
        
        # Separate Query, Key, Value projections
        # Q and K must have same dimension for dot product
        self.q_proj = nn.Linear(d_model, self.d_k)  # Can project to different dimension!
        self.k_proj = nn.Linear(d_model, self.d_k)  # Same as Q for compatibility
        self.v_proj = nn.Linear(d_model, self.d_v)  # Can be different from K/Q
        
        # Output projection: maps from d_v back to d_model
        self.c_proj = nn.Linear(self.d_v, d_model)
        
        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: (seq_len, batch, d_model)
        returns: (seq_len, batch, d_model)
        """
        seq_len, batch_size, d_model = x.shape
        
        # Transpose to (batch, seq_len, d_model) for attention computation
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)
        
        # Calculate Q, K, V separately for all heads
        q = self.q_proj(x)  # (batch, seq_len, d_k)
        k = self.k_proj(x)  # (batch, seq_len, d_k)
        v = self.v_proj(x)  # (batch, seq_len, d_v)
        
        # Split into multiple heads: (batch, seq_len, n_heads, head_dim)
        # Then transpose to: (batch, n_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim_k).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim_v).transpose(1, 2)
        
        # Compute attention scores
        # (batch, n_heads, seq_len, head_dim_k) @ (batch, n_heads, head_dim_k, seq_len)
        # -> (batch, n_heads, seq_len, seq_len)
        # IMPORTANT: Scale by sqrt(head_dim_k) - the dimension of keys/queries!
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim_k))
        
        # Apply causal mask (prevent attending to future positions)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float('-inf'))
        
        # Apply softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        # (batch, n_heads, seq_len, seq_len) @ (batch, n_heads, seq_len, head_dim_v)
        # -> (batch, n_heads, seq_len, head_dim_v)
        y = att @ v
        
        # Re-assemble all head outputs side by side
        # (batch, n_heads, seq_len, head_dim_v) -> (batch, seq_len, n_heads, head_dim_v)
        # -> (batch, seq_len, d_v)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_v)
        
        # Output projection (maps d_v back to d_model for residual connection)
        y = self.resid_dropout(self.c_proj(y))
        
        # Transpose back to (seq_len, batch, d_model)
        y = y.transpose(0, 1)
        
        return y
    
class MLP(nn.Module):
    """
    Position-wise Feed-Forward Network.
    GPT-2 uses 4*d_model as the inner dimension with GELU activation.
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model)
        self.c_proj = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: (seq_len, batch, d_model)
        returns: (seq_len, batch, d_model)
        """
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class TransformerBlock(nn.Module):
    """
    Single Transformer block with pre-normalization.
    Architecture: LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual
    """
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, dropout=0.1):
        super().__init__()
        self.ln_1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, d_k, d_v, dropout)
        self.ln_2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, dropout)
        
    def forward(self, x):
        """
        x: (seq_len, batch, d_model)
        returns: (seq_len, batch, d_model)
        """
        # Attention with residual connection (pre-norm)
        x = x + self.attn(self.ln_1(x))
        # MLP with residual connection (pre-norm)
        x = x + self.mlp(self.ln_2(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size=50257, d_model=1024, n_heads=2, n_blocks=4, max_seq_len = 1024, dropout = 0.1, attention_d_k = None, attention_d_v= None):
        super().__init__()

        
        # If attention_d_k/d_v not specified, they default to d_model in CausalSelfAttention
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.max_seq_len = max_seq_len
        self.attention_d_k = attention_d_k  # Optional: use smaller dim for efficiency
        self.attention_d_v = attention_d_v  # Optional: use smaller dim for efficiency
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks (now with flexible attention dimensions)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, attention_d_k, attention_d_v, dropout) 
            for _ in range(n_blocks)
        ])
        
        # Final layer norm
        self.ln_f = RMSNorm(d_model)
        
        # Language modeling head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying: share weights between token embedding and lm_head
        self.token_embedding.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to residual projections (GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_blocks))
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2 paper."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch) - input token indices
        returns: (seq_len, batch, vocab_size) - logits for next token prediction
        """
        seq_len, batch_size = tokens_seq.shape
        
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max {self.max_seq_len}"
        
        # Create position indices
        positions = torch.arange(seq_len, dtype=torch.long, device=tokens_seq.device)
        positions = positions.unsqueeze(1).expand(seq_len, batch_size)
        
        # Get embeddings
        token_emb = self.token_embedding(tokens_seq)  # (seq_len, batch, d_model)
        pos_emb = self.position_embedding(positions)   # (seq_len, batch, d_model)
        
        # Combine embeddings with dropout
        x = self.dropout(token_emb + pos_emb)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)  # (seq_len, batch, vocab_size)
        
        return logits


################################################################################
# 6. K-Means Monosemantic (DISABLED by default)
################################################################################


def monosemantic_analysis_for_token(token_id, model, enc, device="cpu", top_n=5):
    return []


################################################################################
# 7. Single code path for text generation
################################################################################

def nucleus_sampling(logits, p=0.95):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    k = torch.searchsorted(cumulative_probs, torch.tensor(p, device=logits.device))
    k = max(k.item(), 1)

    topk_probs = sorted_probs[:k]
    topk_indices = sorted_indices[:k]
    topk_probs = topk_probs / topk_probs.sum()
   
    chosen = torch.multinomial(topk_probs, 1).item()
    return topk_indices[chosen].item()



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
                    save_dir="./saved_models"):
    """
    We add `prompt` as an explicit argument so we can pass it down from main().
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    next_sample_time = start_time
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        partial_loss = 0.0
        partial_count = 0

        step_in_epoch = 0
        for batch_idx, batch_tokens in enumerate(loader, start=1):
            step_in_epoch += 1
            global_step += 1

            batch_tokens = batch_tokens.to(device)  # (seq_len, batch)

            logits = model(batch_tokens)  # (seq_len, batch, vocab_size)
            loss = compute_next_token_loss(logits, batch_tokens)

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
                    print(f"\n[{model_name}] Generating sample text (greedy) at epoch={epoch}, step={batch_idx}...")
                    text_greedy, ann_greedy = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=None,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Greedy Sample: {text_greedy}")
                    print(f" Annotated: {ann_greedy}\n")

                    # print(f"[{model_name}] Generating sample text (top-p=0.95) at epoch={epoch}, step={batch_idx}...")
                    # text_topp, ann_topp = generate_text(
                    #     model, enc, prompt, max_new_tokens=20, device=device,
                    #     top_p=0.2,
                    #     monosemantic_info=monosemantic_info,
                    #     do_monosemantic=(monosemantic_info is not None)
                    # )
                    # print(f" Top-p (p=0.2) Sample: {text_topp}")
                    # print(f" Annotated: {ann_topp}\n")

                    # print(f"[{model_name}] Generating sample text (top-p=0.95) at epoch={epoch}, step={batch_idx}...")
                    # text_topp, ann_topp = generate_text(
                    #     model, enc, prompt, max_new_tokens=20, device=device,
                    #     top_p=0.3,
                    #     monosemantic_info=monosemantic_info,
                    #     do_monosemantic=(monosemantic_info is not None)
                    # )
                    # print(f" Top-p (p=0.3) Sample: {text_topp}")
                    # print(f" Annotated: {ann_topp}\n")

                    # print(f"[{model_name}] Generating sample text (top-p=0.95) at epoch={epoch}, step={batch_idx}...")
                    # text_topp, ann_topp = generate_text(
                    #     model, enc, prompt, max_new_tokens=20, device=device,
                    #     top_p=0.5,
                    #     monosemantic_info=monosemantic_info,
                    #     do_monosemantic=(monosemantic_info is not None)
                    # )
                    # print(f" Top-p (p=0.5) Sample: {text_topp}")
                    # print(f" Annotated: {ann_topp}\n")

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
        print(f"[{model_name}] *** End of Epoch {epoch} *** Avg Loss: {avg_loss:.4f}")

    print(f"\n[{model_name}] Training complete! Saving model weights")
    save_model_weights(model, model_name, save_dir)

################################################################################
# 9. Main
################################################################################

def main():
    args = parse_args()

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
    vocab_size=vocab_size,
    d_model=embed_size,  # Uses the embed_size from args
    n_heads=8,           # Increased from 2 for better performance
    n_blocks=4,          # 4 transformer blocks
    max_seq_len=block_size,  # Use block_size from args
    dropout=0.1).to(device)    

    models = {
      #"kgram_mlp_seq": kgram_model,
      #  "lstm_seq": lstm_model,
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
            prompt=args.prompt  # <--- Pass the user-specified prompt here
        )

        # Final generation from the user-provided prompt (args.prompt).
        with torch.no_grad():
            # 1) Greedy
            text_greedy, ann_greedy = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=None,
            )
            text_topp2, ann_topp2 = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=0.2,
            )
            text_topp3, ann_topp3 = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=0.3,
            )
            text_topp4, ann_topp4 = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=0.5,
            )
            # 2) top-p=0.95
            text_topp, ann_topp = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=0.95,
            )
            # 3) top-p=1.0 => full distribution random sampling
            text_topp1, ann_topp1 = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=1.0,
            )

        print(f"[{model_name}] Final sample (greedy) from prompt: '{args.prompt}'")
        print(text_greedy)
        print(f"Annotated:\n{ann_greedy}\n")

        print(f"[{model_name}] Final sample (top-p=0.2) from prompt: '{args.prompt}'")
        print(text_topp2)
        print(f"Annotated:\n{ann_topp2}")

        print(f"[{model_name}] Final sample (top-p=0.3) from prompt: '{args.prompt}'")
        print(text_topp3)
        print(f"Annotated:\n{ann_topp3}")

        print(f"[{model_name}] Final sample (top-p=0.5) from prompt: '{args.prompt}'")
        print(text_topp4)
        print(f"Annotated:\n{ann_topp4}")

        print(f"[{model_name}] Final sample (top-p=0.95) from prompt: '{args.prompt}'")
        print(text_topp)
        print(f"Annotated:\n{ann_topp}\n")

        print(f"[{model_name}] Final sample (top-p=1.0) from prompt: '{args.prompt}'")
        print(text_topp1)
        print(f"Annotated:\n{ann_topp1}")

        print("--------------------------------------------------")

    # Finally, let's share how I'm feeling:
    print("\n*** I'm feeling great today! Hope you're well, too. ***")

if __name__ == "__main__":
    main()