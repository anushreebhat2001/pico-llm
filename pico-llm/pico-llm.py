# starter code by matus & o1-pro
import argparse
import time
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
    K-gram MLP: Looks at last k tokens to predict the next one.
    Uses embeddings instead of one-hot encoding for efficiency.
    
    MY IMPLEMENTATION: I changed this from using one-hot vectors (which would be HUGE)
    to using embeddings (much more efficient). This was a key optimization I learned about!
    """

    def __init__(self, vocab_size, k=3, embed_size=1024, num_inner_layers=1, chunk_size=1):
        super().__init__()
        self.k = k  # how many previous tokens to look at (context window)
        self.vocab_size = vocab_size  # size of our vocabulary (~50k for GPT-2)
        self.embed_size = embed_size  # dimension of our embedding vectors
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size

        # so this is PART 1: Embedding layer - This is the key change I made!!!!!!
        # Instead of one-hot vectors (which would be 50k dimensional), 
        # we map each token to a dense vector of size embed_size
        # This is WAY more efficient and learns better representations probably. 
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # and for PART 2: Build the MLP (Multi-Layer Perceptron)
        # Math here: if we look at k tokens, each with embed_size dimensions,
        # our input will be k * embed_size dimensional
        input_dim = k * embed_size  # e.g., 3 * 1024 = 3072 for k=3
        hidden_dim = 2048  # I chose this - could experiment with different sizes
        
        layers = []
        
        # First layer: takes our concatenated embeddings and maps to hidden space
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())  # SiLU activation - learned (basically chatGPTed) this is better than ReLU
        
        # Add extra hidden layers if specified (depth helps with complex patterns)
        for _ in range(num_inner_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())  # same activation for consistency
        
        # Final output layer: maps hidden representation to vocabulary probabilities
        layers.append(nn.Linear(hidden_dim, vocab_size))
        
        # Pack everything into a sequential model for easy forward pass
        self.net = nn.Sequential(*layers)

    def forward(self, tokens_seq):
        """
        MY FORWARD PASS IMPLEMENTATION:
        tokens_seq: (seq_len, batch) - our input token sequences
        return: (seq_len, batch, vocab_size) - predictions for next token at each position
        
        The tricky part here is that for each position, we need to look at the 
        previous k tokens and predict what comes next!
        """
        seq_len, batch_size = tokens_seq.shape
        outputs = []

        # Process in chunks to avoid memory issues (learned this the hard way lol claude helped)
        start = 0
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            block_outputs = []
            
            # For each timestep in this chunk...
            for t in range(start, end):
                batch_logits = []
                
                # Process each item in the batch separately
                for b in range(batch_size):
                    # STEP 1: Get context window (last k tokens before position t)
                    if t < self.k:
                        # Edge case: if we don't have k previous tokens, pad with zeros
                        needed = self.k - t
                        context_ids = [0]*needed + tokens_seq[:t, b].tolist()
                    else:
                        # Normal case: take the last k tokens
                        context_ids = tokens_seq[t-self.k:t, b].tolist()

                    # STEP 2: Convert token IDs to embeddings (this is the key improvement)
                    context_tensor = torch.tensor(
                        context_ids, 
                        dtype=torch.long, 
                        device=tokens_seq.device
                    )
                    context_emb = self.embedding(context_tensor)  # shape: (k, embed_size)
                    
                    # STEP 3: Flatten embeddings to feed into MLP
                    # We concatenate all k embeddings into one big vector
                    context_flat = context_emb.flatten().unsqueeze(0)  # shape: (1, k*embed_size)
                    
                    # STEP 4: Pass through our MLP to get predictions
                    logits_b = self.net(context_flat)  # shape: (1, vocab_size)
                    batch_logits.append(logits_b)
                    
                # Combine all batch items for this timestep
                block_outputs.append(torch.cat(batch_logits, dim=0).unsqueeze(0))  # (1, batch, vocab_size)

            # Combine all timesteps in this chunk
            block_outputs = torch.cat(block_outputs, dim=0)  # (chunk_size, batch, vocab_size)
            outputs.append(block_outputs)
            start = end

        # Combine all chunks to get final output
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
# 5. Transformer Implementation
################################################################################

class RMSNorm(nn.Module):
    """
    MY IMPLEMENTATION here: Root Mean Square Normalization
    This is used in modern transformers like LLaMA instead of LayerNorm
    
    Why RMSNorm? It's simpler than LayerNorm (no bias term, no mean centering)
    but works just as well, I learned this is becoming the new standard. (again asked Claude)
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps  # small number to avoid division by zero
        
        # Learnable scale parameter (like LayerNorm's gamma)
        # Initialized to ones, so initially it's just identity
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        """
        MY RMSNorm forward pass:
        x: (..., dim) - input can be any shape as long as last dim is 'dim'
        
        The math: RMS = sqrt(mean(x^2) + eps)
        Then normalize: x / RMS, and scale by learnable weight
        """
        # Compute Root Mean Square over the last dimension
        # Why x**2? Because RMS is the square root of the mean of squares!
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize by RMS and apply learnable scaling
        # This is like z-score normalization but without subtracting the mean
        return self.weight * (x / rms)


class CausalSelfAttention(nn.Module):
    """
    MY IMPLEMENTATION: Single attention head with causal masking
    
    This is the core of the transformer! Each attention head looks at the sequence
    and figures out which tokens should pay attention to which other tokens.
    
    "Causal" means we can only look backwards - no cheating by seeing future tokens
    """
    def __init__(self, d_model, head_dim):
        super().__init__()
        self.d_model = d_model  # input dimension (e.g., 1024)
        self.head_dim = head_dim  # this head's dimension (e.g., 128)
        
        # Scale factor for attention scores - prevents softmax saturation
        # This is from the "Attention Is All You Need" paper!
        self.scale = head_dim ** -0.5  # = 1/sqrt(head_dim)
        
        # The famous Q, K, V projections (no bias needed)
        # Query: "what am I looking for?"
        # Key: "what do I contain?" 
        # Value: "what information do I have?"
        self.query = nn.Linear(d_model, head_dim, bias=False)
        self.key = nn.Linear(d_model, head_dim, bias=False)
        self.value = nn.Linear(d_model, head_dim, bias=False)
    
    def forward(self, x):
        """
        MY ATTENTION FORWARD PASS:
        x: (seq_len, batch, d_model) - input sequence
        returns: (seq_len, batch, head_dim) - attended output
        
        This implements the scaled dot-product attention with causal masking!
        """
        seq_len, batch, d_model = x.shape
        
        # STEP 1: Project input to Query, Key, Value
        Q = self.query(x)  # (seq_len, batch, head_dim) - what each token is looking for
        K = self.key(x)    # (seq_len, batch, head_dim) - what each token offers
        V = self.value(x)  # (seq_len, batch, head_dim) - the actual content to attend to
        
        # STEP 2: Reshape for batch matrix multiplication
        # PyTorch's bmm expects (batch, seq_len, head_dim)
        Q = Q.transpose(0, 1)  # (batch, seq_len, head_dim)
        K = K.transpose(0, 1)
        V = V.transpose(0, 1)
        
        # STEP 3: Compute attention scores - the magic happens here!
        # Q @ K^T gives us a (batch, seq_len, seq_len) matrix where entry (i,j) 
        # represents how much token i should attend to token j
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (batch, seq_len, seq_len)
        
        # STEP 4: Apply causal mask - this is CRUCIAL for language modeling!
        # We can't let tokens see into the future, so we mask out upper triangle
        # triu creates upper triangular matrix filled with -inf above diagonal
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
        scores = scores + mask.unsqueeze(0)  # broadcast mask across batch dimension
        
        # STEP 5: Apply softmax to get attention weights (probabilities)
        # -inf values become 0 after softmax, effectively masking them out
        attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)
        
        # STEP 6: Apply attention weights to values
        # This is the weighted sum - we take linear combination of all values
        # based on attention weights
        output = torch.bmm(attn_weights, V)  # (batch, seq_len, head_dim)
        
        # STEP 7: Reshape back to original format
        output = output.transpose(0, 1)  # (seq_len, batch, head_dim)
        
        return output


class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads that sum their outputs
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Create attention heads
        self.heads = nn.ModuleList([
            CausalSelfAttention(d_model, self.head_dim)
            for _ in range(n_heads)
        ])
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x):
        """
        x: (seq_len, batch, d_model)
        returns: (seq_len, batch, d_model)
        """
        # Run each head
        head_outputs = [head(x) for head in self.heads]
        
        # Concatenate
        concat_output = torch.cat(head_outputs, dim=-1)
        
        # Output projection
        output = self.out_proj(concat_output)
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    """
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # Standard 4x expansion
        
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x):
        """
        x: (seq_len, batch, d_model)
        """
        x = self.fc1(x)
        x = F.silu(x)  # SiLU activation
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block with Pre-Norm architecture
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        
        self.norm1 = RMSNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads)
        
        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model)
    
    def forward(self, x):
        """
        x: (seq_len, batch, d_model)
        """
        # Self-attention with residual (Pre-Norm)
        x = x + self.attention(self.norm1(x))
        
        # Feed-forward with residual (Pre-Norm)
        x = x + self.ffn(self.norm2(x))
        
        return x


class TransformerModel(nn.Module):
    """
    MY COMPLETE TRANSFORMER IMPLEMENTATION hehe:
    This is a causal decoder-only transformer just like GPT almost i guess.
    
    Architecture: Embedding -> N x TransformerBlocks -> Norm -> Output Projection
    Each block has: RMSNorm -> MultiHeadAttention -> RMSNorm -> FeedForward
    
    "Causal" = can only see past tokens (autoregressive language modeling)
    "Decoder-only" = we don't have an encoder part (unlike original Transformer)
    """
    def __init__(self, vocab_size=50257, d_model=1024, n_heads=8, n_blocks=4):
        super().__init__()
        # Store hyperparameters
        self.vocab_size = vocab_size  # size of vocabulary (GPT-2 uses ~50k)
        self.d_model = d_model        # hidden dimension (1024 is pretty standard)
        self.n_heads = n_heads        # number of attention heads (8 is common)
        self.n_blocks = n_blocks      # number of transformer layers (GPT-2 small has 12, but 4 is good for learning)
        
        # COMPONENT 1: Token embedding layer
        # Maps token IDs (integers) to dense vectors of size d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # COMPONENT 2: Stack of transformer blocks (this is the main architecture!)
        # Each block has attention + feedforward with residual connections
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads)
            for _ in range(n_blocks)
        ])
        
        # COMPONENT 3: Final layer normalization
        # Good practice to normalize before the final prediction
        self.final_norm = RMSNorm(d_model)
        
        # COMPONENT 4: Output projection (sometimes called "unembedding")
        # Maps from d_model back to vocabulary size for token prediction
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        # COMPONENT 5: Initialize weights properly (this matters a lot!)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        MY WEIGHT INITIALIZATION SCHEME:
        Proper initialization is super important for training stability!
        
        I learned that transformers are sensitive to initialization - too big and
        gradients explode, too small and they don't learn. 0.02 std is the sweet spot.
        """
        if isinstance(module, nn.Linear):
            # Initialize linear layer weights with small random values
            # 0.02 standard deviation is what GPT-2 uses - works well in practice!
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                # Initialize biases to zero (standard practice)
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Embedding layers also get small random initialization
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, tokens):
        """
        MY TRANSFORMER FORWARD PASS:
        tokens: (seq_len, batch) - input token IDs
        returns: (seq_len, batch, vocab_size) - logits for next token prediction
        
        This is the complete forward pass through my transformer!
        """
        # STEP 1: Convert token IDs to embeddings
        # This maps each token (integer) to a rich vector representation
        x = self.token_embedding(tokens)  # (seq_len, batch, d_model)
        
        # STEP 2: Pass through all transformer blocks sequentially
        # Each block does: norm -> attention -> residual, norm -> ffn -> residual
        for block in self.blocks:
            x = block(x)  # x keeps the same shape throughout
        
        # STEP 3: Final layer normalization
        # Helps with training stability and final predictions
        x = self.final_norm(x)
        
        # STEP 4: Project back to vocabulary space
        # This gives us logits (unnormalized probabilities) for each token in vocab
        logits = self.output_proj(x)  # (seq_len, batch, vocab_size)
        
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
    """
    MY IMPLEMENTATION for this: Nucleus sampling (top-p sampling)
    This is WAY better than greedy decoding! Instead of always picking the most likely token,
    we pick from the smallest set of tokens that covers p% of the probability mass.
    
    Why is this better? Greedy can be repetitive, random can be nonsensical,
    but nucleus gives us a good balance of coherent and diverse text!
    
    Example: if p=0.9, we might need the top 20 tokens to cover 90% probability,
    so we sample randomly from just those 20 (not the full 50k vocab!)
    """
    
    # Handle edge cases first (learned to always check these!)
    if p >= 1.0:
        # p=1.0 means sample from full distribution (completely random)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()
    
    if p <= 0.0:
        # p=0.0 means greedy (just pick the most likely)
        return torch.argmax(logits).item()
    
    # STEP 1: Convert raw logits to probabilities using softmax
    probs = F.softmax(logits, dim=-1)
    
    # STEP 2: Sort tokens by probability (highest first)
    # This gives us sorted_probs and the corresponding original indices
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # STEP 3: Calculate cumulative probability 
    # cumsum_probs[i] = sum of probabilities from index 0 to i
    cumsum_probs = torch.cumsum(sorted_probs, dim=0)
    
    # STEP 4: Find where cumulative probability first exceeds p
    cutoff_mask = cumsum_probs >= p
    
    if cutoff_mask.any():
        # Find the first index where cumsum >= p
        cutoff_idx = torch.where(cutoff_mask)[0][0].item()
    else:
        # Safety fallback (shouldn't happen unless p is super tiny)
        cutoff_idx = 0
    
    # STEP 5: Keep only the "nucleus" - top tokens that sum to p% probability
    top_probs = sorted_probs[:cutoff_idx + 1]
    top_indices = sorted_indices[:cutoff_idx + 1]
    
    # STEP 6: Renormalize so probabilities sum to 1 again
    # (since we removed the tail of the distribution)
    top_probs = top_probs / top_probs.sum()
    
    # STEP 7: Sample from this truncated distribution
    sampled_idx = torch.multinomial(top_probs, num_samples=1).item()
    chosen_token = top_indices[sampled_idx].item()
    
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
                    prompt="Once upon a"):
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
    num_epochs = 3
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
        d_model=embed_size,
        n_heads=8,
        n_blocks=4
    ).to(device)

    # MY MODEL DICTIONARY - I enabled all three models for comparison
    # Before this change, some models might have been disabled or incomplete 
    models = {
        "kgram_mlp_seq": kgram_model,    # My improved K-gram model with embeddings
        "lstm_seq": lstm_model,          # LSTM baseline for comparison
        "transformer": transformer,      # My complete transformer implementation!
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
