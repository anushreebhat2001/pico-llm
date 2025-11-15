import argparse
import time
import random
import math
import json
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datasets import load_dataset
import tiktoken

################################################################################
# Args
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Transformer training on Wiki with attention & activation recording")
    parser.add_argument("--input_files", nargs="*", default=None)
    parser.add_argument("--max_steps_per_epoch", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--embed_size", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--device_id", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("--save_name", type=str, default="transformer_model")
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--train_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--prompt", type=str, default="Quantum mechanics is")
    parser.add_argument("--record_attn", action="store_true")
    parser.set_defaults(record_attn=False)
    parser.add_argument("--record_activations", action="store_true")
    parser.set_defaults(record_activations=False)
    parser.add_argument("--record_interval", type=int, default=100, 
                        help="Record attention/activations every N steps")
    parser.add_argument("--use_pos_emb", action="store_true")
    parser.set_defaults(use_pos_emb=True)
    parser.add_argument("--pre_norm", action="store_true")
    parser.set_defaults(pre_norm=True)
    parser.add_argument("--test_split_ratio", type=float, default=0.1)
    parser.add_argument("--save_losses_name", type=str, default="loss_tables.json")
    parser.add_argument("--save_attn_name", type=str, default="attention_records.pt")
    parser.add_argument("--save_activations_name", type=str, default="activation_records.pt")
    parser.add_argument("--save_state_dict_name", type=str, default="model_state_dict.pt")
    parser.add_argument("--save_full_model_name", type=str, default="model_full.pt")
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--sample_interval_seconds", type=int, default=60)
    parser.add_argument("--train_subset_size", type=int, default=20000)
    parser.add_argument("--max_record_samples", type=int, default=100,
                        help="Maximum number of samples to record for analysis")
    args = parser.parse_args()
    return args

################################################################################
# Save/load utilities
################################################################################

def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    print(f"Saved JSON: {path}")

def save_torch(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(obj, path)
    print(f"Saved torch object: {path}")

################################################################################
# Dataset handling
################################################################################

class MixedSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, wiki_seqs, other_seqs, p_wiki=1.0):
        super().__init__()
        self.wiki_seqs = wiki_seqs
        self.other_seqs = other_seqs
        self.p_wiki = p_wiki
        self.has_wiki = len(wiki_seqs) > 0
        self.has_other = len(other_seqs) > 0
        self.total_length = len(wiki_seqs) + len(other_seqs)
        if self.total_length == 0:
            raise ValueError("No data found!")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        r = random.random()
        if self.has_wiki and self.has_other:
            if r < self.p_wiki:
                seq = self.wiki_seqs[random.randint(0, len(self.wiki_seqs)-1)]
            else:
                seq = self.other_seqs[random.randint(0, len(self.other_seqs)-1)]
        elif self.has_wiki:
            seq = self.wiki_seqs[random.randint(0, len(self.wiki_seqs)-1)]
        else:
            seq = self.other_seqs[random.randint(0, len(self.other_seqs)-1)]
        return torch.tensor(seq, dtype=torch.long)

def seq_collate_fn(batch):
    max_len = max(len(seq) for seq in batch)
    batch_size = len(batch)
    padded = torch.zeros(max_len, batch_size, dtype=torch.long)
    for i, seq in enumerate(batch):
        L = seq.size(0)
        padded[:L, i] = seq
    return padded

################################################################################
# Loss helper
################################################################################

def compute_next_token_loss(logits, tokens):
    seq_len = logits.shape[0]
    if seq_len < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    preds = logits[:-1, :, :].reshape(-1, logits.size(-1))
    gold = tokens[1:, :].reshape(-1)
    return F.cross_entropy(preds, gold)

################################################################################
# Transformer blocks, attention, RMSNorm
################################################################################

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        ms = x.pow(2).mean(-1, keepdim=True)
        return x / torch.sqrt(ms + self.eps) * self.weight

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, record_attn=False):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.record_attn = record_attn
        self.attention_records = []

    def forward(self, x, tokens_batch=None):
        seq_len, batch_size, _ = x.shape
        x_b = x.transpose(0,1)
        q = self.q_proj(x_b).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(x_b).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x_b).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), 1).bool()
        att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Record both per-head and mean attention patterns
        if self.record_attn:
            att_per_head = att.detach().cpu()  # [batch, n_heads, seq_len, seq_len]
            att_mean = att.mean(dim=1).detach().cpu()  # [batch, seq_len, seq_len]
            tokens_cpu = tokens_batch.transpose(0,1).detach().cpu() if tokens_batch is not None else None
            self.attention_records.append({
                'att_per_head': att_per_head,  # Individual head patterns
                'att_mean': att_mean,  # Mean across heads
                'tokens': tokens_cpu
            })
        
        y = (att @ v).transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)
        y = self.resid_dropout(self.c_proj(y))
        return y.transpose(0,1)

class MLP(nn.Module):
    def __init__(self, d_model, dropout=0.1, record_activations=False):
        super().__init__()
        self.fc = nn.Linear(d_model, 4*d_model)
        self.proj = nn.Linear(4*d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.record_activations = record_activations
        self.activation_records = []
    
    def forward(self, x):
        # Record pre-activation (input to MLP)
        if self.record_activations:
            pre_act = x.detach().cpu()
        
        hidden = F.gelu(self.fc(x))
        
        # Record post-GELU activations (neuron activations)
        if self.record_activations:
            post_gelu = hidden.detach().cpu()
            self.activation_records.append({
                'pre_mlp': pre_act,
                'post_gelu': post_gelu
            })
        
        return self.dropout(self.proj(hidden))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, pre_norm=True, record_attn=False, record_activations=False):
        super().__init__()
        self.pre_norm = pre_norm
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, record_attn=record_attn)
        self.mlp = MLP(d_model, record_activations=record_activations)
    
    def forward(self, x, tokens_seq=None):
        if self.pre_norm:
            x = x + self.attn(self.ln1(x), tokens_batch=tokens_seq)
            x = x + self.mlp(self.ln2(x))
        else:
            x = self.ln1(x + self.attn(x, tokens_batch=tokens_seq))
            x = self.ln2(x + self.mlp(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_blocks=4, max_seq_len=1024,
                 use_pos_emb=True, pre_norm=True, record_attn=False, record_activations=False):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model) if use_pos_emb else None
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, pre_norm, record_attn, record_activations) 
            for _ in range(n_blocks)
        ])
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.max_seq_len = max_seq_len
        self.record_attn = record_attn
        self.record_activations = record_activations
        self.use_pos_emb = use_pos_emb

    def forward(self, tokens_seq):
        seq_len, batch_size = tokens_seq.shape
        positions = torch.arange(seq_len, device=tokens_seq.device).unsqueeze(1).expand(seq_len, batch_size)
        x = self.token_emb(tokens_seq) + (self.pos_emb(positions) if self.use_pos_emb else 0)
        
        # Clear previous records
        if self.record_attn or self.record_activations:
            for blk in self.blocks:
                blk.attn.attention_records = []
                blk.mlp.activation_records = []
        
        for blk in self.blocks:
            x = blk(x, tokens_seq=tokens_seq)
        
        logits = self.lm_head(self.ln_f(x))
        return logits
    
    def get_attention_records(self):
        """Extract all attention records from all blocks"""
        records = []
        for i, blk in enumerate(self.blocks):
            for rec in blk.attn.attention_records:
                records.append({
                    'block': i,
                    'att_per_head': rec['att_per_head'],
                    'att_mean': rec['att_mean'],
                    'tokens': rec['tokens']
                })
        return records
    
    def get_activation_records(self):
        """Extract all activation records from all blocks"""
        records = []
        for i, blk in enumerate(self.blocks):
            for rec in blk.mlp.activation_records:
                records.append({
                    'block': i,
                    'pre_mlp': rec['pre_mlp'],
                    'post_gelu': rec['post_gelu']
                })
        return records

################################################################################
# Sampling
################################################################################

def generate_text(model, enc, init_text, max_new_tokens=20, device="cpu", top_p=None):
    was_training = model.training
    model.eval()
    context_tokens = enc.encode(init_text)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)
            logits = model(seq_tensor)
            next_logits = logits[-1, 0, :]
            if top_p is None:
                chosen = torch.argmax(next_logits).item()
            else:
                probs = F.softmax(next_logits, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=0)
                k = max(torch.searchsorted(cum_probs, torch.tensor(top_p, device=device)).item(),1)
                topk_probs = sorted_probs[:k]; topk_idx = sorted_idx[:k]
                topk_probs /= topk_probs.sum()
                chosen = topk_idx[torch.multinomial(topk_probs,1).item()].item()
            context_tokens.append(chosen)
    model.train(was_training)
    return enc.decode(context_tokens)

################################################################################
# Training loop
################################################################################

def train_one_model(model, train_loader, test_loader, epochs, model_name, device,
                    lr=1e-4, log_steps=100, sample_interval=60, max_steps_per_epoch=None,
                    enc=None, prompt="Quantum mechanics is", save_dir="./saved_models",
                    save_name_base="transformer", save_losses_name="loss_tables.json", 
                    save_state_dict_name="model_state_dict.pt",
                    record_interval=100, max_record_samples=100,
                    save_attn_name="attention_records.pt",
                    save_activations_name="activation_records.pt"):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    next_sample_time = time.time() + sample_interval
    global_step = 0
    train_loss_table, test_loss_table = [], []
    
    # Storage for attention and activation records
    all_attention_records = []
    all_activation_records = []
    record_count = 0
    
    print(f"Recording strategy: Every {record_interval} steps, max {max_record_samples} samples")
    print(f"Records will be saved to disk after every epoch")

    for epoch in range(1, epochs+1):
        model.train()
        epoch_train_losses, epoch_test_losses = [], []
        step_in_epoch = 0
        test_iter = iter(test_loader)
        
        for batch_tokens in train_loader:
            step_in_epoch += 1
            global_step += 1
            batch_tokens = batch_tokens.to(device)
            
            # Check if we should record on this step
            should_record = (global_step % record_interval == 0) and (record_count < max_record_samples)
            if should_record:
                model.record_attn = True
                model.record_activations = True
                for blk in model.blocks:
                    blk.attn.record_attn = True
                    blk.mlp.record_activations = True
            
            logits = model(batch_tokens)
            loss = compute_next_token_loss(logits, batch_tokens)
            
            # Extract and save records if we were recording
            if should_record:
                attn_recs = model.get_attention_records()
                act_recs = model.get_activation_records()
                if attn_recs:
                    all_attention_records.extend(attn_recs)
                if act_recs:
                    all_activation_records.extend(act_recs)
                record_count += 1
                
                # Turn off recording
                model.record_attn = False
                model.record_activations = False
                for blk in model.blocks:
                    blk.attn.record_attn = False
                    blk.mlp.record_activations = False
                
                print(f"[{model_name}] Recorded sample {record_count}/{max_record_samples} at step {global_step}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(float(loss.item()))

            try:
                test_batch = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                test_batch = next(test_iter)
            test_batch = test_batch.to(device)
            model.eval()
            with torch.no_grad():
                test_logits = model(test_batch)
                test_loss = compute_next_token_loss(test_logits, test_batch)
            model.train()
            epoch_test_losses.append(float(test_loss.item()))

            if step_in_epoch % log_steps == 0:
                avg_recent = sum(epoch_train_losses[-log_steps:])/min(len(epoch_train_losses), log_steps)
                print(f"[{model_name}] Epoch {epoch} Step {step_in_epoch} avg_train_loss={avg_recent:.4f}")

            if enc is not None and time.time() >= next_sample_time:
                print(f"\n[{model_name}] Sample generation at epoch {epoch} step {step_in_epoch}")
                print(" Greedy:", generate_text(model, enc, prompt, max_new_tokens=20, device=device))
                next_sample_time = time.time() + sample_interval

            if max_steps_per_epoch and step_in_epoch >= max_steps_per_epoch:
                break

        train_loss_table.append(epoch_train_losses)
        test_loss_table.append(epoch_test_losses)

        # Save losses after each epoch
        save_json({'train': train_loss_table, 'test': test_loss_table},
                  os.path.join(save_dir, save_losses_name))

        # Save model state after each epoch
        save_torch(model.state_dict(),
                   os.path.join(save_dir, f"{save_name_base}_epoch{epoch}_{save_state_dict_name}"))
        
        # Save attention and activation records after each epoch
        if all_attention_records:
            save_torch(all_attention_records, 
                      os.path.join(save_dir, save_attn_name))
            print(f"  → Saved {len(all_attention_records)} attention records (accumulated through epoch {epoch})")
        if all_activation_records:
            save_torch(all_activation_records,
                      os.path.join(save_dir, save_activations_name))
            print(f"  → Saved {len(all_activation_records)} activation records (accumulated through epoch {epoch})")

    return train_loss_table, test_loss_table

################################################################################
# Main
################################################################################

def main():
    args = parse_args()
    device = torch.device(args.device_id if torch.cuda.is_available() else "cpu")
    enc = tiktoken.get_encoding("gpt2")

    # Load Wiki dataset
    wiki_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").select(range(args.train_subset_size))
    wiki_seqs = []
    for sample in wiki_dataset:
        t = sample['text'].strip()
        if t:
            toks = enc.encode(t)[:args.block_size]
            if toks:
                wiki_seqs.append(toks)

    # Custom input files
    other_seqs = []
    if args.input_files:
        for fp in args.input_files:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    toks = enc.encode(line.strip())[:args.block_size]
                    if toks:
                        other_seqs.append(toks)

    # Split train/test
    all_seqs = wiki_seqs + other_seqs
    random.shuffle(all_seqs)
    test_count = max(1, int(len(all_seqs)*args.test_split_ratio))
    test_idx = set(range(test_count))
    train_idx = set(range(test_count, len(all_seqs)))
    train_seqs = [all_seqs[i] for i in train_idx]
    test_seqs = [all_seqs[i] for i in test_idx]

    train_dataset = MixedSequenceDataset(train_seqs, [], p_wiki=1.0)
    test_dataset = MixedSequenceDataset(test_seqs, [], p_wiki=1.0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=seq_collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=seq_collate_fn)

    model = TransformerModel(vocab_size=enc.n_vocab, d_model=args.embed_size,
                             n_heads=args.n_heads, n_blocks=args.n_blocks,
                             max_seq_len=args.block_size, use_pos_emb=args.use_pos_emb,
                             pre_norm=args.pre_norm, 
                             record_attn=args.record_attn,
                             record_activations=args.record_activations).to(device)

    if args.load_model:
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        print(f"Loaded model from {args.load_model}")

    os.makedirs(args.save_dir, exist_ok=True)

    train_one_model(model, train_loader, test_loader, epochs=args.train_epochs,
                    model_name=args.save_name, device=device, lr=args.lr,
                    log_steps=args.log_steps, sample_interval=args.sample_interval_seconds,
                    max_steps_per_epoch=args.max_steps_per_epoch, enc=enc,
                    prompt=args.prompt, save_dir=args.save_dir,
                    save_name_base=args.save_name, save_losses_name=args.save_losses_name, 
                    save_state_dict_name=args.save_state_dict_name,
                    record_interval=args.record_interval,
                    max_record_samples=args.max_record_samples,
                    save_attn_name=args.save_attn_name,
                    save_activations_name=args.save_activations_name)

    # Save final model
    save_torch(model.state_dict(), os.path.join(args.save_dir, args.save_state_dict_name))
    save_torch(model, os.path.join(args.save_dir, args.save_full_model_name))
    
    print("\n" + "="*80)
    print("Training complete!")
    if args.record_attn:
        print(f"Attention records saved to: {os.path.join(args.save_dir, args.save_attn_name)}")
    if args.record_activations:
        print(f"Activation records saved to: {os.path.join(args.save_dir, args.save_activations_name)}")
    print("="*80)

if __name__ == "__main__":
    main()