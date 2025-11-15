#!/usr/bin/env python3
# visualize_analysis.py

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import tiktoken
from collections import defaultdict

################################################################################
# Args
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize attention patterns and monosemanticity analysis")
    parser.add_argument("--save_dir", type=str, default="./saved_models",
                        help="Directory containing saved attention and activation records")
    parser.add_argument("--attention_file", type=str, default="attention_records.pt",
                        help="Filename of attention records")
    parser.add_argument("--activation_file", type=str, default="activation_records.pt",
                        help="Filename of activation records")
    parser.add_argument("--output_dir", type=str, default="./analysis_output",
                        help="Directory to save visualization outputs")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Which recorded sample to visualize")
    parser.add_argument("--max_seq_len", type=int, default=50,
                        help="Maximum sequence length to display in heatmaps")
    parser.add_argument("--n_top_neurons", type=int, default=20,
                        help="Number of top neurons to analyze for monosemanticity")
    parser.add_argument("--n_examples", type=int, default=10,
                        help="Number of examples to use for feature analysis")
    return parser.parse_args()

################################################################################
# Attention Visualization (AlphaFold-style)
################################################################################

def plot_attention_heads(attention_records, sample_idx, output_dir, max_seq_len=50, enc=None):
    """
    Create AlphaFold-style attention heatmaps for each head in each layer.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group records by block
    records_by_block = defaultdict(list)
    for rec in attention_records:
        records_by_block[rec['block']].append(rec)
    
    n_blocks = len(records_by_block)
    
    for block_idx in range(n_blocks):
        block_records = records_by_block[block_idx]
        if sample_idx >= len(block_records):
            print(f"Warning: Sample {sample_idx} not available for block {block_idx}")
            continue
        
        rec = block_records[sample_idx]
        att_per_head = rec['att_per_head']  # [batch, n_heads, seq_len, seq_len]
        tokens = rec['tokens']  # [batch, seq_len]
        
        batch_size, n_heads, seq_len, _ = att_per_head.shape
        
        # Take first item in batch and truncate to max_seq_len
        seq_len_display = min(seq_len, max_seq_len)
        att_display = att_per_head[0, :, :seq_len_display, :seq_len_display].numpy()
        
        # Get token strings if available
        token_labels = None
        if tokens is not None and enc is not None:
            token_ids = tokens[0, :seq_len_display].numpy()
            token_labels = [enc.decode([int(tid)]) for tid in token_ids]
            # Clean up token labels for display
            token_labels = [t.replace('\n', '\\n').replace('\t', '\\t')[:10] for t in token_labels]
        
        # Create figure with subplots for each head
        n_cols = min(4, n_heads)
        n_rows = (n_heads + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        fig.suptitle(f'Block {block_idx} - Attention Heads (Sample {sample_idx})', fontsize=16)
        
        if n_heads == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for head_idx in range(n_heads):
            ax = axes[head_idx]
            
            # Plot heatmap
            im = ax.imshow(att_display[head_idx], cmap='viridis', aspect='auto', 
                          interpolation='nearest', vmin=0, vmax=att_display[head_idx].max())
            
            ax.set_title(f'Head {head_idx}')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            
            # Add token labels if available (only for smaller sequences)
            if token_labels is not None and seq_len_display <= 20:
                ax.set_xticks(range(len(token_labels)))
                ax.set_yticks(range(len(token_labels)))
                ax.set_xticklabels(token_labels, rotation=90, fontsize=8)
                ax.set_yticklabels(token_labels, fontsize=8)
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for idx in range(n_heads, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'attention_block{block_idx}_sample{sample_idx}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")

def plot_mean_attention(attention_records, sample_idx, output_dir, max_seq_len=50, enc=None):
    """
    Plot mean attention across all heads for each layer.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group records by block
    records_by_block = defaultdict(list)
    for rec in attention_records:
        records_by_block[rec['block']].append(rec)
    
    n_blocks = len(records_by_block)
    
    fig, axes = plt.subplots(1, n_blocks, figsize=(5*n_blocks, 5))
    if n_blocks == 1:
        axes = [axes]
    
    fig.suptitle(f'Mean Attention Across Heads (Sample {sample_idx})', fontsize=16)
    
    for block_idx in range(n_blocks):
        block_records = records_by_block[block_idx]
        if sample_idx >= len(block_records):
            continue
        
        rec = block_records[sample_idx]
        att_mean = rec['att_mean']  # [batch, seq_len, seq_len]
        tokens = rec['tokens']
        
        seq_len = att_mean.shape[1]
        seq_len_display = min(seq_len, max_seq_len)
        att_display = att_mean[0, :seq_len_display, :seq_len_display].numpy()
        
        # Get token labels
        token_labels = None
        if tokens is not None and enc is not None:
            token_ids = tokens[0, :seq_len_display].numpy()
            token_labels = [enc.decode([int(tid)]) for tid in token_ids]
            token_labels = [t.replace('\n', '\\n').replace('\t', '\\t')[:10] for t in token_labels]
        
        ax = axes[block_idx]
        im = ax.imshow(att_display, cmap='viridis', aspect='auto', 
                      interpolation='nearest', vmin=0, vmax=att_display.max())
        ax.set_title(f'Block {block_idx}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        
        if token_labels is not None and seq_len_display <= 20:
            ax.set_xticks(range(len(token_labels)))
            ax.set_yticks(range(len(token_labels)))
            ax.set_xticklabels(token_labels, rotation=90, fontsize=8)
            ax.set_yticklabels(token_labels, fontsize=8)
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'attention_mean_sample{sample_idx}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_attention_patterns_summary(attention_records, output_dir, n_samples=5):
    """
    Create summary visualizations of attention patterns across multiple samples.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by block and head
    records_by_block = defaultdict(list)
    for rec in attention_records:
        records_by_block[rec['block']].append(rec)
    
    for block_idx, block_records in records_by_block.items():
        n_samples_actual = min(n_samples, len(block_records))
        
        # Average attention patterns across samples
        all_att_per_head = []
        for i in range(n_samples_actual):
            att = block_records[i]['att_per_head'][0]  # Take first batch item
            all_att_per_head.append(att.numpy())
        
        avg_att = np.mean(all_att_per_head, axis=0)  # [n_heads, seq_len, seq_len]
        n_heads = avg_att.shape[0]
        
        # Plot averaged attention patterns
        fig, axes = plt.subplots(1, n_heads, figsize=(4*n_heads, 4))
        if n_heads == 1:
            axes = [axes]
        
        fig.suptitle(f'Block {block_idx} - Average Attention Patterns (n={n_samples_actual})', fontsize=14)
        
        for head_idx in range(n_heads):
            ax = axes[head_idx]
            im = ax.imshow(avg_att[head_idx], cmap='viridis', aspect='auto')
            ax.set_title(f'Head {head_idx}')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'attention_averaged_block{block_idx}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")

################################################################################
# Monosemanticity Analysis
################################################################################

def analyze_neuron_activations(activation_records, output_dir, n_top_neurons=20, n_examples=10):
    """
    Analyze and visualize individual neuron activations for monosemanticity.
    This identifies neurons that activate strongly for specific patterns.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by block
    records_by_block = defaultdict(list)
    for rec in activation_records:
        records_by_block[rec['block']].append(rec)
    
    for block_idx, block_records in records_by_block.items():
        n_examples_actual = min(n_examples, len(block_records))
        
        # Collect all post-GELU activations (neuron activations)
        all_activations = []
        for i in range(n_examples_actual):
            act = block_records[i]['post_gelu']  # [seq_len, batch, hidden_dim]
            # Reshape to [tokens, neurons]
            seq_len, batch_size, hidden_dim = act.shape
            act_flat = act.reshape(-1, hidden_dim).numpy()  # [seq_len*batch, hidden_dim]
            all_activations.append(act_flat)
        
        # Concatenate all activations
        all_activations = np.concatenate(all_activations, axis=0)  # [n_tokens, n_neurons]
        n_tokens, n_neurons = all_activations.shape
        
        print(f"\nBlock {block_idx}: Analyzing {n_tokens} tokens across {n_neurons} neurons")
        
        # Find neurons with highest max activation
        max_activations = np.max(all_activations, axis=0)
        top_neuron_indices = np.argsort(max_activations)[-n_top_neurons:][::-1]
        
        # Plot distribution of max activations per neuron
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Block {block_idx} - Neuron Activation Analysis', fontsize=14)
        
        # 1. Distribution of max activations
        ax = axes[0, 0]
        ax.hist(max_activations, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Max Activation Value')
        ax.set_ylabel('Number of Neurons')
        ax.set_title('Distribution of Max Activations per Neuron')
        ax.axvline(np.percentile(max_activations, 95), color='r', linestyle='--', 
                   label='95th percentile')
        ax.legend()
        
        # 2. Top neurons activation patterns
        ax = axes[0, 1]
        top_activations = all_activations[:, top_neuron_indices[:10]]
        im = ax.imshow(top_activations.T, aspect='auto', cmap='hot', interpolation='nearest')
        ax.set_xlabel('Token Index')
        ax.set_ylabel('Neuron Index (Top 10)')
        ax.set_title('Activation Patterns of Top 10 Neurons')
        plt.colorbar(im, ax=ax)
        
        # 3. Sparsity analysis - what % of neurons are active per token
        threshold = np.percentile(all_activations, 95)
        active_neurons_per_token = np.sum(all_activations > threshold, axis=1)
        ax = axes[1, 0]
        ax.hist(active_neurons_per_token, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Number of Active Neurons')
        ax.set_ylabel('Number of Tokens')
        ax.set_title(f'Sparsity: Active Neurons per Token (threshold={threshold:.2f})')
        
        # 4. Activation frequency - how often each neuron activates strongly
        neuron_activation_freq = np.sum(all_activations > threshold, axis=0)
        ax = axes[1, 1]
        ax.hist(neuron_activation_freq, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Activation Frequency')
        ax.set_ylabel('Number of Neurons')
        ax.set_title('Distribution of Neuron Activation Frequencies')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'monosemanticity_block{block_idx}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
        
        # Create detailed plot for top neurons
        plot_top_neuron_details(all_activations, top_neuron_indices[:n_top_neurons], 
                                block_idx, output_dir)

def plot_top_neuron_details(all_activations, top_neuron_indices, block_idx, output_dir):
    """
    Create detailed visualizations for top neurons to understand what they're detecting.
    """
    n_neurons = len(top_neuron_indices)
    n_cols = 4
    n_rows = (n_neurons + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    fig.suptitle(f'Block {block_idx} - Top {n_neurons} Neuron Activation Patterns', fontsize=14)
    
    if n_neurons == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, neuron_idx in enumerate(top_neuron_indices):
        ax = axes[i]
        activations = all_activations[:, neuron_idx]
        
        # Plot activation histogram
        ax.hist(activations, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Activation Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Neuron {neuron_idx}')
        
        # Add statistics
        mean_act = np.mean(activations)
        max_act = np.max(activations)
        sparsity = np.sum(activations > np.percentile(activations, 95)) / len(activations)
        ax.axvline(mean_act, color='r', linestyle='--', label=f'Mean: {mean_act:.2f}')
        ax.text(0.95, 0.95, f'Max: {max_act:.2f}\nSparsity: {sparsity:.2%}',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(n_neurons, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'top_neurons_block{block_idx}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def analyze_feature_correlation(activation_records, output_dir, n_examples=10):
    """
    Analyze correlation between different neurons to identify potential polysemantic features.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    records_by_block = defaultdict(list)
    for rec in activation_records:
        records_by_block[rec['block']].append(rec)
    
    for block_idx, block_records in records_by_block.items():
        n_examples_actual = min(n_examples, len(block_records))
        
        # Collect activations
        all_activations = []
        for i in range(n_examples_actual):
            act = block_records[i]['post_gelu']
            seq_len, batch_size, hidden_dim = act.shape
            act_flat = act.reshape(-1, hidden_dim).numpy()
            all_activations.append(act_flat)
        
        all_activations = np.concatenate(all_activations, axis=0)
        
        # Compute correlation matrix (sample subset of neurons for visualization)
        n_neurons_sample = min(100, all_activations.shape[1])
        neuron_indices = np.random.choice(all_activations.shape[1], n_neurons_sample, replace=False)
        activations_sample = all_activations[:, neuron_indices]
        
        correlation_matrix = np.corrcoef(activations_sample.T)
        
        # Plot correlation matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', 
                      vmin=-1, vmax=1, interpolation='nearest')
        ax.set_title(f'Block {block_idx} - Neuron Correlation Matrix ({n_neurons_sample} neurons)')
        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Neuron Index')
        plt.colorbar(im, ax=ax, label='Correlation')
        
        output_path = os.path.join(output_dir, f'neuron_correlation_block{block_idx}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")

################################################################################
# Main
################################################################################

def main():
    args = parse_args()
    
    # Set up encoder for token decoding
    enc = tiktoken.get_encoding("gpt2")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load attention records
    attention_path = os.path.join(args.save_dir, args.attention_file)
    if os.path.exists(attention_path):
        print(f"\nLoading attention records from: {attention_path}")
        attention_records = torch.load(attention_path)
        print(f"Loaded {len(attention_records)} attention records")
        
        # Create attention visualizations
        print("\n" + "="*80)
        print("Generating Attention Visualizations (AlphaFold-style)")
        print("="*80)
        
        # Individual head attention patterns
        plot_attention_heads(attention_records, args.sample_idx, 
                           os.path.join(args.output_dir, "attention_heads"),
                           max_seq_len=args.max_seq_len, enc=enc)
        
        # Mean attention patterns
        plot_mean_attention(attention_records, args.sample_idx,
                          os.path.join(args.output_dir, "attention_mean"),
                          max_seq_len=args.max_seq_len, enc=enc)
        
        # Summary across samples
        plot_attention_patterns_summary(attention_records, 
                                       os.path.join(args.output_dir, "attention_summary"),
                                       n_samples=args.n_examples)
    else:
        print(f"Warning: Attention file not found: {attention_path}")
    
    # Load activation records
    activation_path = os.path.join(args.save_dir, args.activation_file)
    if os.path.exists(activation_path):
        print(f"\nLoading activation records from: {activation_path}")
        activation_records = torch.load(activation_path)
        print(f"Loaded {len(activation_records)} activation records")
        
        # Create monosemanticity analysis
        print("\n" + "="*80)
        print("Generating Monosemanticity Analysis")
        print("="*80)
        
        analyze_neuron_activations(activation_records, 
                                  os.path.join(args.output_dir, "monosemanticity"),
                                  n_top_neurons=args.n_top_neurons,
                                  n_examples=args.n_examples)
        
        analyze_feature_correlation(activation_records,
                                   os.path.join(args.output_dir, "feature_correlation"),
                                   n_examples=args.n_examples)
    else:
        print(f"Warning: Activation file not found: {activation_path}")
    
    print("\n" + "="*80)
    print(f"All visualizations saved to: {args.output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()