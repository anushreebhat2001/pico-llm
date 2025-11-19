# eval_3seq.py

import argparse
import torch
import tiktoken

# This must import the SAME definitions you used for training.
# If your training file is named pico-llm.py, keep a copy as pico_llm.py to import.
from pico_llm import TransformerModel, generate_text


def load_model(weights_path, device="cpu",
               d_model=128, n_heads=8, n_blocks=6, block_size=1024):
    """
    Load the trained 'kvcache_transformer' weights.

    Hyperparameters MUST match the ones from training:
      d_model (embed_size), n_heads, n_blocks, block_size.
    """
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab

    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_blocks=n_blocks,
        block_size=block_size,
    ).to(device)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)   # should load with no missing/unexpected keys now
    model.eval()
    return model, enc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="outputs_3seqs_fullpattern/kvcache_transformer_final_weights.pt",
        help="Path to kvcache_transformer final weights",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="2 4 6 8 10",
        help="Prompt to feed into the model (for 3seq world).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="How many new tokens to generate.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling; use None for greedy.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device, e.g. 'cpu' or 'cuda:0'.",
    )
    args = parser.parse_args()

    # Pick device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA not available, falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # These MUST match your training CLI: embed_size=128, n_heads=8, n_blocks=6, block_size=1024
    model, enc = load_model(
        args.weights,
        device=device,
        d_model=128,
        n_heads=8,
        n_blocks=6,
        block_size=1024,
    )

    print("Prompt:", args.prompt)
    text, annotated = generate_text(
        model,
        enc,
        init_text=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=device,
        top_p=args.top_p,
        monosemantic_info=None,
        do_monosemantic=False,
    )

    print("\nGenerated text:")
    print(text)
    print("\nAnnotated (same here since monosemantic is off):")
    print(annotated)


if __name__ == "__main__":
    main()
