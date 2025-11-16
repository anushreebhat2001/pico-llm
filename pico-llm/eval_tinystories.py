# eval_tinystories.py

import argparse
import torch
import tiktoken

# Import the SAME definitions you used for training.
# Make sure your training file is importable as `pico_llm`
# (e.g., keep a copy named pico_llm.py in the same directory).
from pico_llm import TransformerModel, generate_text


def load_model(
    weights_path,
    device="cpu",
    d_model=1024,
    n_heads=8,
    n_blocks=6,
    block_size=1024,
):
    """
    Load the trained TinyStories transformer weights.

    Hyperparameters MUST match the ones from training:
      - d_model (same as --embed_size used for training)
      - n_heads
      - n_blocks
      - block_size
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
    model.load_state_dict(state)  # should load cleanly if hparams match
    model.eval()
    return model, enc


def main():
    parser = argparse.ArgumentParser(description="Evaluate TinyStories transformer.")
    parser.add_argument(
        "--weights",
        type=str,
        default="outputs_tinystories_full/kvcache_transformer_final_weights.pt",
        help="Path to TinyStories transformer weights.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Prompt to feed into the model.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="How many new tokens to generate.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling; use None for greedy.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device, e.g. 'cpu' or 'cuda:0'.",
    )
    # Optional: override architecture if you didn't use the defaults.
    parser.add_argument(
        "--d_model",
        type=int,
        default=1024,   # set this to the embed_size you used for TinyStories
        help="Model width (must match training --embed_size).",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=8,
        help="Number of attention heads (must match training).",
    )
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=6,
        help="Number of transformer blocks (must match training).",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=1024,
        help="Context length (must match training).",
    )

    args = parser.parse_args()

    # Pick device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load model + tokenizer with matching hyperparameters
    model, enc = load_model(
        args.weights,
        device=device,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        block_size=args.block_size,
    )

    print("Prompt:")
    print(args.prompt)

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
    print("\nAnnotated (same as generated since monosemantic is off):")
    print(annotated)


if __name__ == "__main__":
    main()
