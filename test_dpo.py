#!/usr/bin/env python3
"""
Test script for DPO (Direct Preference Optimization) implementation.

This script provides examples of how to run DPO training with the extended pico-llm.

Usage examples:

1. Basic DPO training (first run normal training, then DPO):
   python pico-llm/pico_llm.py --epochs 2 --batch_size 8 --block_size 128 --embed_size 256 --output_dir outputs_base
   python pico-llm/pico_llm.py --dpo_training --epochs 2 --dpo_epochs 2 --dpo_beta 0.1 --num_preference_pairs 50 --output_dir outputs_dpo

2. DPO training with specific reference model:
   python pico-llm/pico_llm.py --dpo_training --reference_model_path outputs_base/kvcache_transformer_final_weights.pt --dpo_beta 0.2 --output_dir outputs_dpo_ref

3. Quick test on simple sequences:
   python pico-llm/pico_llm.py --tinystories_weight 0.0 --input_files 3seqs.txt --dpo_training --epochs 1 --dpo_epochs 1 --block_size 64 --embed_size 128 --output_dir outputs_dpo_test

"""

import os
import sys

def run_basic_dpo_demo():
    """
    Run a basic DPO demonstration with minimal settings.
    """
    print("="*60)
    print("RUNNING BASIC DPO DEMONSTRATION")
    print("="*60)
    
    # Step 1: Train base model (quick training)
    print("\nStep 1: Training base model...")
    base_cmd = [
        sys.executable, "pico-llm/pico_llm.py",
        "--epochs", "2",
        "--batch_size", "8", 
        "--block_size", "128",
        "--embed_size", "256",
        "--max_steps_per_epoch", "20",
        "--learning_rate", "1e-3",
        "--output_dir", "outputs_dpo_demo_base",
        "--device_id", "cpu"  # Use CPU for demo to ensure compatibility
    ]
    
    print("Running command:", " ".join(base_cmd))
    result = os.system(" ".join(base_cmd))
    
    if result != 0:
        print("❌ Base training failed!")
        return False
    
    # Step 2: Run DPO training
    print("\n" + "="*50)
    print("Step 2: Running DPO training...")
    dpo_cmd = [
        sys.executable, "pico-llm/pico_llm.py",
        "--dpo_training",
        "--reference_model_path", "outputs_dpo_demo_base/kvcache_transformer_final_weights.pt",
        "--dpo_epochs", "2", 
        "--dpo_beta", "0.1",
        "--num_preference_pairs", "20",  # Small number for demo
        "--output_dir", "outputs_dpo_demo",
        "--device_id", "cpu",
        "--epochs", "1"  # Still need base training params even with --dpo_training
    ]
    
    print("Running command:", " ".join(dpo_cmd))
    result = os.system(" ".join(dpo_cmd))
    
    if result != 0:
        print("❌ DPO training failed!")
        return False
    
    print("\n" + "="*60)
    print("✅ DPO DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("✅ Check outputs_dpo_demo/ for results:")
    print("  - transformer_dpo_final_weights.pt (DPO-trained model)")
    print("  - dpo_loss_logs.json (DPO training losses)")
    print("  - Generation comparisons were printed during training")
    print("="*60)
    
    return True

def run_simple_sequence_dpo():
    """
    Run DPO on simple sequences (3seqs.txt) for quick testing.
    """
    print("="*60)
    print("RUNNING SIMPLE SEQUENCE DPO TEST")  
    print("="*60)
    
    cmd = [
        sys.executable, "pico-llm/pico_llm.py",
        "--tinystories_weight", "0.0",
        "--input_files", "pico-llm/3seqs.txt", 
        "--dpo_training",
        "--epochs", "1",
        "--dpo_epochs", "1",
        "--block_size", "64",
        "--embed_size", "128", 
        "--batch_size", "4",
        "--max_steps_per_epoch", "10",
        "--num_preference_pairs", "10",
        "--output_dir", "outputs_dpo_3seqs",
        "--prompt", "0 1 2",
        "--device_id", "cpu"
    ]
    
    print("Running command:", " ".join(cmd))
    result = os.system(" ".join(cmd))
    
    if result == 0:
        print("✅ Simple sequence DPO test completed successfully!")
    else:
        print("❌ Simple sequence DPO test failed!")
    
    return result == 0

def print_usage():
    """
    Print usage instructions for DPO.
    """
    print(__doc__)

if __name__ == "__main__":
    print("DPO Test Script for Pico-LLM")
    print("="*40)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print_usage()
        sys.exit(0)
    
    # Check if pico_llm.py exists
    if not os.path.exists("pico-llm/pico_llm.py"):
        print("❌ Error: pico-llm/pico_llm.py not found!")
        print("Please run this script from the correct directory.")
        sys.exit(1)
    
    print("Available tests:")
    print("1. Basic DPO demo (full pipeline)")
    print("2. Simple sequence DPO (quick test)")
    print("3. Print usage examples")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        success = run_basic_dpo_demo()
    elif choice == "2": 
        success = run_simple_sequence_dpo()
    elif choice == "3":
        print_usage()
        success = True
    else:
        print("Invalid choice!")
        success = False
    
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n💥 Test failed!")
        sys.exit(1)
