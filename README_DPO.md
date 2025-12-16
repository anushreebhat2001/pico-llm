# DPO (Direct Preference Optimization) Extension for Pico-LLM

This document describes the DPO extension implemented for the pico-llm project as part of the final project requirements.

## What is DPO?

Direct Preference Optimization (DPO) is a method for training language models to prefer certain outputs over others without requiring a separate reward model or reinforcement learning. It directly optimizes the model to satisfy preference pairs (preferred vs rejected completions for the same prompt).

## Key Features Implemented

### 1. Preference Dataset Generation
- Automatic generation of preference pairs from prompts
- Quality-based scoring using heuristics:
  - Length completeness
  - Repetition penalty  
  - Punctuation endings
- Support for custom preference data

### 2. DPO Loss Function
- Implementation of the standard DPO objective: `L_DPO = -log(σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x))))`
- Configurable β parameter for KL regularization
- Sequence-level log probability computation

### 3. Training Pipeline
- Reference model (frozen) vs policy model (trainable) setup
- Batch processing of preference pairs
- Learning rate scheduling optimized for preference learning
- Evaluation with pre/post DPO generation comparison

## Usage

### Command Line Arguments

New DPO-specific arguments:

```bash
--dpo_training              # Enable DPO training mode
--dpo_beta 0.1              # DPO regularization parameter (default: 0.1)
--reference_model_path PATH # Path to reference model weights
--num_preference_pairs 1000 # Number of preference pairs to generate
--dpo_epochs 3              # Number of DPO training epochs
```

### Example Usage

#### 1. Basic DPO Training (Two-Stage)

First, train a base model:
```bash
python pico-llm/pico_llm.py \
  --epochs 3 \
  --batch_size 16 \
  --block_size 512 \
  --embed_size 512 \
  --output_dir outputs_base
```

Then apply DPO:
```bash
python pico-llm/pico_llm.py \
  --dpo_training \
  --reference_model_path outputs_base/kvcache_transformer_final_weights.pt \
  --dpo_epochs 3 \
  --dpo_beta 0.1 \
  --num_preference_pairs 500 \
  --output_dir outputs_dpo
```

#### 2. Quick Test on Simple Sequences

```bash
python pico-llm/pico_llm.py \
  --tinystories_weight 0.0 \
  --input_files pico-llm/3seqs.txt \
  --dpo_training \
  --epochs 1 \
  --dpo_epochs 2 \
  --block_size 64 \
  --embed_size 128 \
  --num_preference_pairs 20 \
  --prompt "0 1 2" \
  --output_dir outputs_dpo_test
```

#### 3. Running the Test Script

```bash
python test_dpo.py
```

This provides interactive options for different DPO tests.

## Implementation Details

### Architecture

The DPO implementation extends the existing pico-llm codebase with:

1. **PreferenceDataset**: PyTorch Dataset for handling preference pairs
2. **generate_preference_pairs()**: Creates preference data using multiple sampling strategies
3. **compute_dpo_loss()**: Implements the DPO objective function
4. **train_dpo_model()**: Main DPO training loop
5. **Evaluation utilities**: Pre/post DPO comparison tools

### Preference Pair Generation

The system automatically creates preference pairs by:
1. Generating multiple completions per prompt using different top-p values
2. Scoring completions based on:
   - Length and completeness
   - Repetition avoidance
   - Proper sentence endings
3. Creating pairs from high-scoring vs low-scoring completions

### Training Process

1. **Standard Training**: First train models using next-token prediction
2. **Preference Generation**: Create preference pairs from trained model
3. **DPO Training**: Fine-tune policy model using DPO loss against reference model
4. **Evaluation**: Compare pre/post DPO generations qualitatively

## Expected Results

DPO training should result in:

- **Improved Generation Quality**: Less repetitive, more coherent outputs
- **Better Preference Satisfaction**: Model learns to prefer higher-quality completions
- **Maintained Fluency**: Core language modeling capabilities preserved
- **Alignment**: Outputs more aligned with quality preferences

## Evaluation Metrics

The implementation tracks:

- **DPO Loss**: Primary optimization objective
- **Generation Comparisons**: Qualitative before/after examples
- **Preference Satisfaction**: How well the model satisfies preference pairs
- **Diversity vs Quality Trade-offs**: Balance between creative and preferred outputs

## Files Added/Modified

### New Files:
- `README_DPO.md` - This documentation
- `test_dpo.py` - Interactive testing script

### Modified Files:
- `pico-llm/pico_llm.py` - Core DPO implementation

### New Functions Added:
- `PreferenceDataset` class
- `generate_preference_pairs()`
- `compute_dpo_loss()`
- `compute_sequence_logprobs()`
- `train_dpo_model()`
- `dpo_collate_fn()`

## Technical Notes

### Memory Considerations
- DPO requires storing both reference and policy models in memory
- Preference pair generation can be memory-intensive for large datasets
- Batch sizes are typically smaller for DPO training

### Hyperparameter Guidelines
- **β (beta)**: 0.1-0.3 typical range. Higher values = stronger preference enforcement
- **Learning Rate**: 1e-4 to 1e-5 (lower than standard training)
- **Epochs**: 2-5 epochs usually sufficient
- **Preference Pairs**: 100-1000 pairs for small models, more for larger ones

### Performance
- DPO training is typically faster than RLHF
- No need for separate reward model training
- Can run on CPU for small models (GPU recommended for larger ones)

## Troubleshooting

### Common Issues:
1. **Memory Errors**: Reduce batch size, preference pairs, or model size
2. **Training Instability**: Lower learning rate or β parameter
3. **Poor Quality Preferences**: Improve preference generation heuristics
4. **Generation Degradation**: Check β parameter (may be too high)

### Debug Tips:
- Monitor DPO loss convergence
- Check preference pair quality manually
- Compare reference vs policy model outputs
- Use smaller models for initial testing

## Future Extensions

Potential improvements:
- Human preference collection interface  
- More sophisticated preference generation strategies
- Integration with reward models
- Multi-objective preference optimization
- Constitutional AI integration

## Citation

This implementation follows the DPO methodology from:
- Rafailov et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023)

## Support

For questions about the DPO implementation:
1. Check this documentation
2. Run `python test_dpo.py --help` for examples
3. Review the code comments in `pico_llm.py`
4. Test with small configurations first
