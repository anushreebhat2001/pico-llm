## Core Tasks to implement - 

## 1. Sanity Check (Baseline Run)

Sanity check that you are able to run the code, which by default will only run an LSTM on TinyStories. It
is possible that the code is too slow or runs out of memory for you: consider using an aggressive memorysaving command-line argument such as “--block size 32”, and also using the simplified sequence
data via “--tinystories weight 0.0 --input files 3seqs.txt --prompt "0 1 2 3 4"”. Make
sure you understand the code, in particular the routine torch.nn.Embedding, which has not been
discussed in class; why is that routine useful?

### Command
```bash
python3 pico-llm.py \
  --tinystories_weight 0.0 \
  --input_files 3seqs.txt \
  --prompt "0 1 2 3 4" \
  --block_size 32 \
  --embed_size 128 \
  --max_steps_per_epoch 10 \
  --device_id cpu \
  --output_dir outputs_embedding
```

```bash
python3 pico-llm.py \
  --tinystories_weight 1.0 \
  --prompt "Once upon a time" \
  --block_size 32 \
  --embed_size 128 \
  --max_steps_per_epoch 10 \
  --device_id cpu \
  --output_dir outputs_embedding
```

For official training - 

```bash
python3 pico-llm.py \
  --tinystories_weight 0.0 \
  --input_files 3seqs.txt \
  --prompt "0 1 2 3 4" \
  --block_size 1024 \
  --embed_size 128 \
  --kgram_k 3 \
  --kgram_chunk_size 1 \
  --test_fraction 0.1 \
  --epochs 15 \
  --max_steps_per_epoch 210 \
  --batch_size 16 \
  --learning_rate 3e-4 \
  --device_id cuda:0 \
  --output_dir outputs_3seqs_fullpattern
```

```bash

python3 pico-llm.py \
  --tinystories_weight 1.0 \
  --prompt "Once upon a time" \
  --block_size 512 \
  --embed_size 512 \
  --kgram_k 3 \
  --max_steps_per_epoch 750 \
  --kgram_chunk_size 16 \
  --batch_size 16 \
  --epochs 10 \
  --learning_rate 3e-4 \
  --test_fraction 0.1 \
  --device_id cuda:0 \
  --output_dir outputs_tinystories_full
```
