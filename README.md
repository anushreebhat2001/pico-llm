## Core Tasks to implement - 

## 1. Sanity Check (Baseline Run)

Sanity check that you are able to run the code, which by default will only run an LSTM on TinyStories. It
is possible that the code is too slow or runs out of memory for you: consider using an aggressive memorysaving command-line argument such as “--block size 32”, and also using the simplified sequence
data via “--tinystories weight 0.0 --input files 3seqs.txt --prompt "0 1 2 3 4"”. Make
sure you understand the code, in particular the routine torch.nn.Embedding, which has not been
discussed in class; why is that routine useful?

### Command
```bash
python pico-llm.py \
  --tinystories_weight 0.0 \
  --input_files 3seqs.txt \
  --prompt "2 4 6 8 10" \
  --block_size 256 \
  --max_steps_per_epoch 210 \
  --device_id cpu \
  --embed_size 256 \
  --kgram_k 3 \
  --save_dir 3seq
```

```bash
python pico-llm.py \
  --tinystories_weight 1.0 \
  --prompt "Once upon a time" \
  --block_size 256 \
  --max_steps_per_epoch 300 \
  --device_id cpu \
  --embed_size 256 \
  --kgram_k 3 \
  --save_dir tinystories
```