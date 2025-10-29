## Core Tasks to implement - 

## 1. Sanity Check (Baseline Run)

Sanity check that you are able to run the code, which by default will only run an LSTM on TinyStories. It
is possible that the code is too slow or runs out of memory for you: consider using an aggressive memorysaving command-line argument such as “--block size 32”, and also using the simplified sequence
data via “--tinystories weight 0.0 --input files 3seqs.txt --prompt "0 1 2 3 4"”. Make
sure you understand the code, in particular the routine torch.nn.Embedding, which has not been
discussed in class; why is that routine useful?

### Command
```bash
python pico-llm.py --tinystories_weight 0.0 \
    --input_files 3seqs.txt \
    --prompt "0 1 2 3 4" \
    --block_size 32 \
    --max_steps_per_epoch 5 \
    --device_id cpu
```

Output - 

```
Using device: cpu, block_size=32, kgram_k=3, chunk_size=1, embed_size=1024
TinyStories weight=0 => skipping TinyStories.
Vocab size: 50257
Reading custom text file: 3seqs.txt
Custom input files: 3333 sequences loaded.

=== Training model: lstm_seq ===

[lstm_seq] Generating sample text (greedy) at epoch=1, step=1...
 Greedy Sample: 0 1 2 3 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 1310
 Annotated: 0 1 2 3 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 1310

[lstm_seq] Generating sample text (top-p=0.95) at epoch=1, step=1...
 Top-p (p=0.95) Sample: 0 1 2 3 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 1310
 Annotated: 0 1 2 3 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 1310

[lstm_seq] Generating sample text (top-p=1.0) at epoch=1, step=1...
 Top-p (p=1.0) Sample: 0 1 2 3 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 1310
 Annotated: 0 1 2 3 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 1310

[lstm_seq] Reached max_steps_per_epoch=5, ending epoch 1 early.
[lstm_seq] *** End of Epoch 1 *** Avg Loss: 8.9536
[lstm_seq] Reached max_steps_per_epoch=5, ending epoch 2 early.
[lstm_seq] *** End of Epoch 2 *** Avg Loss: 2.4768
[lstm_seq] Reached max_steps_per_epoch=5, ending epoch 3 early.
[lstm_seq] *** End of Epoch 3 *** Avg Loss: 0.2965
[lstm_seq] Final sample (greedy) from prompt: '0 1 2 3 4'
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
Annotated:
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
--------------------------------------------------

*** I'm feeling great today! Hope you're well, too. ***
```
