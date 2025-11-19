# Pico-LLM Sanity Checks and Core Implementations

## **Question 1.** Sanity check: running the baseline code (LSTM on TinyStories and 3seq)

### Configurations - 

**(a) LSTM on TinyStories**

- tinystories_weight : 1.0
- Num of epochs : 3
- n_heads : 8
- n_blocks : 6
- batch_size : 16
- learning_rate : 1e-3
- block_size : 32
- embed_size : 128
- max_steps_per_epoch : 10
- test_fraction : 0.1


**(b) LSTM on 3seqs input file**

- tinystories_weight : 1.0
- input_files : 3seqs.txt
- Num of epochs : 3
- n_heads : 8
- n_blocks : 6
- batch_size : 16
- learning_rate : 1e-3
- block_size : 32
- embed_size : 128
- max_steps_per_epoch : 10
- test_fraction : 0.1


### Plot

TinyStories training sanity (train/test loss over epochs):

**Train loss per step + test loss per epoch**
![TinyStories sanity plot](pico-llm/trained_outputs/outputs_embedding_tiny/LTSM_tiny.png)

3seqs.txt training sanity (train/test loss over epochs):

**Train loss per step + test loss per epoch**
![3seqs sanity plot](pico-llm/trained_outputs/outputs_embedding/LTSM_sq3.png)



These runs verify that the provided training loop, data pipeline, and generation code all work as expected. 

Both train and test loss decrease over epochs (see the plot above), which confirms that the LSTM.

---

## **Question 2.** KGramMLPSeqModel + sanity checks (one-hot vs embedding)

### Configurations

- tinystories_weight : 1.0
- input_files : 3seqs.txt
- Num of epochs : 3
- n_heads : 8
- n_blocks : 6
- batch_size : 16
- learning_rate : 1e-3
- kgram_k : 3
- num_inner_mlp_layers : 1
- kgram_chunk_size : 1
- block_size : 32
- embed_size : 128
- max_steps_per_epoch : 10
- test_fraction : 0.1

**(a) K-gram MLP on 3seq with one-hot inputs**

**Train loss per step + test loss per epoch**
![One-hot 3seq sanity plot](pico-llm/trained_outputs/outputs_onehot/kgram_onehot.png)


**(b) K-gram MLP on 3seq with `nn.Embedding`**

**Train loss per step + test loss per epoch**
![Embedding 3seq sanity plot](pico-llm/trained_outputs/outputs_embedding/kgram_embedding.png)



### Explanation - 

Both runs use the same k-gram architecture and training loop but differ in how token context is represented. 

In the one-hot version, training loss decreases extremely slowly, indicating poor optimization and a very high-dimensional input space. 

In the embedding version, the loss drops quickly and the train/test curves behave well. This demonstrates that the sequence-to-sequence k-gram implementation is correct and that using `torch.nn.Embedding` is both efficient and beneficial for learning.

---

**(c) K-gram MLP on 3seq trained with final configurations**

### Configurations

- tinystories_weight : 0.0
- input_files : 3seqs.txt
- Num of epochs : 15
- n_heads : 8
- n_blocks : 6
- batch_size : 16
- learning_rate : 3e-4
- kgram_k : 3
- num_inner_mlp_layers : 1
- kgram_chunk_size : 1
- block_size : 1024
- embed_size : 128
- max_steps_per_epoch : 210
- test_fraction : 0.1

**Train loss per step + test loss per epoch**
![Embedding 3seq sanity plot](pico-llm/trained_outputs/outputs_3seqs_fullpattern/kgram_full.png)


## 3. **Question 3.** Nucleus (top-p) sampling

### Configurations

- tinystories_weight : 0.0
- input_files : 3seqs.txt
- Num of epochs : 15
- n_heads : 8
- n_blocks : 6
- batch_size : 16
- learning_rate : 3e-4
- kgram_k : 3
- num_inner_mlp_layers : 1
- kgram_chunk_size : 1
- block_size : 1024
- embed_size : 128
- max_steps_per_epoch : 210
- test_fraction : 0.1

### Nucleus Sampling Output Table using the trained 3seq.txt on various top-p values for kgram_mlp_seq:

| Top-p Value | Generated Output (for last epoch)                                                                |
|-------------|-----------------------------------------------------------------------------------|
| Greedy      | 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24                  |
| 0.2         | 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24                  |
| 0.5         | 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24                  |
| 0.75        | 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24                  |
| 0.95        | 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22scans                  |
| 1.0         | 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17ometers oxidative 146                |


### Configurations

- tinystories_weight : 1.0
- Num of epochs : 10
- n_heads : 8
- n_blocks : 6
- batch_size : 16
- learning_rate : 3e-4
- kgram_k : 3
- num_inner_mlp_layers : 1
- kgram_chunk_size : 1
- block_size : 512
- embed_size : 512
- max_steps_per_epoch : 750
- test_fraction : 0.1


### Nucleus Sampling Output Table using the tinystories data on various top-p values for kgram_mlp_seq:


| Top-p Value | Generated Output (for last epoch)                                                                                    |
|-------------|------------------------------------------------------------------------------------------------------|
| Greedy      | Once upon a time, there was a little girl named Lily. She loved to play outside and explore the world around her |
| 0.2         | Once upon a time, there was a little girl named Lily. She loved to play outside and explore the world around her |
| 0.5         | Once upon a time, there was a little girl named Lily. She loved to eat cherries, especially the soft ones        |
| 0.75        | Once upon a time, there was a dog named Max. Max loved to play with his owner, a lot. His                        |
| 0.95        | Once upon a time, there was a little girl named Lily. She loved to run and play outside and run around with      |
| 1.0         | Once upon a time, there were two friends Jack and Annie. They loved to play together and think. Today was special    |


### Explanation

As the `top-p` value increases, the generated text becomes more diverse and creative. Lower values like `0.2` or `0.5` stick closely to greedy decoding, while higher values like `0.75`, `0.95` and `1.0` allow the model to explore alternative continuations.

---

## 4. TransformerModel: causal decoder-only transformer with RMSNorm

### Configurations

- tinystories_weight : 0.0
- input_files : 3seqs.txt
- Num of epochs : 15
- n_heads : 8
- n_blocks : 6
- batch_size : 16
- learning_rate : 3e-4
- block_size : 1024
- embed_size : 128
- max_steps_per_epoch : 210
- test_fraction : 0.1

### Plot

Transformer on 3seq sanity plot (train/test loss):

![TinyStories sanity plot](pico-llm/trained_outputs/outputs_3seqs_fullpattern/3seq_transformer.png)

### 3seq.txt – Perfect Fit with Low Generalization Risk
For the 3seq.txt dataset, both training and test loss drop sharply within the first few hundred steps and plateau close to zero.

### Configurations

- tinystories_weight : 1.0
- Num of epochs : 10
- n_heads : 8
- n_blocks : 6
- batch_size : 16
- learning_rate : 3e-4
- block_size : 512
- embed_size : 512
- max_steps_per_epoch : 750
- test_fraction : 0.1

### Plot

Transformer on tinystories sanity plot (train/test loss):

![TinyStories sanity plot](pico-llm/trained_outputs/outputs_tinystories_full/outputs_tinystories_full.png)

### Tinystories – Perfect Fit with Low Generalization Risk
In the TinyStories training curve, we observe that while the training loss steadily decreases over the global steps, the test loss flattens early and begins to slightly increase in the later epochs. (Maybe overfitting)

# Optional Tasks

# Own Dataset - 

We used a custom subset of 30,000 lines from the hugging face Wikipedia corpus dataset as our training data to study model behavior on long-form, factual text beyond TinyStories enabling the model to learn from longer, information-rich sentences and diverse real-world topics.

![TinyStories sanity plot](pico-llm/trained_outputs/outputs_wiki_final/outputs_wiki_final.png)


# Overfitting vs Underfitting
### Configuration - 

- tinystories_weight : 1.0
- n_heads : 16
- n_blocks : 8
- batch_size : 16
- learning_rate : 3e-4
- block_size : 512
- embed_size : 1024
- test_fraction : 0.1

## Quantitative analysis

### For Underfitting -


![TinyStories sanity plot](pico-llm/trained_outputs/outputs_tiny_underfit/outputs_tiny_underfit.png)

#### Configuration:
epochs = 3 | max_steps_per_epoch = 100

#### Observation:

- Both train and test losses remain high (~ 2–2.5) and closely aligned.

- The loss curve flattens early, indicating that the model stops improving after very few updates.

- The small gap between train and test losses suggests the model is not learning enough patterns to differentiate training vs unseen data.

- Fails to capture even training regularities.

### For Overfitting -

![TinyStories sanity plot](pico-llm/trained_outputs/outputs_tiny_overfit/outputs_tiny.png)

#### Configuration:
epochs = 15 | max_steps_per_epoch = 2500

#### Observation:

- Training loss drops sharply (≈ 10 → 0.3) while test loss initially decreases but then stabilizes or slightly rises (~ 1.0–1.2). 

- A clear gap forms between train and test curves as training progresses. The widening gap between training and test losses quantitatively demonstrates overfitting.

- The model’s performance on unseen data ceases to improve even though training loss keeps falling. The model has begun to memorize the training samples instead of learning generalizable patterns.


## Qualitative analysis

### For Underfitting -

```
Prompt: Once upon a time 
```
```
Generated text: Once upon a time, there was a little girl named Lily. She had a ball forest that her mum they would play with her mommy and new on her shoes and asked her favorite things. She was a circle, she was very brave.

Her mommy didn't mind, her medicine the safest her mum loved to visit it. Lily's mom took the bushes and or jumple to learn her new friends feel better. Lily and wanted to cactus.

After they picked up because they got very clever that it forgot in the garden and hopped over and whist down so much fun. They played together with past and watched the Jill remembered, "Did you."
```

### For Overfitting -

```
Prompt: Once upon a time 
```
```
Generated text: Once upon a time, there were two friends, Bobby and Milly. Bobby was a jolly little boy who loved to play with him. One day, Bobby wanted to play with his laser so he asked Milly if he could play with him. Milly said yes and he was so excited. So Bobby and Milly played with the laser together. They had so much fun and laughed until they both started to get tired. Bobby invited his friend to join her in the garden. When they had finished playing, Bobby said â€œLet's have a rest under the sun.â€ Lilly agreed and they both said â€œLet's have a great day!â€ And so Bobby and Milly spent the day playing with the laser and having lots of fun. Bobby and Milly had so much fun that day. The end!!!!!!!!!!!!!!!!!
```
#### Observations

| Condition    | Qualitative output                    | Diversity             | Evidence of Memorization |
| ------------ | ----------------------------------- | --------------------- | ------------------------ | 
| **Underfit** | Poor – incoherent, broken sentences since training pattern is not established | Random/illogical      | None                     | 
| **Overfit**  | High fluency, grammatical but memorized training patterns instead of generalizing it          | Very low – repetitive | Strong                   |
