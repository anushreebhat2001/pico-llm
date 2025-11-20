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
![TinyStories sanity plot](pico-llm/trained_outputs/outputs_embedding_tiny/loss_means_epoch_ltsm.png)

3seqs.txt training sanity (train/test loss over epochs):

**Train loss per step + test loss per epoch**
![3seqs sanity plot](pico-llm/trained_outputs/outputs_embedding/loss_means_epoch_lstm.png)



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
![One-hot 3seq sanity plot](pico-llm/trained_outputs/outputs_onehot/loss_means_epoch_kgram.png)


**(b) K-gram MLP on 3seq with `nn.Embedding`**

**Train loss per step + test loss per epoch**
![Embedding 3seq sanity plot](pico-llm/trained_outputs/outputs_embedding/loss_means_epoch_kgram.png)



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
![Embedding 3seq sanity plot](pico-llm/trained_outputs/outputs_3seqs_fullpattern/loss_means_epoch_kgram.png)


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
- use_position_emb

### Plot

Transformer on 3seq sanity plot (train/test loss):

![KV Cache sanity plot](pico-llm/trained_outputs/outputs_3seqs_fullpattern/loss_means_epoch_kv.png)

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
- use_position_emb

### Plot

Transformer on tinystories sanity plot (train/test loss):

![TinyStories sanity plot](pico-llm/trained_outputs/outputs_tinystories_full/loss_means_epoch_kv.png)

### Tinystories – overfitting
In the TinyStories training curve, we observe that while the training loss steadily decreases over the epochs, the test loss flattens early and begins to slightly increase in the later epochs.

# Optional Tasks

# Q1. Own Dataset - 

We used a custom subset of 30,000 lines from the hugging face Wikipedia corpus dataset as our training data to study model behavior on long-form, factual text beyond TinyStories enabling the model to learn from longer, information-rich sentences and diverse real-world topics.

### Configuration - 

- wiki_weight : 1.0
- n_heads : 16
- n_blocks : 8
- batch_size : 16
- learning_rate : 3e-4
- block_size : 512
- embed_size : 1024
- test_fraction : 0.1
- epochs : 15
- max_steps_per_epoch : 1500
- use_position_emb

![TinyStories sanity plot](pico-llm/trained_outputs/outputs_wiki_512/loss_means_epoch_kv.png)

Overfit so decreasing epochs according to the image to around 3-4, 4500 global steps.(where the test loss is min)

### Configuration - 

- wiki_weight : 1.0
- n_heads : 16
- n_blocks : 8
- batch_size : 16
- learning_rate : 3e-4
- block_size : 1024
- embed_size : 1024
- test_fraction : 0.1
- epochs : 3
- max_steps_per_epoch : 1500
- use_position_emb

![TinyStories sanity plot](pico-llm/trained_outputs/outputs_wiki_1024/loss_means_epoch_kv.png)


# Q2. Overfitting vs Underfitting
### Configuration - 

- tinystories_weight : 1.0
- n_heads : 16
- n_blocks : 8
- batch_size : 16
- learning_rate : 3e-4
- block_size : 512
- embed_size : 1024
- test_fraction : 0.1
- use_position_emb

## Quantitative analysis

### For Underfitting -


![TinyStories sanity plot](pico-llm/trained_outputs/outputs_tiny_underfit/loss_means_epoch_kv.png)

#### Configuration:
epochs = 3 | max_steps_per_epoch = 100

#### Observation:

- Both train and test losses remain high (~ 2–2.5) and closely aligned.

- The loss curve flattens early, indicating that the model stops improving after very few updates. (not even 0.5 loss difference within test curve)

- The small gap between train and test losses suggests the model is not learning enough patterns to differentiate training vs unseen data.


### For Overfitting -

![TinyStories sanity plot](pico-llm/trained_outputs/outputs_tiny_overfit/loss_means_epoch_kv.png)

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


# Q3. Change in Hyperparameters
### Change in embed size- 

Taking the 2 configurations of the custom wiki dataset where one has 512 embed size and one has 1024 embed size - 

![Wiki hyperparams](pico-llm/trained_outputs/hyperparams/compare_embed_loss.png)

The model with **embedding size 1024** converges noticeably faster, showing a steeper decline in loss across the first three epochs. This happens because larger embeddings offer greater expressive power and smoother optimization dynamics, enabling the model to capture token relationships more effectively early in training. In contrast, the **512-dimensional** model learns more slowly and plateaus higher, reflecting its lower representational capacity.


###Transformer Depth and Width (number of heads and blocks)

We have 16 heads and 8 blocks with 512-dimensional embeddings but have 30k datalines in tiny stories and wiki.

![Wiki hyperparams](pico-llm/trained_outputs/hyperparams/overfitting_tiny_vs_wiki_16H8B.png)

The model’s capacity far exceeds the size of both datasets, leading to overfitting in different ways. TinyStories overfits quickly as the model memorizes short, repetitive stories, while Wiki’s higher and dense diverse data exposes capacity limits — the model continues reducing training loss even as test loss rises steadily, reflecting poor generalization.

# Q4. Effect of positional embedding



# Q5. Interpretability: Analyzing Attention Head Behavior

To understand the internal mechanisms of the trained Transformer, we analyzed the attention patterns of specific heads. We collected the attention matrices while processing the following prompt:

> **PROMPT:** "Once upon a time, there was a little girl named Lily who loved to play outside."

### Configuration 

- **tinystories_weight**: 1.0
- **Num of epochs**: 10
- **n_heads**: 8
- **n_blocks**: 6
- **batch_size**: 16
- **learning_rate**: 3e-4
- **block_size**: 512
- **embed_size**: 512
- **max_steps_per_epoch**: 750
- **test_fraction**: 0.1
- **Layer (Block Index)**: 3 (Layer 3 for middle analysis)
- **Head Indices**: 0, 1, 2, 3 (to compare multiple heads)
- use_position_emb
- **Prompt Token Length**: 17


### Attention Map Visualization: Layer 3, Head 0, Head 1, Head 2, Head 3

#### **Figure 1**: Attention Heatmap for Layer 3, Head 0
![Attention Heatmap for Layer 3, Head 0](pico-llm/trained_outputs/plots_interpretability/Figure_1.png)

#### **Figure 2**: Attention Heatmap for Layer 3, Head 1
![Attention Heatmap for Layer 3, Head 1](pico-llm/trained_outputs/plots_interpretability/Figure_2.png)

#### **Figure 3**: Attention Heatmap for Layer 3, Head 2
![Attention Heatmap for Layer 3, Head 2](pico-llm/trained_outputs/plots_interpretability/Figure_3.png)

#### **Figure 4**: Attention Heatmap for Layer 3, Head 3
![Attention Heatmap for Layer 3, Head 3](pico-llm/trained_outputs/plots_interpretability/Figure_4.png)

The plots show the attention weight ($W_{q,k}$) assigned by a query token at position $q$ (Y-axis) to a key token at position $k$ (X-axis). Darker colors indicate low weight, while yellow/green indicates high weight (up to 1.0).

### Detailed Analysis and Interpretation

#### **Layer 3, Head 0: Contextual and Grammatical Attention**
Head 0 in Layer 3 focuses on establishing the **grammatical structure** of the sentence by attending to key auxiliary tokens and the immediate context.

1. **Positional Indexing and Local Context (High-Weight Diagonals)**
   - The earliest query tokens (e.g., $q=0$ to $q=4$) show extremely high attention weights (yellow, $\approx 1.0$) to the immediately preceding tokens, including position $k=0$.
   - **Interpretation**: The model emphasizes **Context Establishment**, ensuring strong awareness of its immediate neighborhood and the sentence's structure.

2. **Focus on Auxiliary/Grammatical Tokens**
   - The head pays particular attention to **grammatical words**, such as **auxiliary verbs** or determiners. This is visible as vertical bands in the heatmap, especially around positions $k=6$ and $k=7$ (which correspond to auxiliary verbs like "was" and determiners like "a").
   - **Interpretation**: Head 0 specializes in identifying syntactic markers, focusing on **structure** and **auxiliary words**.

#### **Layer 3, Head 1: Subject-Verb-Object Relations**

### Detailed Analysis and Interpretation

#### **Layer 3, Head 0: Contextual and Grammatical Attention**

Head 0 in Layer 3 focuses on establishing the **grammatical structure** of the sentence by attending to key auxiliary tokens and the immediate context.

1. **Positional Indexing and Local Context (High-Weight Diagonals)**
   - The earliest query token, **'Once'** at position $q=0$, receives a **weight of 1.000**, indicating that it is a pivotal token in setting the **context** for the rest of the sequence.
   - **Interpretation**: This behavior shows that the model prioritizes **context establishment** and **sentence initialization**.

2. **Focus on Auxiliary/Grammatical Tokens**
   - **'Once'** is a strong anchor, with other tokens like **'there'** and **'time'** attending to it, establishing a strong **grammatical relationship** between them.
   - **Interpretation**: Head 0 seems to specialize in **structural recognition**, giving high attention to the starting token and reinforcing sentence structure.

#### **Key Token Attention: 'Once' (Position 0)**
- **High Attention to 'Once'**:
    - **'Once'** is given a high weight (1.000), which is likely because it is the **start of the sentence**, and the model uses it to build the sentence structure.
  
- **Subsequent Attention**:
    - **'there'** at position $q=5$ attends to **'Once'** with weight **0.370**, indicating that **"Once"** is crucial for the context of the following tokens.
    - Other tokens like **'girl'**, **'Lily'**, and **'loved'** also attend to **'Once'**, but with lower weights, suggesting that they use the context established by **"Once"** to continue forming relationships within the sentence.

---

### Visualizing Attention Maps:
- **Head 0, Layer 3** shows an intense focus on **'Once'**, reinforcing the idea that this head is concerned with **contextual initiation** and **grammatical structure**.


---

### Visualizing Attention Maps:
The attention maps for **Layer 3, Head 1** reveal how the model dynamically attends to the **subject-verb-object** relationships. By focusing on key tokens like **'girl'**, **'Lily'**, and **'was'**, **Head 1** captures the underlying syntactic dependencies that define sentence meaning.


#### **Layer 3, Head 2: Entity Recognition and Focus on Keywords**
Head 2 focuses on **named entity recognition** and linking relevant entities, such as names and actions.

1. **Token Attention to Key Subjects and Actions**
   - **Query 11 ('Lily')** attends heavily to **'girl'** (`0.399`), **'there'** (`0.214`), and **'Once'** (`0.155`), suggesting that it tracks key entities and their roles in the sentence.
   - **Interpretation**: This suggests that **Head 2** specializes in recognizing **key subjects** and their respective actions, such as **"Lily"** as the subject and **"girl"** as its reference.

2. **Strong Attention to Actions**
   - **Query 13 ('loved')** attends strongly to **'Once'** (`0.171`), **'there'** (`0.101`), and **'Lily'** (`0.078`), indicating that the head tracks the **actions** associated with the subject.
   - **Interpretation**: Head 2 likely helps the model focus on **action-based dependencies**, connecting actions like "loved" to the subject and context.

#### **Layer 3, Head 3: Token Attention to Relationships and Dependencies**
Head 3 focuses on understanding **subject-object relationships** and how the model ties together actions, entities, and context.

1. **Token Attention to Key Subjects and Relationships**
   - **Query 16 ('outside')** attends to **'girl'** (`0.245`), **'Lily'** (`0.159`), and **'outside'** itself (`0.123`).
   - **Interpretation**: This shows how **Head 3** helps build a cohesive relationship between the subject, actions, and descriptors (like **"outside"**).

2. **Linking Actions and Tokens**
   - **Query 17 ('.')** attends heavily to **'girl'** (`0.286`), **'outside'** (`0.195`), and **'Lily'** (`0.112`), suggesting that **Head 3** tracks the connections between actions and descriptions.
   - **Interpretation**: Head 3 appears to track **higher-level dependencies** between the main subject, actions, and important events in the sentence.

### **Conclusion and Inferences**

- **Head Specialization**: Each attention head in Layer 3 appears to specialize in different aspects of language:
  - **Head 0**: Focuses on establishing the **context** and **grammatical structure**.
  - **Head 1**: Focuses on establishing **subject-verb-object relationships**, linking key **nouns** and **verbs** to form the sentence's core syntactic structure.
  - **Head 2**: Specializes in **entity recognition** and **action tracking**.
  - **Head 3**: Focuses on **high-level dependencies** and **relationship building**.

- **Layer 3 Attention Behavior**: Layer 3 is responsible for **complex contextual understanding**, as demonstrated by the specialization of the attention heads. While **Head 0** captures grammatical structures, **Head 1**, **Head 2**, and **Head 3** build upon these structures to establish deeper relationships between the sentence elements.

---

### Visualizing Attention Maps:
- The heatmaps for each head show how the model dynamically attends to different parts of the sentence at each layer. These plots visually depict the attention relationships between the tokens in the input prompt.
- The comparison across heads highlights how each head focuses on distinct aspects of the input, from grammatical structure to entity relationships.

---

# Q7. Analyzing Pre and Post Normalization Effects
### Configuration - 

- tinystories_weight : 1.0
- n_heads : 16
- n_blocks : 8
- batch_size : 16
- learning_rate : 3e-4
- block_size : 512
- embed_size : 1024
- test_fraction : 0.1
- use_position_emb
- use_post_norm

![TinyStories Pre post norm plots](pico-llm/trained_outputs/outputs_postnorm_tinystories/Pre_post_norm_plots.png)

We observe that across all epochs, both training and test losses for the post-norm variant are lower than those of the pre-norm model. Post-norm places layer normalization after the residual addition, which stabilizes gradient flow by preventing the residual branch from dominating early in training. This reduces gradient variance, improves conditioning of the Transformer blocks, and leads to smoother optimization in small sized datasets like Tiny Stories.
