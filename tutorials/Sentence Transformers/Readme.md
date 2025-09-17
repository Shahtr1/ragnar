### Why we need sentence embeddings?

Imagine every sentence becomes a point in space (a vector). Sentences that mean the same thing are close together; sentences that mean different things are far apart. If you can do that, you can:

- Find similar sentences fast (semantic search).
- Cluster documents by meaning.
- Detect paraphrases.
- Use embeddings as numeric features for downstream models.

### 1) From raw text to numbers (tokenization → embeddings)

Before any neural network can work on text, text must become numbers.

Example sentence:
`"I love apples"`

Tokenization (BPE / WordPiece)

- The tokenizer splits string into tokens (subword units). Example (illustrative, not exact): `["I", "love", "ap", "ples"]`
- Each token maps to an integer id: e.g. `I → 10, love → 423, ap → 301, ples → 402`.

### Why integers?

Because computers don’t work with text, they only work with numbers (binary). We can’t directly feed `"love"` into a neural net. But if we assign `"love"` an ID (say `423`), then we can look it up in a table of vectors (an embedding matrix).

### Embedding lookup (turn IDs into vectors)

Now, suppose we have an embedding matrix:

$$
E \in \mathbb{R}^{V \times d}
$$

- $V$ = vocabulary size (say 30,000 tokens).
- $d$ = embedding dimension (say 768 in BERT, or 4 in toy example).

Each row = a learned vector for one token.
Example (tiny d = 4):

| Token  | ID  | Embedding vector       |
| ------ | --- | ---------------------- |
| "I"    | 10  | \[0.1, 0.2, 0.0, 0.5]  |
| "love" | 423 | \[0.3, 0.4, -0.1, 0.2] |
| "ap"   | 301 | \[0.0, 0.2, 0.1, 0.0]  |
| "ples" | 402 | \[0.1, 0.0, 0.1, -0.1] |

### How does the model decide what numbers go into those vectors (rows of the embedding matrix)?

1. At the very beginning (before training)

- The embedding matrix $ E \in \mathbb{R}^{V \times d} $ is created.
- Every entry is usually just random numbers (tiny random floats, like 0.01, -0.03, …).
- So initially, `"I"`, `"love"`, `"ap"`, `"ples"` have random vectors — no meaning yet.

At this stage the network knows nothing about language.

2. Training with a task

The model then trains on a task — like predicting the next word.
Example sentence from dataset:

`"I love apples"`

Task: predict the next token given the previous ones.

- Input: `"I love"`
- Target: `"apples"`

So the network will:

- Take token IDs → look up their random embeddings → process with transformer layers.
- Try to predict `"apples"` as the next token.
- Compare its guess with the true answer using loss (cross-entropy).
- Backpropagate the error → update weights, including the embeddings.

3. Updating embeddings with gradient descent

Here’s the key:
When the model is wrong, gradients flow back all the way to the embedding matrix.

Example:

- "love" was used in the input.
- The prediction was bad.
- Backprop says: “hey, the row for `"love"` should shift slightly so that next time, the network predicts better.”

So the vector for "love" (row 423 in $E$) gets nudged:

old vector: `[0.3,0.1,−0.2,0.5]`
new vector: `[0.31,0.12,−0.19,0.49]`

_“The embedding matrix is a group of vectors, each vector has dimensions. The 4th dimension of the vector for `‘love’` decreased slightly `(0.5 → 0.49)` because backpropagation decided that lowering it reduces the model’s error.”_

Do this billions of times across a giant text dataset → embeddings move into positions that capture semantic meaning.

4. What ends up happening?

Through training:

- Words that appear in similar contexts get vectors close to each other.
  (e.g. `"cat"` and `"dog"` appear in sentences like “The \_\_\_ is sleeping.”)
- Directions in the vector space start representing relationships.
  `"king - man + woman ≈ queen"`

So embeddings are not hand-crafted — they are learned automatically through the task + gradient descent.

#### Forward pass (one run of the model):

When you give the model the sentence `"I love apples"`, here’s what happens immediately:

Token → embedding
Each token ID (like `"I" → 10, "love" → 423`) is looked up in the embedding matrix.
Example vectors:

- `"I"` → `[0.1, 0.2, 0.0, 0.5]`
- `"love"` → `[0.3, 0.4, -0.1, 0.2]`

If we only used token embeddings, the model would know what the words are, but not their order.

Example:

- `"I love apples"` → vectors for `["I", "love", "apples"]`
- `"apples love I"` → the same vectors, just in a different sequence.

Without extra info, the model can’t easily distinguish these.

The solution: positional embeddings

We give each position in the sequence its own vector.

- Position 0 (first word in the sentence) → vector like `[0.05, -0.01, 0.02, 0.03]`
- Position 1 (second word) → vector like `[-0.02, 0.01, 0.04, 0.00]`
- Position 2 → another vector … and so on.

### How they’re used

When processing each token:

$ final token vector = token embedding + positional embedding $

Example:

- Token `"I"` → `[0.1, 0.2, 0.0, 0.5]`
- Position 0 vector → `[0.05, -0.01, 0.02, 0.03]`
- Add them: `[0.15, 0.19, 0.02, 0.53]`

So now the vector encodes both meaning and order.

2. Transformers — the core: self-attention
