# GPT From Scratch: Building a Character-Level Language Model 🚀🔥

This notebook provides a step-by-step guide to building a Generative Pre-trained Transformer (GPT) model from scratch, focusing on character-level text generation. We'll go from downloading and preprocessing data to implementing self-attention and training a full transformer model.  This is heavily inspired by Andrej Karpathy's excellent [Zero To Hero](https://karpathy.ai/zero-to-hero.html) series, specifically the "makemore" and "micrograd" videos, and the nanoGPT repository. We are building, in essence, a simplified nanoGPT.

## Table of Contents

1.  [Data Loading and Preprocessing](#data-loading) 📚
2.  [Tokenization and Encoding](#tokenization) 🧮
3.  [Data Splitting (Train/Validation)](#data-splitting) ✂️
4.  [Batch Generation](#batch-generation) 📦
5.  [Bigram Language Model (Baseline)](#bigram-model) 📝
6.  [Training Loop (Bigram)](#training-loop-bigram) 🔄
7.  [Mathematical Trick: Self-Attention](#self-attention-trick) 🧙‍♂️
8.  [Self-Attention Implementation](#self-attention-implementation) 🧠
9.  [Multi-Head Attention and Feedforward Network](#multi-head-attention) 🤯
10. [Transformer Block](#transformer-block) 🧱
11. [Full GPT Model](#full-gpt-model) 🤖
12. [Training Loop (Full GPT)](#training-loop-gpt) 🚀
13. [Text Generation](#text-generation) ✍️
14. [Layer Normalization Discussion](#layer-normalization) 📊


## 1. Data Loading and Preprocessing <a name="data-loading"></a>📚

We begin by downloading the Tiny Shakespeare dataset, a classic text corpus for language modeling.

```python
# We always start with a dataset to train on. Let's download the tiny shakespeare dataset
!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

This cell uses `wget` to fetch the `input.txt` file from Karpathy's char-rnn repository.  The output shows the download progress and confirms successful retrieval.

Next, we read the downloaded file and inspect its contents:

```python
#read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print('Length of the data set in characters ', len(text))
print(text[:1000])
```

We open the file in read mode (`'r'`) with UTF-8 encoding to handle a wider range of characters.  The file's contents are read into the `text` variable. We print the total number of characters and the first 1000 characters to get a glimpse of the data.  This confirms the dataset is Shakespearean text.

## 2. Tokenization and Encoding <a name="tokenization"></a>🧮

Language models operate on numerical representations of text, not the raw characters themselves.  This section handles the crucial step of *tokenization*, where we convert characters to integers.  We're using a *character-level* tokenizer, meaning each individual character is a token.

```python
# all the unique characters
chars = sorted(list(set(text)))
vocab_size=len(chars)
print(''.join(chars) )
print(vocab_size)
```

Here, we create a vocabulary:

*   `set(text)`:  Creates a set of all unique characters in the text, eliminating duplicates.
*   `list(...)`: Converts the set back into a list.
*   `sorted(...)`: Sorts the list alphabetically.
*   `chars`:  Holds the sorted list of unique characters (our vocabulary).
*   `vocab_size`: Stores the number of unique characters.
*	The print out put shows the vocab and it's size.

We then build *encoder* and *decoder* functions:

```python
# create a mapping from characters to integers
stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of numbers
decode = lambda l: ''.join([itos[i] for i in l ]) # decoder: take a list of integers output a string
print(encode("hii there"))
print(decode(encode("hii there")))
```

*   `stoi`: A dictionary mapping each character (`ch`) to its corresponding integer index (`i`).  This is our "string to integer" mapping.
*   `itos`:  The inverse dictionary, mapping integers (`i`) back to characters (`ch`).  This is our "integer to string" mapping.
*   `encode`: A function that takes a string `s` and returns a list of integers, using the `stoi` mapping.
*   `decode`: A function that takes a list of integers `l` and returns the corresponding string, using the `itos` mapping.

The `print` statements demonstrate the encoding and decoding process, verifying that they are inverses of each other.

Finally convert all of the text into a tensor
```python
# lets encode the entire text dataset and store into a torch.tensor
import torch # useing pytorch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000]) # the 1000 characters we looked at earieler will to the gpt looks like this
```
* imports pytorch
*  `data`: coverts the encodded data into torch tensor


## 3. Data Splitting (Train/Validation) <a name="data-splitting"></a>✂️

A fundamental step in machine learning is splitting the data into training and validation sets. This allows us to evaluate how well our model generalizes to unseen data.

```python
# lets split the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train , rest val
train_data = data[:n]
val_data = data[n:]
```

We split the data:

*   `n`: Calculates the index representing 90% of the data length.
*   `train_data`:  Contains the first 90% of the data, used for training the model.
*   `val_data`:  Contains the remaining 10% of the data, used for evaluating the model's performance on unseen data.

## 4. Batch Generation <a name="batch-generation"></a>📦

To train efficiently, we process data in *batches*. This section defines a function to generate batches of input sequences (`x`) and corresponding target sequences (`y`).

```python
block_size = 8
train_data[:block_size+1]
```
```python
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):\n",
    context = x[:t+1]\n",
    target = y[t]\n",
    print(f"when input is {context} the target: {target}")
```
*  Illustrates the concept of context and target.
*  For a given `block_size`, it shows how the input sequence (`context`) gradually increases, and the corresponding `target` is the next character in the sequence.

```python
torch.manual_seed(1337)
batch_size = 4 # how many independendent sequences will be process in paralle
block_size = 8 # what is the maximum context length of predictions?

def get_batch(split):
  # generate a small batch of data of inputs x and targets y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size,(batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1: i+block_size+1] for i in ix])
  return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size): # batch dimension
  for t in range(block_size): # time dimension
    context = xb[b, :t+1]
    target = yb[b,t]
    print(f\"when input is {context.tolist()} the target: {target}\")
```

Key parameters:

*   `batch_size`:  The number of independent sequences processed in parallel.  Higher batch sizes can lead to faster training but require more memory.
*   `block_size`: The maximum length of the input sequences (context length).  This determines how far back in the text the model can "look" for patterns.

The `get_batch` function:

1.  Selects either the training or validation data based on the `split` argument.
2.  Generates `batch_size` random starting indices (`ix`) within the chosen dataset.  These indices ensure we don't go out of bounds.
3.  Creates input tensors `x` and target tensors `y` by stacking sequences of length `block_size`.  Crucially, `y` is offset by one position relative to `x`, representing the next character prediction task.

The code then calls `get_batch` to create a sample batch (`xb`, `yb`) and prints the shapes and contents of the input and target tensors.  The nested loops further illustrate how the context and target relate within a batch.

## 5. Bigram Language Model (Baseline) <a name="bigram-model"></a>📝

We start with a simple *bigram language model*.  A bigram model predicts the next character based only on the *immediately preceding* character. It's a basic model but serves as a good starting point.

```python
import torch
import torch.nn as nn
from torch.nn  import functional as F
torch.manual_seed(1337)

class BigramLanguage(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    # each token directly reads off the logit for the next token from a look up table
    self.token_embdding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx, targets= None):
     # idx and tragets are both (B, T) tensors of integers
     logits = self.token_embdding_table(idx) # (B, T, X)
     if targets is None:\n",
      loss = None\n",
     else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)



     return logits, loss

  def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of the indices in the current context
    for _ in range(max_new_tokens):
      # get the predictions
      logits, loss = self(idx)
      # focus only on the last time step
      logits = logits[:, -1, :] # becomes (B, C)
      # apply soft max to get the probabilities
      probs = F.softmax(logits, dim=1) # (B, C)
      # Sample from the distrubution
      idx_next = torch.multinomial(probs, num_samples=1) # (B, C)
      # append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx

m = BigramLanguage(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
```

The `BigramLanguage` class:

*   `__init__`:
    *   Initializes the model.
    *   `self.token_embedding_table`:  This is the core of the bigram model. It's an `nn.Embedding` layer, which acts as a lookup table.  It maps each token (character index) to a vector of `vocab_size` values.  These values are the *logits* for the next character.
*   `forward`:
    *   Performs the forward pass of the model.
    *   `logits = self.token_embedding_table(idx)`:  This retrieves the logits for the next character, given the input indices `idx`.  `idx` has shape (Batch, Time), and `logits` has shape (Batch, Time, VocabSize).
    *   The code then handles the calculation of the loss (if `targets` are provided) using `F.cross_entropy`. This requires reshaping the `logits` and `targets` to be compatible with the cross-entropy function.
*   `generate`:
    *   Generates new text, starting from a given context `idx`.
    *   The loop iterates `max_new_tokens` times, generating one character at a time.
    *   `logits, loss = self(idx)`:  Gets the logits for the next character.
    *   `logits = logits[:, -1, :]`:  Crucially, we only consider the logits from the *last* time step, because the bigram model only cares about the current character.
    *   `probs = F.softmax(logits, dim=-1)`:  Converts the logits into probabilities using the softmax function.
    *   `idx_next = torch.multinomial(probs, num_samples=1)`:  Samples the next character from the probability distribution.  `torch.multinomial` is essential for generating diverse text; it introduces randomness.
    *   `idx = torch.cat((idx, idx_next), dim=1)`:  Appends the newly generated character to the sequence.

The code then creates an instance of the `BigramLanguage` model, performs a forward pass with a sample batch, prints the shape of the logits and the loss, and finally generates 100 characters of text starting from a "zero" context. The generated text is gibberish, as expected from a simple bigram model.

## 6. Training Loop (Bigram) <a name="training-loop-bigram"></a>🔄

This section implements the training loop for the bigram model, showing how to update the model's parameters to minimize the loss.

```python
# create a pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32

for steps in range(10000):
  # sample a baych data
  xb, yb = get_batch('train')

  # evaluate loss
  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

print(loss.item())
```

```python
print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=400)[0].tolist()))
```

*   `optimizer = torch.optim.AdamW(...)`: Creates an AdamW optimizer, a variant of Adam with weight decay, which often improves generalization.  We specify the model's parameters (`m.parameters()`) and the learning rate (`lr`).
*   The `for` loop iterates for a specified number of steps (here, 10,000).  Each step represents one update to the model's parameters.
    *   `xb, yb = get_batch('train')`:  Samples a new batch of training data.
    *   `logits, loss = m(xb, yb)`:  Performs the forward pass, calculating the logits and the loss.
    *   `optimizer.zero_grad(set_to_none=True)`:  Resets the gradients of the model's parameters.  This is crucial; otherwise, gradients would accumulate across batches.  `set_to_none=True` is a minor optimization.
    *   `loss.backward()`:  Calculates the gradients of the loss with respect to the model's parameters (backpropagation).
    *   `optimizer.step()`:  Updates the model's parameters based on the calculated gradients.

After training, the code prints the final loss value and generates a longer sequence of text (400 characters). The output text, while still not perfect English, shows a significant improvement compared to the untrained model.  It starts to capture some basic statistical relationships between characters (e.g., spaces often follow punctuation).

## 7. Mathematical Trick: Self-Attention <a name="self-attention-trick"></a>🧙‍♂️

This section introduces the core concept of self-attention, explaining the mathematical trick that allows us to efficiently compute weighted averages of past tokens.

```python
torch.manual_seed(42)
a = torch.tril( torch.ones(3,3))
a = a / torch.sum(a,1, keepdim=True)
b = torch.randint(0, 10, (3,2)).float()
c = a @ b

print('a=')
print(a)
print('--')
print('b=')
print(b)
print('--')
print('c=')
print(c)
```

*  Demonstrates a weighted aggregation using matrix multiplication.
*   `a`:  A lower triangular matrix of ones, normalized so that each row sums to 1. This represents the weights.
* `b`: random 3,2 matrix
* `c`: matrix multiplication.

```python
#the mathematical trick in self-attention
# consider the following toy example:
torch.manual_seed(1337)
B, T, C = 4, 8, 2 # batch , time , channels
x = torch.randn(B, T, C)
x.shape
```

* Sets up a toy example with batch size (`B`), time steps (`T`), and channels (`C`).
*   `x`:  A random tensor representing the input sequence.

```python
# we want x[b,t] = mean_{i<=t} x[b, i]
xbow = torch.zeros((B, T, C))
for b in range(B):
  for t in range(t):
    xprev = x [b, :t+1] # (t, C)
    xbow[b, t] = torch.mean(xprev, 0)
```

*  Calculates the average of all previous tokens for each time step using nested loops. This is inefficient but conceptually clear.
*   `xbow`:  Stores the "bag of words" representation, where each token's representation is the average of all preceding tokens.

```python
# version 2 : using matrix multiply for a weighted aggregation
wei = torch.tril(torch.ones(T, T))
wei =wei/  wei.sum(1, keepdim=True)
xbow2 = wei @ x # (B,T, T) @ (B, T, c) -------> (B, T, C)
torch.allclose(xbow, xbow2)
```
*   Replicates the averaging operation using matrix multiplication.
*    `wei`: create the lower trangular matrix and then make each row sum to one.
*    `xbow2`: weighted averages using matrix multiplication

```python
# version 3: use softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
torch.allclose(xbow, xbow3)
```
*    Introduces the use of `softmax` for normalization.
* `tril`: creates a lower triangular matrix
* `wei`: initilized to zeros
* `wei = wei.masked_fill(tril == 0, float('-inf'))`: This is the key.  It sets the *upper* triangular elements of `wei` to negative infinity.  This prevents "future" tokens from influencing the representation of "past" tokens.
* `wei = F.softmax(wei, dim=-1)`: Applies softmax along the last dimension (rows).  The negative infinities ensure that the weights for future tokens become zero after the softmax.

## 8. Self-Attention Implementation <a name="self-attention-implementation"></a>🧠

This section implements a single head of self-attention, the core building block of the Transformer.

```python
# vesrion 4: self-attention!
torch.manual_seed(1337)
B, T, C = 4,8,32 # bacth, time, channels
x = torch.randn(B, T, C)

# lets see a single head perform self attention
head_size =16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x) # (B, T, 16)
q = query(x) # (B, T, 16)
wei = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) -----> (B, T, T)

tril = torch.tril(torch.ones(T,T))
wei = wei.masked_fill(tril==0 , float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v
out.shape
```

*   `head_size`:  The dimensionality of the key, query, and value vectors.
*   `key`, `query`, `value`:  Linear transformations that project the input `x` into key, query, and value spaces.  These are learnable parameters.
*   `k = key(x)`, `q = query(x)`, `v = value(x)`:  Compute the key, query, and value vectors.
*   `wei = q @ k.transpose(-2, -1)`:  Calculates the *unnormalized* attention weights. This is the dot product of the query and key, measuring the similarity between each query and all keys.  The transpose operation is necessary for proper matrix multiplication.
*    `tril = torch.tril(torch.ones(T,T))`: as befor to make sure that the model can't look into the future.
*    `wei = F.softmax(wei, dim=-1)`: softmax to get the distrubution.
*   `out = wei @ v`:  Calculates the weighted sum of the values, using the attention weights.  This is the output of the self-attention head.

```python
wei[0]
```

* Prints the attention weights for the first batch element.
*   This shows how each token attends to itself and previous tokens.

```python
k = torch.randn(B,T,head_size)
q = torch.randn(B,T,head_size)
wei = q @ k.transpose(-2, -1) * head_size**-0.5
```
* `wei = q @ k.transpose(-2, -1) * head_size**-0.5`: this is the scaled attention.

```python
k.var()
q.var()
wei.var()
```
* Shows how the scaling affects the variance of the attention weights.

```python
torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5]), dim=-1)
torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])*8, dim=-1) # gets too peaky, converges to one-hot
```
* This demonstrates the effect of scaling on the softmax output.  Without scaling, the softmax can become "peaky" (close to a one-hot vector), which can hinder learning.

## 9. Multi-Head Attention and Feedforward Network <a name="multi-head-attention"></a>🤯

This builds upon the single-head self-attention by implementing *multi-head attention* and a *feedforward network*.  These are crucial components of the Transformer.

```python
class LayerNorm1d: # (used to be BatchNorm1d)

  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

  def __call__(self, x):
    # calculate the forward pass
    xmean = x.mean(1, keepdim=True) # batch mean
    xvar = x.var(1, keepdim=True) # batch variance
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]

torch.manual_seed(1337)
module = LayerNorm1d(100)
x = torch.randn(32, 100) # batch size 32 of 100-dimensional vectors
x = module(x)
x.shape
```
```python
x[:,0].mean(), x[:,0].std() # mean,std of one feature across all batch inputs
x[0,:].mean(), x[0,:].std() # mean,std of a single input from the batch, of its features
```
* Shows a class `LayerNorm1d`, which is like batch norm.
```python
print('cuda' if torch.cuda.is_available() else 'cpu')
```
* Checks if the cuda is avaiable.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

*   **Hyperparameters:** Defines key hyperparameters like `batch_size`, `block_size`, `learning_rate`, etc.  These control the training process and model architecture.  The `device` variable is set to `'cuda'` if a GPU is available, otherwise `'cpu'`.

*   **Data Loading and Preprocessing:** This section is identical to the earlier parts, loading the Tiny Shakespeare dataset, creating the character-to-integer mappings, and splitting the data into training and validation sets.

*   **`get_batch` Function:**  Same as before, but with the addition of `x, y = x.to(device), y.to(device)`. This line moves the generated batches to the chosen device (GPU or CPU).

*    **`estimate_loss` Function:**
    *   This function is used to estimate the model's performance on both the training and validation sets *without* updating the model's parameters.  This is crucial for monitoring overfitting.
    *   `@torch.no_grad()`:  This decorator disables gradient calculation, saving memory and computation during evaluation.
    *   `model.eval()`:  Sets the model to evaluation mode. This affects layers like dropout and batch normalization, which behave differently during training and evaluation.
    *   The function iterates through the training and validation sets, calculates the loss for multiple batches, and averages the results.
    *   `model.train()`:  Sets the model back to training mode after evaluation.
*   **`Head` Class:** This is the single-head self-attention implementation, identical to the previous section.
*   **`MultiHeadAttention` Class:**
    *   `__init__`:
        *   Creates multiple `Head` instances and stores them in an `nn.ModuleList`.
        *   `self.proj`:  A linear layer that projects the concatenated outputs of the heads back to the embedding dimension (`n_embd`).  This is often called the "output projection."
        *   `self.dropout`:  Applies dropout to the output of the projection, further regularizing the model.
    *   `forward`:
        *   Concatenates the outputs of all the heads along the channel dimension (`dim=-1`).
        *  `out = self.dropout(self.proj(out))`: Applies the output projection and dropout.
* **`FeedForward` Class:**
    *   Implements a simple feedforward network.
    *   `self.net`: A sequence of layers:
        *   `nn.Linear(n_embd, 4 * n_embd)`:  Expands the input to four times the embedding dimension.
        *   `nn.ReLU()`:  Applies the ReLU activation function, introducing non-linearity.
        *   `nn.Linear(4 * n_embd, n_embd)`:  Projects back to the embedding dimension.
        *   `nn.Dropout(dropout)`:  Applies dropout for regularization.

## 10. Transformer Block <a name="transformer-block"></a>🧱

This section combines the multi-head attention and feedforward network into a single Transformer block.

```python
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

*   **`Block` Class:**
    *   `__init__`:
        *   `head_size = n_embd // n_head`:  Calculates the size of each attention head.
        *   `self.sa`:  An instance of `MultiHeadAttention`.
        *   `self.ffwd`:  An instance of `FeedFoward`.
        *   `self.ln1`, `self.ln2`:  `nn.LayerNorm` layers.  Layer normalization is applied *before* each sub-layer (attention and feedforward). This is known as "pre-norm" and is generally preferred over "post-norm" (where normalization would be applied after the sub-layer).
    *   `forward`:
        *   `x = x + self.sa(self.ln1(x))`:  This is the core of the residual connection.  The input `x` is first normalized using `self.ln1`, then passed to the multi-head attention layer (`self.sa`).  The *output* of the attention layer is then *added* to the original input `x`.  This residual connection helps with gradient flow during training, allowing the model to learn identity mappings easily.
        *   `x = x + self.ffwd(self.ln2(x))`:  Similar to the attention step, the input is normalized, passed to the feedforward network, and the output is added back to the input (another residual connection).

## 11. Full GPT Model <a name="full-gpt-model"></a>🤖

This section defines the complete GPT model, integrating the Transformer blocks and embedding layers.

```python
# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
```

*   **`BigramLanguageModel` Class (Now a Full GPT):** The name is slightly misleading now, as this class implements the full GPT model, not just a bigram model.
    *   `__init__`:
        *   `self.token_embedding_table`:  Embeds the input tokens (character indices) into vectors of size `n_embd`.
        *   `self.position_embedding_table`:  Learns positional embeddings for each position in the input sequence (up to `block_size`).  Positional embeddings are crucial because self-attention itself is permutation-invariant; it doesn't inherently know the order of the tokens.
        *   `self.blocks`:  A sequence of `n_layer` Transformer blocks.  `nn.Sequential` makes it easy to apply these blocks one after another.  The `*` unpacks the list of blocks.
        *   `self.ln_f`:  A final layer normalization applied after all the Transformer blocks.
        *   `self.lm_head`:  A linear layer that projects the output of the Transformer to the vocabulary size.  This produces the logits for the next character prediction.
    *   `forward`:
        *   `tok_emb = self.token_embedding_table(idx)`:  Gets the token embeddings.
        *   `pos_emb = self.position_embedding_table(torch.arange(T, device=device))`:  Creates a tensor representing the positions (0, 1, 2, ..., T-1) and gets the corresponding positional embeddings.  `torch.arange` creates the sequence, and `device=device` ensures it's on the same device as the model.
        *   `x = tok_emb + pos_emb`:  Adds the token embeddings and positional embeddings. This is how positional information is incorporated into the model.
        *   `x = self.blocks(x)`:  Applies the sequence of Transformer blocks.
        *   `x = self.ln_f(x)`:  Applies the final layer normalization.
        *   `logits = self.lm_head(x)`:  Calculates the logits.
        *   The rest of the `forward` method calculates the cross-entropy loss if `targets` are provided, similar to the bigram model.
    *   `generate`:
        *   This method is almost identical to the bigram model's `generate` method, with one crucial difference:
        *   `idx_cond = idx[:, -block_size:]`:  This line *crops* the input sequence `idx` to the last `block_size` tokens.  This is because the model has a limited context window (defined by `block_size`). We only feed the most recent `block_size` tokens to the model for prediction.
*   `model = BigramLanguageModel()`: Creates an instance of the model.
*   `m = model.to(device)`: Moves the model to the specified device (GPU or CPU).
* `print(...)`: print out the number of parameters.

## 12. Training Loop (Full GPT) <a name="training-loop-gpt"></a>🚀

This section presents the training loop for the full GPT model.

```python
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
```

*   **Optimizer:** Creates an AdamW optimizer, just like in the bigram training loop.
*   **Training Loop:**
    *   The `for` loop iterates for `max_iters` steps.
    *   **Evaluation:**  `if iter % eval_interval == 0 or iter == max_iters - 1:`:  Periodically (every `eval_interval` steps) and at the very end of training, the `estimate_loss()` function is called to evaluate the model's performance on both the training and validation sets. This is printed to monitor progress and check for overfitting.
    *   **Batch Sampling, Forward Pass, Backward Pass, Parameter Update:**  These steps are the same as in the bigram training loop:  A batch is sampled, the forward pass is performed, gradients are calculated, and the model's parameters are updated.
*  **Text Generation:**
    *   `context = torch.zeros((1, 1), dtype=torch.long, device=device)`:  Creates an initial "zero" context (a single token with value 0).  This is the starting point for text generation.  It's important that this context is on the same device as the model.
    *   `print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))`:  Generates 2000 characters of text using the trained model and decodes the generated integer sequence back into a string.

## 13. Text Generation <a name="text-generation"></a>✍️
This is handled by the same generate method as in the model it self so the previous step covers it.

## 14. Layer Normalization Discussion <a name="layer-normalization"></a>📊
Layer norm was implemented with the other layers in step 9.

