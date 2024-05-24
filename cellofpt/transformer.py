import mlx
import mlx.nn as nn
import mlx.core as mx

import math, random
import mlx.optimizers
import numpy as np
# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 16 # what is the maximum context length for predictions?
max_iters = 100
learning_rate = 1e-3
eval_interval = 20
eval_iters = 200
n_embd = 16
# ------------

mx.random.seed(1337)

# here are all the unique characters that occur in this text
notes = [chr(num%7+ord('A'))+str(math.floor(num/7)+2) for num in range(2,18)] #effectively chars
fingerings:list = [0,1,3,4,0,1,3,4,0,1,2,4,0,1,2,4] #fingering for C major scale up to D natural on A string
# create a mapping from characters to integers
ntoi:dict[str,int] = {note:idx for idx, note in enumerate(notes)} #note to integer represenation of note [stoi]
iton:dict[int,str] = {idx:note for note, idx in ntoi.items()} #integer to note [itos]
itof:dict[int,int] = {i:f for i, f in enumerate(fingerings)} #integer to fingering value 
ntof:dict[str,int] = {iton[i]:f for i, f in itof.items()} #note to fingering value

def random_music(n:int, low:str = 'C2', high:str = 'D4')->list[int]:
    lowest, highest = ntoi[low], ntoi[high]
    rand_notes = [random.randint(lowest,highest) for _ in range(n)]
    return rand_notes

def get_fingerings(notes:list[int])->list[int]:
    return[itof[note] for note in notes]

music = random_music(1000) #list[int] make some random notes
fingerings = get_fingerings(music)


# chars = sorted(list(set(text)))
vocab_size = len(notes)
embs = nn.Embedding(vocab_size,n_embd)
# encode = lambda s: [ntoi[n] for n in s] # encoder: take a string, output a list of integers
# decode = lambda l: ''.join([iton[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
music = mx.array(music, dtype=mx.int32)
fingerings = mx.array(fingerings, dtype=mx.int32)
n = int(0.9*len(music)) # first 90% will be train, rest val
train_data = music[:n], fingerings[:n]
val_data = music[n:], fingerings[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = mx.random.randint(0, len(data[0]) - block_size, (batch_size,))
    # print([[i,i+block_size] for i in ix])
    # print([data[i:i+block_size] for i in ix[:5]])
    x = mx.stack([data[0][i:i+block_size] for i in ix.tolist()]) #notes, need tolist otherwise type errors
    y = mx.stack([data[1][i:i+block_size] for i in ix.tolist()]) #fingerings
    #this line just adds an extra embedding dimension, which is 1 as we are not really embedding anything yet
    # x,y = mx.expand_dims(x,axis=-1), mx.expand_dims(y,axis=-1)
    return x, y




# @torch.no_grad() #does this exist in mlx?
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = mx.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            # logits, loss = model(X, Y)
            loss = loss_fn(model,X,Y)
            # print(loss)
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
        # self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # self.dropout = nn.Dropout(dropout)

    def __call__(self,x):
        return self.forward(x)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        # wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        #in mlx this is how you would swap the second and third axes
        wei = q @ k.transpose(0,-1,-2) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)

        #no trilling, we can see the future notes
        # wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        # mx.where(mx.tril(wei,k=0) == 0, -mx.inf, wei)
        wei = nn.softmax(wei, axis=-1) # (B, T, T)
        # wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    '''Multiple heads running in parallel'''
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = nn.Linear(n_embd, n_embd)

    def __call__(self, x:mx.array):
        return self.forward(x)

    def forward(self, x):
        # out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = mx.concatenate([h(x) for h in self.heads], axis=-1)
        out = self.proj(out)
        return out
    
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear( 4*n_embd, n_embd),
            # nn.Dropout(dropout),
        )

    def __call__(self, x:mx.array):
        return self.forward(x)
    
    def forward(self, x):
        return self.net(x)

# head_size = 16
# n_heads = 4
# train = get_batch('train')[0]
# train_embd = embs(train)
# print(f"{train_embd.shape=}")
# h = Head(head_size)
# out = h.forward(train_embd)
# mh = MultiHeadAttention(num_heads=n_heads, head_size=head_size//n_heads)
# out = mh.forward(train_embd)
# ff = FeedFoward(n_embd)
# out = ff.forward(train_embd)
# print(f"feedforward: {out.shape=}")


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

    def __call__(self, x:mx.array):
        return self.forward(x)

    def forward(self, x): 
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) #block size is context length
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            nn.LayerNorm(n_embd),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def __call__(self, idx:mx.array):
        return self.forward(idx)

    def forward(self, idx:mx.array):
        
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers #NOTE no more targets
        tok_emb = self.token_embedding_table(idx) # (B,T,C) where C is n_embd
        pos_emb = self.position_embedding_table(mx.arange(T)) # (T, C)
        x = tok_emb + pos_emb # (B,T,C)
        # x = self.sa_heads(x) # (B,T,C)
        # x = self.ffwd(x) # (B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B,T,vocab_size)


        B, T, C = logits.shape
        # logits = logits.view(B*T, C)
        # targets = targets.view(B*T)
        logits = logits.reshape(B*T, C)
        # targets = targets.reshape(B*T)
        # loss = nn.losses.cross_entropy(logits, targets)

        return logits

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop idx to the last block_size tokens ie in case we give more context than block_size somehow
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = nn.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            # idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx_next = mx.random.categorical(probs, num_samples=1) # (B, 1)

            # append sampled index to the running sequence
            idx = mx.concatenate((idx, idx_next), axis=1) # (B, T+1)
        return idx



# print(get_batch('val'))
train_batch = get_batch('train')[0] #0 for notes, 1 for fingerings  
print(f"{train_batch.shape=}")

model = BigramLanguageModel()
def loss_fn(model, X, y:mx.array):
    logits = model(X)
    BT = logits.shape[0]
    targets = y.reshape(BT)
    loss = nn.losses.cross_entropy(logits, targets)
    return mx.mean(loss) #do i need to take the mean?
    return mx.mean(nn.losses.cross_entropy(model(X), y))
# m = model.to(device)

# # create a mlx optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# Create the gradient function and the optimizer
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
optimizer = mlx.optimizers.AdamW(learning_rate = learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        model.freeze()
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train'].item():.4f}, val loss {losses['val'].item():.4f}")
        model.unfreeze()

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits = model(xb)

    loss, grads = loss_and_grad_fn(model, xb, yb)
    # Update the model with the gradients. So far no computation has happened.
    #NOTE I assume this does zero grad? Haven't seen any explicit zero gradding in the mlx docs. Need to make sure...
    optimizer.update(model, grads) 
    # Compute the new parameters but also the optimizer state.
    mx.eval(model.parameters(), optimizer.state)

# # generate from the model
model.freeze()
Xin = mx.arange(0,16,1)
print(Xin)
Xin = mx.expand_dims(Xin,axis=-1)
Xin = mx.random.randint(0,16,(5, block_size))
logits = model(Xin)
# print(logits)
input_notes = [iton[i.item()] for i in mx.flatten(Xin)]
fingerings = logits.argmax(axis=1).tolist()
print(f"Input notes:{input_notes}")
print(f"fingerings: {fingerings}")


#save to file
def generate_lilypond(notes, fingerings):
    # LilyPond header
    lilypond_template = r"""
\version "2.24.0"
\relative c' {
  \clef bass
  \key c \major
"""

    previous_octave = 2  # Start relative to C2

    for note, fingering in zip(notes, fingerings):
        pitch_name = note[:-1].lower()  # Extract the pitch name (c, d, e, f, etc.) and convert to lowercase
        current_octave = int(note[-1])  # Extract the octave number

        # Determine the correct relative pitch
        if current_octave > previous_octave:
            pitch = pitch_name + "'" * (current_octave - previous_octave)
        elif current_octave < previous_octave:
            pitch = pitch_name + "," * (previous_octave - current_octave)
        else:
            pitch = pitch_name

        lilypond_template += f"  {pitch}4-\\markup {{ \\finger {fingering} }}\n"

        # Update the previous octave
        previous_octave = current_octave

    # Close the LilyPond notation block
    lilypond_template += "}\n"
    with open('output.ly', 'w') as file:
        file.write(lilypond_template)

    return lilypond_template

# Example usage
# notes = ['C2', 'D4', 'E2', 'F4']
# fingerings = [1, 2, 3, 4]
lilypond_output = generate_lilypond(input_notes, fingerings)
print(lilypond_output)
