import mlx.nn as nn
import mlx.core as mx

# hyperparameters
block_size = 8 # what is the maximum context length for predictions?
n_embd = 8
dropout = 0.2
# ------------

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    def __call__(self,x):
        return self.forward(x)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        #NOTE nh = number of heads in the multihead attention block
        k = self.key(x)   # (B,T,C) -> (B,T,C//n_heads)
        q = self.query(x) # (B,T,C) -> (B,T,C//n_heads)
        # compute attention scores ("affinities")
        # wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        #in mlx this is how you would swap the second and third axes
        wei = q @ k.transpose(0,-1,-2) * C**-0.5 # (B, T, C//nh) @ (B, C//nh, T) -> (B, T, T)

        #NOTE no trilling, we can see the future notes
        # mx.where(mx.tril(wei,k=0) == 0, -mx.inf, wei)
        wei = nn.softmax(wei, axis=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C//nh)
        out = wei @ v # (B, T, T) @ (B, T, C//nh) -> (B, T, C//nh)
        return out

class MultiHeadAttention(nn.Module):
    '''Multiple heads running in parallel'''
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = nn.Linear(n_embd, n_embd)
        #dropout here too maybe?
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x:mx.array):
        return self.forward(x)

    def forward(self, x):
        out = mx.concatenate([h(x) for h in self.heads], axis=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear( 4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def __call__(self, x:mx.array):
        return self.forward(x)
    
    def forward(self, x):
        return self.net(x)


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

# super simple model
class MusicFingeringModel(nn.Module):

    def __init__(self,n_head,vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) #block size is context length
        self.blocks = nn.Sequential(
            Block(n_embd, n_head),
            Block(n_embd, n_head),
            Block(n_embd, n_head),
            nn.LayerNorm(n_embd),
        )
        self.lm_head = nn.Linear(n_embd, 5)

    def __call__(self, idx:mx.array):
        return self.forward(idx)

    def forward(self, idx:mx.array):
        '''
        Note on shapes:
        B is batch size (batch per forward pass)
        T is block size (context length)
        C is n_embd 
        '''
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers #NOTE no more targets
        tok_emb = self.token_embedding_table(idx) # (B,T,C) where C is n_embd
        pos_emb = self.position_embedding_table(mx.arange(T)) # (T, C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) 
        logits = self.lm_head(x) # (B,T,num_fingers = 5)


        B, T, C = logits.shape
        logits = logits.reshape(B*T, C)
        return logits

    def generate(self, idx, max_new_tokens):
        '''Generate new stuff as in a language model 
        NOTE currently unnused but could be adapted to generate music'''
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



