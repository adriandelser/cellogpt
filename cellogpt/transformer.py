import mlx.nn as nn
import mlx.core as mx

# hyperparameters
block_size = 16 # what is the maximum context length for predictions?
n_embd = 16
dropout = 0.0
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
        #in mlx this is how you would swap the second and third axes
        wei = q @ k.transpose(0,-1,-2) * C**-0.5 # (B, T, C//nh) @ (B, C//nh, T) -> (B, T, T)

        # mask
        wei = mx.where(mx.tril(wei,k=0) == 0, -mx.inf, wei) #apply mask
        wei = nn.softmax(wei, axis=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C//nh)
        out = wei @ v # (B, T, T) @ (B, T, C//nh) -> (B, T, C//nh)
        return out
    
class CrossAttentionHead(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def __call__(self,x1:mx.array, x2: mx.array):
        return self.forward(x1, x2)

    def forward(self, x1, x2):
        '''keys and values from x1 (encoder stack output), queries from x2 (decoder)'''
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x2.shape #C is n_embd
        # NOTE nh = number of heads in the multihead attention block
        # let's do query from decoder, key and value from encoder like in the original paper
        # TODO see what difference other combinations make
        # NOTE x1 from encoder, x2 from decoder 
        k = self.key(x1)   # (B,T1,C) -> (B,T1,C//n_heads)
        v = self.value(x1) # (B,T1,C//nh)
        q = self.query(x2) # (B,T2,C) -> (B,T2,C//n_heads)
        # compute attention scores ("affinities")
        #in mlx this is how you would swap the second and third axes
        wei = q @ k.transpose(0,-1,-2) * C**-0.5 # (B, T2, C//nh) @ (B, C//nh, T1) -> (B, T2, T1)
        wei = mx.where(mx.tril(wei,k=0) == 0, -mx.inf, wei) #apply mask
        wei = nn.softmax(wei, axis=-1) # (B, T2, T1)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        out = wei @ v # (B, T2, T1) @ (B, T1, C//nh) -> (B, T2, C//nh)
        return out

class MultiHeadSelfAttention(nn.Module):
    '''Multiple heads running in parallel'''
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x:mx.array):
        return self.forward(x)

    def forward(self, x):
        out = mx.concatenate([h(x) for h in self.heads], axis=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class MultiHeadCrossAttention(nn.Module):
    '''Multiple heads running in parallel'''
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = [CrossAttentionHead(head_size) for _ in range(num_heads)]
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x1:mx.array,x2:mx.array):
        return self.forward(x1, x2)

    def forward(self, x1, x2):
        out = mx.concatenate([h(x1,x2) for h in self.heads], axis=-1)
        out = self.proj(out) # (B, T2, C)
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
    

class EncoderBlock(nn.Module):
    '''Encoder block based on Attention is all you need.
    NOTE we do layer norm before the attention blocks instead of after'''
    def __init__(self, n_embd:int, n_head:int=4):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadSelfAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def __call__(self,x:mx.array):
        return self.forward(x)
    
    def forward(self, x:mx.array):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class DecoderBlock(nn.Module):
    '''Decoder block based on Attention is all you need.
    NOTE we do layer norm before the attention blocks instead of after'''

    def __init__(self, n_embd:int, n_head:int=4):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadSelfAttention(n_head, head_size)
        self.ca = MultiHeadCrossAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
    
    def __call__(self,x1, x2):
        return self.forward(x1, x2)
    
    def forward(self, x1, x2):
        '''x1 from encoder, x2 from decoder'''
        x2 = x2 + self.sa(self.ln1(x2))
        x = x2 + self.ca(x1, self.ln2(x2))
        # print(f"{x.shape=}")
        x = x + self.ffwd(self.ln3(x)) # (B,T2,C)
        return x


# super simple model
class MusicFingeringModel(nn.Module):

    def __init__(self, n_head, num_notes, num_fingers):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.notes_token_embedding_table = nn.Embedding(num_notes, n_embd)
        self.notes_position_embedding_table = nn.Embedding(block_size, n_embd) #block size is context length
        self.fingering_token_embedding_table = nn.Embedding(num_fingers, n_embd)
        self.fingering_position_embedding_table = nn.Embedding(block_size, n_embd) #block size is context length
        self.encoder_stack = nn.Sequential(
            EncoderBlock(n_embd, n_head),
            EncoderBlock(n_embd, n_head),
            EncoderBlock(n_embd, n_head),
        ) #three encoders
        self.decoder_stack = [DecoderBlock(n_embd, n_head) for _ in range(3)] #three Decoder blocks
        
        #no softmax at the output as argmax is sufficient on the output of the linear layer
        self.lm_head = nn.Sequential(nn.LayerNorm(n_embd), nn.Linear(n_embd, num_fingers)) #where 5 is the vocab size, ie 5 possible fingerings [0,1,2,3,4]

    def __call__(self, encoder_idx:mx.array, decoder_idx:mx.array)->mx.array:
        return self.forward(encoder_idx, decoder_idx)
    
    def forward_encoder(self, encoder_idx):
        B, T1 = encoder_idx.shape
        notes_emb = self.notes_token_embedding_table(encoder_idx)
        notes_pos_emb = self.notes_position_embedding_table(mx.arange(T1))
        x1 = notes_emb + notes_pos_emb
        x1 = self.encoder_stack(x1)
        return x1

    def forward_decoder(self, x1, decoder_idx):
        B, T2 = decoder_idx.shape
        fingering_emb = self.fingering_token_embedding_table(decoder_idx)
        fingering_pos_emb = self.fingering_position_embedding_table(mx.arange(T2))
        x2 = fingering_emb + fingering_pos_emb
        for decoder in self.decoder_stack:
            x2 = decoder(x1, x2)
        logits = self.lm_head(x2)
        # B, T, C = logits.shape 
        # logits = logits.reshape(B*T, C)
        return logits
    
    def generate_fingerings(self, encoder_idx, start_token, max_length):
        x1 = self.forward_encoder(encoder_idx)
        decoder_input = [start_token]  # Start with the start token
        for _ in range(max_length):
            decoder_idx = mx.array(decoder_input).reshape(1, -1)
            logits = self.forward_decoder(x1, decoder_idx)
            next_fingering = logits[:, -1, :].argmax(axis=-1).item()
            decoder_input.append(next_fingering)
        return decoder_input[1:]  # Remove the start token from the output

    def forward(self, encoder_idx:mx.array, decoder_idx:mx.array):
        '''
        Note on shapes:
        B is batch size (batch per forward pass)
        T is block size (context length)
        C is n_embd 
        '''
        # print(f"input is {idx=}")
        B, T1 = encoder_idx.shape
        B, T2 = decoder_idx.shape
        # idx and targets are both (B,T) tensor of integers #NOTE no more targets

        notes_emb = self.notes_token_embedding_table(encoder_idx) # (B,T1,C) where C is n_embd
        notes_pos_emb = self.notes_position_embedding_table(mx.arange(T1)) # (T1, C)

        fingering_emb = self.fingering_token_embedding_table(decoder_idx) # (B,T2,C) where C is n_embd
        fingering_pos_emb = self.fingering_position_embedding_table(mx.arange(T2)) # (T2, C)

        x1 = notes_emb + notes_pos_emb # (B,T1,C)

        x1 = self.encoder_stack(x1) 
        
        x2 = fingering_emb + fingering_pos_emb #(B,T2,C)
        for decoder in self.decoder_stack:
            x2 = decoder(x1,x2) 
        
        # print(f"{x2.shape=}")
        logits = self.lm_head(x2) # (B,T,num_fingers = 5)

        B, T, C = logits.shape
        logits = logits.reshape(B*T, C)
        # print(f"{logits.shape=}")
        # print(f"{logits[0]=}")
        # if mx.all(mx.isnan(logits[0])):
        #     print(f"{idx=}\n{self.fingering_token_embedding_table(idx)=},\n {self.notes_token_embedding_table(idx)=}")
        #     import sys
        #     sys.exit()
        return logits

  


