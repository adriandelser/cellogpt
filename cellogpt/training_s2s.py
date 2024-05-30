import mlx
import mlx.core as mx
import mlx.nn as nn
from data import get_data
import mlx.optimizers
from seq2seq_transformer import MusicFingeringModel, block_size

#hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
max_iters = 200
learning_rate = 1e-3
eval_interval = 20
eval_iters = 200
vocab_size = 16 #16 possible notes


train_data, val_data = get_data(training_split=0.9)

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = mx.random.randint(0, len(data[0]) - block_size, (batch_size,))
    x = mx.stack([data[0][i:i+block_size] for i in ix.tolist()]) #notes, need tolist otherwise type errors
    y = mx.stack([data[1][i:i+block_size] for i in ix.tolist()]) #fingerings
    return x, y

# @torch.no_grad() #does this exist in mlx?
def estimate_loss(model:nn.Module):
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


def loss_fn(model, X, y:mx.array):
    logits = model(X)
    BT = logits.shape[0]
    targets = y.reshape(BT)
    # print(logits.shape,y.shape, targets.shape)
    # print(X.shape, y.shape, logits.shape)
    loss = nn.losses.cross_entropy(logits, targets)
    # print(f"{loss.shape=}")
    return mx.mean(loss) 


if __name__=='__main__':
    from mlx.utils import tree_flatten

    model = MusicFingeringModel(n_head=4, num_fingers=5, num_notes=16)
    num_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"{num_params=}")
  
    # Create the gradient function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    # # create a mlx optimizer
    optimizer = mlx.optimizers.AdamW(learning_rate = learning_rate)

    train_losses = []
    val_losses = []
    for iter in range(max_iters):
        if iter == 100:
            learning_rate = 2e-4
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            model.freeze()
            losses = estimate_loss(model)
            train_loss = losses['train'].item()
            val_loss = losses['val'].item()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
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
    # model.freeze()
    model.save_weights('weights_s2s.safetensors')


    # import matplotlib.pyplot as plt
    # plt.title("Training loss")
    # plt.ylabel("Loss")
    # plt.xlabel("iteration")
    # plt.plot(train_losses)
    # plt.show()
    
