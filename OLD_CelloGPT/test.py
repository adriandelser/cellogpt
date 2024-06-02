import mlx.core as mx
import mlx.nn as nn

embedding = nn.Embedding(num_embeddings=5,dims=2)
a = embedding([0,1,2,3,4,5])

print(a)