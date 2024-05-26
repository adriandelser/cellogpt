import mlx.core as mx
import mlx.nn as nn




Xin = mx.arange(0,8,1)
Xin = mx.expand_dims(Xin,axis=0)
print(Xin, Xin.shape)
