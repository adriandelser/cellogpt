'''run: make inference'''

import mlx.core as mx
from transformer import MusicFingeringModel, block_size
from data import iton
vocab_size = 16

model = MusicFingeringModel(n_head=4, vocab_size=vocab_size)

mx.random.seed(1338)

model.load_weights('weights.safetensors')
Xin = mx.random.randint(0,16,(20, block_size))
logits = model(Xin)
input_notes = [iton[i.item()] for i in mx.flatten(Xin)]
print(f"{len(input_notes)=}")
fingerings = logits.argmax(axis=1).tolist()
print(f"Input notes:{input_notes}")
print(f"fingerings: {fingerings}")

#write to lilypond file
from lilypond import absolute_to_lilypond
lilypond_output = absolute_to_lilypond(input_notes, fingerings)