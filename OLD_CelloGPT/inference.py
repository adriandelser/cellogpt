'''run: make inference'''

import mlx.core as mx
from transformer import MusicFingeringModel, block_size
from data import iton
vocab_size = 16
print(block_size)

model = MusicFingeringModel(n_head=4, vocab_size=vocab_size)
model.load_weights('weights.safetensors')

mx.random.seed(1339)

num_note_groups = 2
num_notes = 2000
notes = mx.random.randint(0,16,(num_notes,)) #num notes = block_size * num_note_groups
fingerings = []
Xin = mx.array([[0]]) #start token is 0 why not
for idx in range(num_notes):
    # print(Xin, notes[idx][None,None])
    Xin = mx.concatenate((Xin,notes[idx][None,None]),axis=1)
    Xin = Xin[:,-block_size:]
    # print(f"{Xin=}, {Xin.shape=}")
    logits = model(Xin)
    # print(f"{logits=}")
    fingering = logits[-1].argmax()
    fingerings.append(fingering.item())

# print(fingerings, len(fingerings))
# print(f"{logits.shape=}")
# print(logits.tolist())

input_notes = [iton[i.item()] for i in mx.flatten(notes)]

print(f"number of notes = {len(input_notes)}")
# fingerings = logits.argmax(axis=1).tolist()
print(f"Input notes:{input_notes}")
print(f"fingerings: {fingerings}")

# #write to lilypond file
from lilypond import absolute_to_lilypond
lilypond_output = absolute_to_lilypond(input_notes, fingerings)