'''run: make inference'''

import mlx.core as mx
from seq2seq_transformer import MusicFingeringModel, block_size
from data import iton
import sys

vocab_size = 16

model = MusicFingeringModel(n_head=4, num_fingers=5, num_notes=16)
model.load_weights('weights_s2s.safetensors')
model.freeze()

mx.random.seed(1339)

num_note_groups = 1
Xin = mx.random.randint(0,16,(num_note_groups, block_size)) #num notes = block_size * num_note_groups
Fin = mx.zeros((num_note_groups, 1)) #start token will be zero apparently
Fin = mx.array([[1, 3, 2, 4, 1, 3, 1, 3]])
Fin = mx.ones((num_note_groups, 1)) #start token will be zero apparently
logits = model(Xin)
print(f"{logits.shape=}")
print(logits.tolist())
# sys.exit()
input_notes = [iton[i.item()] for i in mx.flatten(Xin)]
print(f"number of notes = {len(input_notes)}")
fingerings = logits.argmax(axis=1).tolist()
print(f"Input notes:{input_notes}")
print(f"fingerings: {fingerings}")

#write to lilypond file
from lilypond import absolute_to_lilypond
lilypond_output = absolute_to_lilypond(input_notes, fingerings, output_path='output/music_s2s.ly')