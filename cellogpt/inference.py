'''run: make inference'''

import mlx.core as mx
from cellogpt.transformer import MusicFingeringModel, block_size
from data import iton, itof
import sys
vocab_size = 16
start_token = 5

print(f"{block_size=}")

model = MusicFingeringModel(n_head=4, num_notes=16, num_fingers=5).train(False)
model.load_weights('weights_s2s.safetensors')
model.freeze()
# mx.random.seed(1339)

num_notes = block_size
notes = mx.random.randint(0,16,(num_notes,)) #num notes = block_size * num_note_groups
#encoder input is first block of notes with extra dimension for compatibility 
encoder_Xin = notes[:block_size][None] 
fingerings = model.generate_fingerings(encoder_Xin, start_token, max_length = block_size)

input_notes = [iton[i.item()] for i in notes]

print(f"number of notes = {len(input_notes)}")
# fingerings = logits.argmax(axis=1).tolist()
print(f"Input notes:{input_notes}")
print(f"fingerings: {fingerings}")

# #write to lilypond file
from lilypond import absolute_to_lilypond
lilypond_output = absolute_to_lilypond(input_notes, fingerings,output_path='output/music.ly')

