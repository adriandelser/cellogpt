import mlx.core as mx
from cellogpt.transformer import MusicFingeringModel, block_size
from data import iton, random_music, encode

# Hyperparameters
# block_size = 32  # Maximum context length for predictions
overlap_size = 8  # Overlap size between consecutive windows
n_embd = 16
dropout = 0.0
n_head = 4
num_notes = 16
num_fingers = 5

# Define the model (Assuming MusicFingeringModel class is defined as in the previous example)
model = MusicFingeringModel(n_head, num_notes, num_fingers).train(False)
model.load_weights('weights_s2s.safetensors')
model.freeze()

# Generate a random sequence of notes
total_notes = block_size * 10  # For demonstration, a sequence 3 times the block size
total_notes = 1000
#special patterns in third position:
# notes = mx.array(encode(random_music(total_notes)))
up_down = ['F3', 'G3', 'A4', 'G3', 'F3']
up_down2 = ['F3', 'G3', 'A4', 'B4','A4', 'G3', 'F3']
notes = mx.array(encode(random_music(total_notes)+up_down*5+up_down2*5))

# Initialize the start token
start_token = 0

def generate_fingerings_with_sliding_window(model, notes, start_token, block_size, overlap_size):
    context_length = block_size
    stride = context_length - overlap_size
    generated_fingerings = []

    total_notes = len(notes)
    num_windows = (total_notes - overlap_size) // stride

    for i in range(num_windows):
        start_idx = i * stride
        end_idx = start_idx + context_length
        if end_idx > total_notes:
            end_idx = total_notes
            start_idx = end_idx - context_length

        # Encoder input for the current window
        encoder_Xin = notes[start_idx:end_idx][None]  # shape (1, context_length)
        
        # Generate fingerings for the current window
        current_fingerings = model.generate_fingerings(encoder_Xin, start_token, context_length)
        print(current_fingerings)
        if i == 0:
            # For the first window, add all fingerings
            generated_fingerings.extend(current_fingerings)
        else:
            # For subsequent windows, add only the new predictions beyond the overlap
            generated_fingerings.extend(current_fingerings[overlap_size:])
    
    return generated_fingerings

# Generate fingerings using the sliding window approach
fingerings = generate_fingerings_with_sliding_window(model, notes, start_token, block_size, overlap_size)

input_notes = [iton[i.item()] for i in notes]

print(f"number of notes = {len(input_notes)}")
# fingerings = logits.argmax(axis=1).tolist()
print(f"Input notes:{input_notes}")
print(f"fingerings: {fingerings}")

# #write to lilypond file
from lilypond import absolute_to_lilypond
lilypond_output = absolute_to_lilypond(input_notes, fingerings,output_path='output/music.ly')