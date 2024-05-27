import math, random
import mlx.core as mx

'''Generate fake music data'''

# here are all the unique notes that occur in this music
notes = [chr(num%7+ord('A'))+str(math.floor(num/7)+2) for num in range(2,18)] #effectively chars
fingerings:list = [0,1,3,4,0,1,3,4,0,1,2,4,0,1,2,4] #fingering for C major scale up to D natural on A string
# create a mapping from characters to integers
ntoi:dict[str,int] = {note:idx for idx, note in enumerate(notes)} #note to integer represenation of note [stoi]
iton:dict[int,str] = {idx:note for note, idx in ntoi.items()} #integer to note [itos]
itof:dict[int,int] = {i:f for i, f in enumerate(fingerings)} #integer to fingering value 
ntof:dict[str,int] = {iton[i]:f for i, f in itof.items()} #note to fingering value

def random_music(n:int, low:str = 'C2', high:str = 'D4')->list[int]:
    lowest, highest = ntoi[low], ntoi[high]
    rand_notes = [random.randint(lowest,highest) for _ in range(n)]
    return rand_notes

def get_fingerings(notes:list[int])->list[int]:
    return[itof[note] for note in notes]


vocab_size = len(notes) #=16

def get_data(training_split:float = 0.9):
    # Train and test splits
    music = random_music(1000) #list[int] make some random notes
    fingerings = get_fingerings(music)
    music = mx.array(music, dtype=mx.int32)
    fingerings = mx.array(fingerings, dtype=mx.int32)
    n = int(training_split*len(music)) # first 90% will be train, rest val
    train_data = music[:n], fingerings[:n]
    val_data = music[n:], fingerings[n:]
    return train_data, val_data