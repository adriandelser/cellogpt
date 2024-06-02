import math, random
import mlx.core as mx

'''Generate fake music data'''
# start_token = '<START>'
# here are all the unique notes that occur in this music
notes = [chr(num%7+ord('A'))+str(math.floor(num/7)+2) for num in range(2,18)] #effectively chars
# tokens = notes + [start_token]
fingerings:list = [0,1,3,4,0,1,3,4,0,1,2,4,0,1,2,4] #fingering for C major scale up to D natural on A string
# create a mapping from characters to integers
ntoi:dict[str,int] = {note:idx for idx, note in enumerate(notes)} #note to integer represenation of note [stoi]
iton:dict[int,str] = {idx:note for note, idx in ntoi.items()} #integer to note [itos]
itof:dict[int,int] = {i:f for i, f in enumerate(fingerings)} #integer to fingering value 
ntof:dict[str,int] = {iton[i]:f for i, f in itof.items()} #note to fingering value
encode = lambda notes: [ntoi[note] for note in notes]
decode = lambda ints: [iton[i] for i in ints] 

def random_music(n:int, low:str = 'C2', high:str = 'D4')->list[str]:
    lowest, highest = ntoi[low], ntoi[high]
    rand_notes = [random.randint(lowest,highest) for _ in range(n)]
    return decode(rand_notes)

def get_fingerings(notes:list[str])->list[int]:
    return[ntof[note] for note in notes]


vocab_size = len(notes) #=16

def get_data(training_split:float = 0.9):
    # Train and test splits
    music = random_music(1000) #list[int] make some random notes
    fingerings = get_fingerings(music)
    music = mx.array(encode(music), dtype=mx.int32)
    fingerings = mx.array(fingerings, dtype=mx.int32)
    n = int(training_split*len(music)) # first 90% will be train, rest val
    train_data = music[:n], fingerings[:n]
    val_data = music[n:], fingerings[n:]
    return train_data, val_data


#special patterns in third position:
up_down = ['F3', 'G3', 'A4', 'G3', 'F3']
up_down_fingerings = [1,2,4,2,1]

up_down2 = ['F3', 'G3', 'A4', 'B4','A4', 'G3', 'F3']
up_down2_fingerings = [2,4,0,1,0,4,2]

n_extra = 40
def get_data_extra(training_split:float = 0.9):
    # Train and test splits
    music = random_music(1000) #list[int] make some random notes
    fingerings = get_fingerings(music)
    music += up_down*n_extra + up_down2*n_extra
    fingerings += up_down_fingerings*n_extra + up_down2_fingerings*n_extra

    music = mx.array(encode(music), dtype=mx.int32)
    fingerings = mx.array(fingerings, dtype=mx.int32)
    n = int(training_split*len(music)) # first 90% will be train, rest val
    train_data = music[:n], fingerings[:n]
    val_data = music[n:], fingerings[n:]
    return train_data, val_data
