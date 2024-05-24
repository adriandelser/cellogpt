import math
import random
#num_notes = 16 # if counting just the main notes ie A-G from bottom C to first D on A string
notes = [chr(num%7+ord('A'))+str(math.floor(num/7)) for num in range(2,18)]
print(f"{notes=}")
ntoi:dict[str,int] = {note:idx for idx, note in enumerate(notes)} #note to integer represenation of note
iton:dict[int,str] = {idx:note for note, idx in ntoi.items()} #integer to note
fingerings:list = [0,1,3,4,0,1,3,4,0,1,2,4,0,1,2,4] #fingering for C major scale up to D natural on A string
itof:dict[int,int] = {i:f for i, f in enumerate(fingerings)} #integer to fingering value 
ntof:dict[str,int] = {iton[i]:f for i, f in itof.items()} #note to fingering value

def random_music(n:int, low:str = 'C0', high:str = 'D2')->list[int]:
    lowest, highest = ntoi[low], ntoi[high]
    rand_notes = [random.randint(lowest,highest) for _ in range(n)]
    return rand_notes
    return [iton[i] for i in rand_notes] #this would return the string values

def get_fingerings(notes:list[int])->list[int]:
    return[itof[note] for note in notes]


notes_range = ['C0', 'D2'] #min, max
random_notes = random_music(200, *notes_range)
print(get_fingerings(random_notes))



