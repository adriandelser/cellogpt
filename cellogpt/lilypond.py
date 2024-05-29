from data import ntof
from pathlib import Path

def absolute_to_lilypond(notes:list[str],fingerings, output_path = 'output/music.ly'):
    """
    Convert absolute music notes to LilyPond relative notation.

    Args:
    notes (list of str): List of absolute notes (e.g., ["C2", "B2", "G3"])

    Returns:
    list of str: List of notes in LilyPond relative notation
    """

    # LilyPond header
    lilypond_template = r"""
        \version "2.24.0"
        \relative c, {
        \clef bass
        \key c \major
        """
    # Dictionary to map note letters to their respective semitone positions
    note_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}

    prev_note = note_map.get('C') + 2*7 #this encodes C2
    # result = []
    # for i in range(0, len(notes)):
    for note, fingering in zip(notes, fingerings):

        current_note = note_map.get(note[0]) + int(note[1])*7
        diff = current_note-prev_note
        lilypond_note = note[0].lower()
        if diff>3:
            lilypond_note += "'"*((diff-4)//7 + 1)
        elif diff<-3:
            lilypond_note += ","*(abs(diff+4)//7+1)
        
        lilypond_template += f"  {lilypond_note}4"
        
        # print(f"{type(fingering)=}")
        if ntof[note] != fingering:
            lilypond_template+="-\\tweak color #red "

        #for fingerings above uncomment this line
        # lilypond_template+= f"-{fingering}\n" #above
        #for fingerings below uncomment this line
        lilypond_template += f"-\\markup {{ \\finger {fingering} }}\n" #below


        prev_note=current_note
    # Close the LilyPond notation block
    lilypond_template += "}\n"
    path = Path(output_path)
    path = path.resolve()
    with open(path, 'w') as file:
        file.write(lilypond_template)
    return lilypond_template
        
if __name__ ==  '__main__':

    # Example usage:
    notes = ['E3', 'D3', 'C2', 'F3', 'C2', 'E3']
    fingerings = [1,2,3,4,1,2]

    lilypond_notes = absolute_to_lilypond(notes, fingerings)
    print(lilypond_notes)  # Output: ['c', 'g,', 'f', 'c']

    '''
    3: ABCDEFG
    2: ABCDEFG
    1: ABCDEFG
    '''

