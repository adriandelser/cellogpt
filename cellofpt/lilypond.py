import abjad

def generate_lilypond(notes, fingerings):
    # LilyPond header
    lilypond_template = r"""
\version "2.24.0"
\relative c, {
  \clef bass
  \key c \major
"""

    previous_pitch = "c"  # Start relative to middle C

    for note, fingering in zip(notes, fingerings):
        pitch_name = note[:-1].lower()  # Extract the pitch name (c, d, e, f, etc.) and convert to lowercase
        octave = int(note[-1])  # Extract the octave number

        # Calculate the relative pitch based on the previous pitch
        if octave > 3:
            pitch = pitch_name + "'" * (octave - 4)
        else:
            pitch = pitch_name + "," * (3 - octave)

        # Add the note with fingering to the template
        lilypond_template += f"  {pitch}4-\\markup {{ \\finger {fingering} }}\n"

        # Update the previous pitch
        previous_pitch = pitch

    # Close the LilyPond notation block
    lilypond_template += "}\n"
    with open('output.ly', 'w') as file:
        file.write(lilypond_template)

    return lilypond_template


# Example usage
notes = ['C2', 'D4', 'E2', 'F4']
fingerings = [1, 2, 3, 4]
lilypond_output = generate_lilypond(notes, fingerings)
print(lilypond_output)







