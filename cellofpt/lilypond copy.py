def generate_lilypond(notes, fingerings):
    # LilyPond header
    lilypond_template = r"""
\version "2.24.0"
\relative c' {
  \clef bass
  \key c \major
"""

    previous_octave = 2  # Start relative to C2
    current_pitch_int = ord('c')-ord('a') # =2
    for note, fingering in zip(notes, fingerings):
        pitch_name = note[:-1].lower()  # Extract the pitch name (c, d, e, f, etc.) and convert to lowercase
        pitch_int = ord(pitch_name)-ord('a') #0 to 7
        current_octave = int(note[-1])  # Extract the octave number


        # Determine the correct relative pitch
        if current_octave > previous_octave:
            pitch = pitch_name + "'" * (current_octave - previous_octave)
        elif current_octave < previous_octave:
            pitch = pitch_name + "," * (previous_octave - current_octave)
        else:
            pitch = pitch_name

        lilypond_template += f"  {pitch}4-\\markup {{ \\finger {fingering} }}\n"

        # Update the previous octave
        previous_octave = current_octave

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
