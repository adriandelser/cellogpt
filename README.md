
  
# Cello Fingering Prediction with Sequence-to-sequence Transformers

## Project Overview

This project aims to generate fingerings for cello music (ie which fingers to use to play each note on the sheet music) using encoder-decoder transformers. The goal is to apply the architecture of sequence-to-sequence transformers, "translating" from music notes to fingerings instead of translating one language to another.
**Note:** This project was motivated by not being able to find adequate fingerings for my sheet music, needing to watch slow motion video of online performances to decipher how to play certain sections. It is also a good opportunity to improve my knowledge of Apple's mlx library. 
**Note:** This project is brand new (started 24/05/2024) and a work in progress. It is on github so I can keep track of my progress.

## Previous Work
Previous research on this topic exists, mostly for piano music (the task of generating fingerings for piano music is referred to Automatic Piano Fingering, or APF). It has shown that sequence-to-sequence transformers (along with other learning-based methods) can be useful for this task. I haven't found databases for the cello yet, so this project is a personal challenge to see what can be achieved using resources available to me.

Some references:
- [Masahiro Suzuki, "Piano Fingering Estimation and Completion with Transformers" Yamaha Corp., 2021.](https://archives.ismir.net/ismir2021/latebreaking/000007.pdf)
- [Ramoneda et al., "Automatic Piano Fingering from Partially Annotated Scores using Autoregressive Neural Networks" Association for Computing Machinery, 2022.](https://dl.acm.org/doi/10.1145/3503161.3548372)



## Project Plan (feasability unknown)

### Step 1: Overfitting a simple dataset 
- I will generate pseudo-random notes in the C major scale spanning from the lowest note on the cello (C2) to the highest note on the A string playable in first position (D4). There are 16 notes in this range. The fingerings for these notes are known, thus we have an initial dataset to play with. **DONE**
- For the transformer model, I will start with a 'note level' model (ie 'character level', without a note tokenizer (very much future work). The main difference with this transformer and an Large Language Model decoder-only tranformer is the presence of cross-attention due to the decoder, as well as allowing 'future' knowledge, ie. instead of cutting off the top triangle of the matrices involved in the attention dot-product, we allow knowledge of future notes. I will need to somehow embed the notes (equivalent to 'vocabulary' in text-based models) and attempt to overfit the simple dataset to see if the decoder obtains coherent results. **DONE**
- I will try to achieve this locally on an M1 Max processor using mlx as the main array and deep learning library.

### Step 2: Expand vocabulary
- Include all notes in the scale
- Label some real pieces? I'll see what the existing research says.
- Depends on how well step 1 goes :)


## Contributing

This project is a work in progress. Suggestions are welcome.

