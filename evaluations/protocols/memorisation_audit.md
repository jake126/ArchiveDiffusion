# Protocol: Memorisation / Nearest-Neighbour Audit

## Purpose

Check whether generated or restored images are overly close to training frames.

This is an audit for responsible reporting. The project should not attempt to extract private or copyrighted training data.

## Method

1. Build embeddings for training frames.
2. Build embeddings for generated/restored outputs.
3. Retrieve nearest training neighbours for each output.
4. Compare distance scores and inspect visual similarity.

## Baselines

- nearest neighbour in pixel space
- nearest neighbour in feature space, optional

## Outputs

- nearest-neighbour contact sheets
- distance distribution plot
- short qualitative notes
