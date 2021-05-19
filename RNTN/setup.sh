#!/bin/bash

# Get trees
data=trainDevTestTrees_PTB.zip
curl -O https://nlp.stanford.edu/sentiment/$data
unzip $data 
rm -f $data

# Convert trees
python tree.py

# Create directory for saved models
mkdir models
