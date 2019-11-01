#!/bin/bash

# This file enables to download easily monolingual word embeddings from fasttext library.
# You could pass languages as command line arguments.

LANGS=$1
LANGS=($LANGS)

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VECS_PATH=${2:-"$ROOT/data/vecs/"}
#VECS="$ROOT/data/vecs"

for lang in "${LANGS[@]}"; do
    wget -P  $VECS_PATH "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.${lang}.vec"
done