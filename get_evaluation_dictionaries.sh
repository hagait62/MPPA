#!/bin/bash

# This file enables to download easily evaluation dictionaries from MUSE dataset.
# You could pass languages as command line arguments,
# in order to download every language pair dictionary (if available).
# Default languages are as in the paper: en de fr es it pt.

LANGS=$1
LANGS=($LANGS)

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DICS=${2:-"$ROOT/data/dictionaries/"}

for src in "${LANGS[@]}"; do
    for tgt in "${LANGS[@]}"; do
        #echo src $src tgt $tgt
        if [ $src != $tgt ]; then
            wget -P $DICS "https://dl.fbaipublicfiles.com/arrival/dictionaries/${src}-${tgt}.5000-6500.txt"
        fi
    done
done