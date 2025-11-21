#!/bin/bash

# Folder for logs
OUTDIR="results_p_sweep"
mkdir -p "$OUTDIR"

echo "Running p' sweep for Index, Mixer and Random with 20 epochs..."
echo "Output folder: $OUTDIR"
echo

# Models to evaluate
MODELS=("index" "mixer" "random")

# p' values
PVALUES=("0.00" "0.05" "0.10" "0.15" "0.20" "0.25")

for M in "${MODELS[@]}"; do
    for P in "${PVALUES[@]}"; do
        echo "Model=$M, p_inter=$P"
        python evaluation/train.py --model "$M" --p_inter "$P" --epochs 20 > "$OUTDIR/${M}_p${P}.txt"
    done
done

echo "Running Dense baseline with 20 epochs..."
python evaluation/train.py --model dense --epochs 20 > "$OUTDIR/dense.txt"

echo
echo "Done! All logs saved in $OUTDIR"
