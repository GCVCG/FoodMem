#!/bin/bash

for i in {4..4}; do
  DATASET_PATH=/vgdata/MTF_Challenge/MTF_Challenge/$i
  python3 -u ./src/eval_map.py --submit_dir "$DATASET_PATH/masks" --truth_dir "/vgdata/MTF_Challenge-20240418T205428Z-001/MTF_Challenge/$i/Mask" --output_dir "/vgdata/mtf/foodmem/$i" --show_error
done