#!/bin/bash

for i in {1..14}; do
  DATASET_PATH=/vgdata/MTF_Challenge/MTF_Challenge/$i
  python3 -u ./src/eval_map.py --submit_dir "$DATASET_PATH/masks_SeTR" --truth_dir "$DATASET_PATH/masks" --output_dir "/vgdata/mtf/SeTR/$i" --show_error
done