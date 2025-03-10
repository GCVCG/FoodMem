#!/bin/bash

for i in {1..14}; do
  DATASET_PATH=/vgdata/MTF_Challenge/MTF_Challenge/$i
  python3 -u ./src/eval_map.py --submit_dir "$DATASET_PATH/masks_only_foodsam" --truth_dir "/vgdata/MTF_Challenge-20240418T205428Z-001/MTF_Challenge/$i/Mask" --output_dir "/vgdata/mtf/foodsam/$i" --show_error
done