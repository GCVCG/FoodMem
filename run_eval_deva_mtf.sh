#!/bin/bash

for i in {1..14}; do
  DATASET_PATH=/vgdata/MTF_Challenge/MTF_Challenge/$i
  rm -rf /vgdata/mtf/deva/$i/error_masks
  python3 -u ./src/eval_map.py --submit_dir "$DATASET_PATH/masks_deva/Annotations" --truth_dir "/vgdata/MTF_Challenge-20240418T205428Z-001/MTF_Challenge/$i/Mask" --output_dir "/vgdata/mtf/deva/$i" --show_error
done