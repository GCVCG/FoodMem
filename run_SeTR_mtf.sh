#!/bin/bash

for i in {1..14}; do
  DATASET_PATH=/vgdata/MTF_Challenge/MTF_Challenge/$i
  find "$DATASET_PATH/images" -mindepth 1 -maxdepth 1 -type f | parallel -I? --max-args 1 --jobs 1 --linebuffer python3 -u ./src/semantic.py --img_path ? --out_path "$DATASET_PATH/masks_SeTR";
done