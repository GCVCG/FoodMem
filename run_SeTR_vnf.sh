#!/bin/bash

DATASET_NAMES=("apple/5.4" "avocado/5.1" "banana/5.2" "blackberry/5.2" "blueberry/5.2" "carrot/5.2" "cucumber/5.2" "grapes/5.2" "peach/5.2" "pear/5.2" "strawberry/5.2")

for i in ${DATASET_NAMES[@]}; do
  DATASET_PATH=/vgdata/MTF_Challenge/vnf/$i
  find "$DATASET_PATH/imgs" -mindepth 1 -maxdepth 1 -type f | parallel -I? --max-args 1 --jobs 1 --linebuffer python3 -u ./src/semantic.py --img_path ? --out_path "$DATASET_PATH/masks_SeTR_vnf";
done