#!/bin/bash

DATASET_NAMES=("apple/5.4" "avocado/5.1" "banana/5.2" "blackberry/5.2" "blueberry/5.2" "carrot/5.2" "cucumber/5.2" "grapes/5.2" "peach/5.2" "pear/5.2" "strawberry/5.2")

for i in ${DATASET_NAMES[@]}; do
  DATASET_PATH=/vgdata/MTF_Challenge/vnf/$i
  python3 -u ./src/eval_map.py --submit_dir "$DATASET_PATH/masks_SeTR_vnf" --truth_dir "$DATASET_PATH/gt_masks" --output_dir "/vgdata/vnf/foodmem/$i" --show_error
done