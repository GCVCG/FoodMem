#!/bin/bash

for i in {1..14}; do
  DATASET_PATH=/vgdata/MTF_Challenge/MTF_Challenge/$i
  python3 -u ./src/eval_map.py --submit_dir "$DATASET_PATH/masks_3/masks" --truth_dir "$DATASET_PATH/masks" --output_dir "/vgdata/mtf/foodmem_3/$i" --show_error
done
#
for i in {1..14}; do
  DATASET_PATH=/vgdata/MTF_Challenge/MTF_Challenge/$i
  python3 -u ./src/eval_map.py --submit_dir "$DATASET_PATH/masks_6/masks" --truth_dir "$DATASET_PATH/masks" --output_dir "/vgdata/mtf/foodmem_6/$i" --show_error
done

for i in {1..14}; do
  DATASET_PATH=/vgdata/MTF_Challenge/MTF_Challenge/$i
  python3 -u ./src/eval_map.py --submit_dir "$DATASET_PATH/masks_9/masks" --truth_dir "$DATASET_PATH/masks" --output_dir "/vgdata/mtf/foodmem_9/$i" --show_error
done