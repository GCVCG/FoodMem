#!/bin/bash


DATASET_PATH=/vgdata/foodseg103
find "$DATASET_PATH/images" -mindepth 1 -maxdepth 1 -type f | parallel -I? --max-args 1 --jobs 1 --linebuffer python3 -u ./src/semantic.py --img_path ? --out_path "$DATASET_PATH/masks_SeTR";
