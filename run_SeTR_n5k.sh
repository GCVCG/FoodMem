#!/bin/bash

DATASET_NAMES=("dish_1550707110" "dish_1550707914" "dish_1550708487" "dish_1550710177" "dish_1550710577" "dish_1550711066" "dish_1550711688" "dish_1550712369" "dish_1550712681" "dish_1550713174" "dish_1551123072" "dish_1551141849" "dish_1551236981" "dish_1551314254" "dish_1551381404" "dish_1551392919" "dish_1557861837" "dish_1561575300" "dish_1561751958" "dish_1562096512" "dish_1563208094" "dish_1563304925" "dish_1563393366" "dish_1563553735" "dish_1563811515" "dish_1563986062" "dish_1564169298" "dish_1565195493" "dish_1566502061" "dish_1568146977" "dish_1574711517")

for i in ${DATASET_NAMES[@]}; do
  DATASET_PATH=/vgdata/MTF_Challenge/n5k/$i
  find "$DATASET_PATH/imgs" -mindepth 1 -maxdepth 1 -type f | parallel -I? --max-args 1 --jobs 1 --linebuffer python3 -u ./src/semantic.py --img_path ? --out_path "$DATASET_PATH/masks_SeTR_n5k";
done