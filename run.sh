#!bin/bash
DATASET_PATH=$1
echo $DATASET_PATH

python ./src/semantic.py --img_path "$DATASET_PATH"/images/0.jpg --out_path "$DATASET_PATH"/masks/0.png
cd XMem2
python process_video.py --video "$DATASET_PATH"/images --masks "$DATASET_PATH"/masks --output "$DATASET_PATH"/masks