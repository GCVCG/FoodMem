#!bin/bash
DATASET_PATH=$1
echo $DATASET_PATH

if [ -d "$DATASET_PATH/masks/" ]; then
    echo "Directory $directory_path already exists. Exiting."
    exit 1
fi

python3 -u ./src/semantic.py --img_path "$DATASET_PATH"/images/001.jpg --out_path "$DATASET_PATH"/masks/
# colors are assigned from semantic segmentation by default. Since we support a single object per scene, we need to
# convert the mask into black and white
convert "$DATASET_PATH"/masks/001.png -threshold 1% "$DATASET_PATH"/masks/001.png
# track them
cd XMem2
python3 -u process_video.py --video "$DATASET_PATH"/images --masks "$DATASET_PATH"/masks --output "$DATASET_PATH"
# changing the permission
chmod -R 777 "$DATASET_PATH"/masks