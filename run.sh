#!bin/bash
DATASET_PATH=$1
echo $DATASET_PATH
# TODO process_video solo acepta jpg, jpeg no funciona. Encontrar donde se busca jpg y hacerlo mas generico
# TODO Hacer overlay opcional con args
# TODO PRIO: Juntar semantic y process_video en un solo .py, hacer que la mascara de semantic se envie directamente a process_video sin guardarlo en disco
# TODO Make this code as a webservice
# TODO Apply VPQ metric
python ./src/semantic.py --img_path "$DATASET_PATH"/images/0.jpg --out_path "$DATASET_PATH"/masks/0.png
cd XMem2
python process_video.py --video "$DATASET_PATH"/images --masks "$DATASET_PATH"/masks --output "$DATASET_PATH"/masks