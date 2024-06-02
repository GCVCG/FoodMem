# FoodMEM: A Fast and Precise Food Video Segmentation
Automatic food segmentation in near real-time
Checkpoint must be added to the root dir (It must look like FoodMEM/ckpts/SETR_MLA/iter_80000.pth): https://drive.google.com/drive/folders/1Bxwj8FDGIdOnEnscjLwB7sisHlMHdo7H?usp=drive_link
Saves must be added to the XMem2 dir (It must look like FoodMEM/XMem2/saves): https://drive.google.com/drive/folders/1pLiy-hyjzscLjmysexPDmp5DW3QJv0t4?usp=drive_link

# Installation
````bash
conda create -n FoodMEM python=3.8

conda activate FoodMEM   

conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge

pip install -U openmim

mim install mmcv-full==1.7.1

pip install -r requirements.txt

cd XMem2

pip install -r requirements.txt

bash scripts/download_models.sh

bash scripts/download_models_demo.sh
````