# FoodSAMPlusPlus
Automatic food segmentation in near real time
Checkpoint must be added to the root dir (It must look like FoodSAMPlusPlus/ckpts/SETR_MLA/iter_80000.pth): https://drive.google.com/drive/folders/1Bxwj8FDGIdOnEnscjLwB7sisHlMHdo7H?usp=drive_link

# Installation
````bash
conda create -n FoodSAMPlusPlus python=3.8

conda activate FoodSAMPlusPlus   

conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge

pip install -U openmim

mim install mmcv-full==1.7.1

pip install -r requirements.txt

cd XMem2

pip install -r requirements.txt

bash scripts/download_models.sh

bash scripts/download_models_demo.sh
````