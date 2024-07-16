# FoodMem: Near Real-time and Precise Food Video Segmentation

---

## Abstract
Food segmentation, including in videos, is vital for addressing real-world health, agriculture, and food biotechnology issues. Current limitations lead to inaccurate nutritional analysis, inefficient crop management, and suboptimal food processing, impacting food security and public health. Improving segmentation techniques can enhance dietary assessments, agricultural productivity, and the food production process. This study introduces the development of a robust framework for high-quality, near-real-time segmentation and tracking of food items in videos, using minimal hardware resources. We present FoodMem, a novel framework designed to segment food items from video sequences of 360-degree unbounded scenes. FoodMem can consistently generate masks of food portions in a video sequence, overcoming the limitations of existing semantic segmentation models, such as flickering and prohibitive inference speeds in video processing contexts. To address these issues, FoodMem leverages a two-phase solution: a transformer segmentation phase to create initial segmentation masks and a memory-based tracking phase to monitor food masks in complex scenes. Our framework outperforms current state-of-the-art food segmentation models, yielding superior performance across various conditions, such as camera angles, lighting, reflections, scene complexity, and food diversity. This results in reduced segmentation noise, elimination of artifacts, and completion of missing segments. Here, we also introduce a new annotated food dataset encompassing challenging scenarios absent in previous benchmarks. Extensive experiments conducted on Nutrition5k and Vegetables & Fruits datasets demonstrate that FoodMem enhances the state-of-the-art by 2.5% mean average precision in food video segmentation and is 58 x faster on average.

![FoodMem architecture](assets/FoodMemModel.png)

We used a single image input for simplicity. Our two-stage framework (a) shows the SETR framework, where it accepts an image and generates a mask, followed by (b) XMem2, which accepts the mask and a set of images as a given input and produces masks for all frames.

## Checkpoints
Checkpoint must be added to the root dir (It must look like FoodMem/ckpts/SETR_MLA/iter_80000.pth): https://drive.google.com/drive/folders/1Bxwj8FDGIdOnEnscjLwB7sisHlMHdo7H?usp=drive_link
Saves must be added to the XMem2 dir (It must look like FoodMem/XMem2/saves): https://drive.google.com/drive/folders/1pLiy-hyjzscLjmysexPDmp5DW3QJv0t4?usp=drive_link

## Installation
````bash
conda create -n FoodMem python=3.8

conda activate FoodMem   

conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge

pip install -U openmim

mim install mmcv-full==1.7.1

pip install -r requirements.txt

cd XMem2

pip install -r requirements.txt

bash scripts/download_models.sh

bash scripts/download_models_demo.sh
````

## Getting started
````bash
python XMem2/process_video.py --video <path/to/folder> --masks <path/to/folder> --output <path/to/folder>
````

## Evaluation
````bash
python .\src\eval_map.py --submit_dir <path/to/folder> --truth_dir <path/to/folder> --output <path/to/folder>
````

## Quantitative results

### Average execution times of the different models

| **Dataset**       | **Frames range** | **FoodSAM** | **DEVA** | **kMean++** | **Ours**       |
|-------------------|------------------|-------------|----------|-------------|----------------|
| **Nutrition5k**   | 19-65            | 00:12:34    | 00:00:40 | 00:01:07    | **00:00:25**   |
| **V&F**           | 172-232          | 00:44:20    | 00:02:04 | 00:05:11    | **00:00:31**   |

*The models include FoodSAM, DEVA, kMean++, and our framework. The inference time was recorded in the format of hours:minutes:seconds.*

### Mean Average Precision (mAP)

| **Dataset**       | **FoodSAM** | **DEVA** | **kMean++** | **Ours**       |
|-------------------|-------------|----------|-------------|----------------|
| **Nutrition5k**   | **0.9192**  | 0.8825   | 0.4232      | 0.9098         |
| **V&F**           | 0.8914      | 0.8548   | 0.4361      | **0.9499**     |

*Comparison of mean average precision scores achieved by different models on two datasets: Nutrition5k and V&F. The models evaluated include FoodSAM, DEVA, kMean++, and our framework.*

### Comparison of Recall Scores

| **Dataset**       | **FoodSAM** | **DEVA** | **kMean++** | **Ours**       |
|-------------------|-------------|----------|-------------|----------------|
| **Nutrition5k**   | **0.7752**  | 0.7301   | 0.6467      | 0.7708         |
| **V&F**           | 0.9441      | 0.9328   | 0.9245      | **0.9469**     |

*Comparison of recall scores achieved by different models on two datasets: Nutrition5k and V&F. The models evaluated include FoodSAM, DEVA, kMean++, and our framework.*

NOTE: FoodSAM performs better than our framework in the Nutrition5k dataset. This is because FoodSAM was trained on datasets where the camera followed a predefined path to capture images, similar to the setup in the Nutrition5k dataset. On the other hand, our framework performs better in the Vegetables & Fruits dataset, where the camera has freedom of movement, resulting in less predictable image capture scenarios.

## Qualitative results

### FoodSAM and FoodMem

![FoodSAMFoodMem](assets/FoodMemFoodSAM.png)

### DEVA and FoodMem

![DEVAFoodMem](assets/DEVACombined.png)

### KMean++ and FoodMem

![KMeanCombined](assets/KMeanCombined.png)

## Acknowledgements

A large part of the code is borrowed from the following projects:

1. [FoodSAM](https://github.com/jamesjg/FoodSAM/)
2. [SETR](https://github.com/fudan-zvg/SETR)
3. [XMem++](https://github.com/mbzuai-metaverse/XMem2)

Also mention the following works that helped us to understand and develop our framework:

1. [Vision Transformer (ViT)](https://github.com/google-research/vision_transformer)
2. [Segment Anything](https://github.com/facebookresearch/segment-anything)
3. [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA)
4. [XMem](https://github.com/hkchengrex/XMem)
5. [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k)
6. [Vegetables & Fruits](https://www.sciencedirect.com/science/article/pii/S2405844023019291)
7. [LabelMe](https://github.com/labelmeai/labelme)
8. [Imagededup](https://github.com/idealo/imagededup)
