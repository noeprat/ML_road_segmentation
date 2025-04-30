# Road Segmentation Project - Group AAA

# Team members

Charles-Edouard Rouault

Tistou Luisiere

Noé Prat

# Code organisation

## Common scripts

`losses.py` defines the different loss functions

`baseline_tree.py` efines useful functions for postprocessing and baseline models

`road_segmentation_charled.py` defines useful functions for preprocessing and data extraction

## CNN

`unet.py` shows the architecture

Run `unet.ipynb` to reproduce the results

## ViT-C

`vision_transformer.py` shows the architecture

Run `vision_transformer.ipynb` to reproduce the results

Weights of the best performing model are in the 'model_weights' folder

CSV files and TXT information are in the 'submissions' folder with the format `vit_c_e{epochs}_b{batch_size}_{loss}` (dice loss by default). CSV is the submitted file, and TXT contains the resulting AIcrowd scores

Plots of loss history and examples are in the 'ML_plots' folder, following the same format

## Ethical risks

Run `Ethics_aug.ipynb` to reproduce results

Dirt track images are in the 'Images Antsirabe' folder

# External resources

## Code 

Transformer Block in `vision_transformer.py` from [https://keras.io/examples/nlp/text_classification_with_transformer/]

## Model

Spectrewolf8-U-Net, from [https://huggingface.co/spectrewolf8/aerial-image-road-segmentation-with-U-NET-xp]

## Data

Massachusetts Road Dataset, available on [https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset/data]

## Libraries

>See `requirements.txt`

numpy

matplotlib

pandas

tqdm

tensorflow

keras

scikit-learn

scikit-image

PIL

pyyaml

h5py
