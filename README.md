# Mines 2024 Data Challenge
This repo is to be used in combination with a kaggle notebook (copied in `./main-unet.ipynb` with the trained weights not available).

Our approach is to use a UNet trained with some data augmentation and with some preprocessing using colorimetric data. More details are given in `Data Challenge Final Presentation.pdf`

The original presentation of the hackathon is available in the `demo.ipynb`  notebook

## Model
We have chosen the UNet architecture to train our network from scratch. We have found an existing implementation at https://github.com/milesial/Pytorch-UNet

## Preprocessing
We use the NDVI index (see `baseline/ndvi.py` and `data-cleaning-v2.ipynb`) to find images that are too cloudy and remove them from the dataset. We take the median of each time series to have a 3D input to the UNet. We rotate the images 90 degrees for data augmentation to reduce overfitting.

## Training
The submissions have been made using a model trained for 30 epochs. The training functions are available in `baseline/train.py`.
