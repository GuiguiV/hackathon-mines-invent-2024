# Our approach

Our approach to this challenge is mainly discussed in the powerpoint presentation. However, here is a summary.
The kaggle notebook has been cloned into `main-unet.ipynb`. However, the code  uses some pretrained weights we uploaded on kaggle (too large for github).

## Model
We have chosen the UNet architecture to train our network from scratch. We have found an existing implementation at https://github.com/milesial/Pytorch-UNet

## Preprocessing
We use the NDVI index (see `baseline.ndvi`) to find images that are too cloudy and remove them from the dataset. We take the median of each time series to have a 3D input to the UNet. We rotate the images 90 degrees for data augmentation to reduce overfitting.

## Training
The submissions have been made using a model trained for 30 epochs. The training functions are available in `baseline.train`.


