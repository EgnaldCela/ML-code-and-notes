# QUESTION: If you train with images 100x100 you must use the same size in the real testing world Ask ab project
# the output of each convolutional layer is a cube, where depth is the number of out channels

# How to load your costum dataset in pytorch?
# Imagine you have a txt file with image path and image class.
"""
Assume labels.csv, with each row being
im1.jpg, 0
img2.jpg, 0
img3.jpg, 5
.
.
.
imgXXX.jpg, 7
"""
# if you want to do object detection, the label can be the bounding box of the image
import os
import pandas as pd
from torchvision import read_image
from torch.utils.data import Dataset

# define our dataset

class MyDataset(Dataset):
    def __init__(self, labels, img_dir, transform=None):
        self.img_dir = img_dir
        self.transforms = transform
        self.labels = pd.read_csv(labels)
        # will be a table w 2 columns, path and label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # create the images paths
        # e.g. img_dir = 'data\img1.jpg'
        imgs_path = os.join(self.img_dir, self.labels.iloc[index,0])
        # accessing by indices must be done using iloc in pandas

        # read the image from the disk
        image = read_image(imgs_path)

        # read label of the image
        label = self.labels.iloc[index,1]

        # at this point you can do transformations on the data, if any
        if self.transforms:
            image = self.transforms(image)

        # return the prepared data together with its label
        return image, label
    
# In object detection, return image, label and bounding box
# If you are doing segmentation, you will have 2 images, image where to find object and segmentation mask


# Pytorch assumes you have datasets with certain structure. Your dataset folder should have these
# 3 folders inside, it will assign the class of the folder inside all images of that folder
# it should be called image folder. go to documentation PyTorch ImageFolder



# TOPIC 2: Autoencoders (a family of models)
# Encoder: data in lower dimensional
# Decoder: reconstructed data
# an embedding is a compressed representation of data

# Autoencoders are alternative to PCA for non linear data
# Autoencoderd are used for removing noise from images for example
# for example add some noise to your image, then at the end of autoencoder make the loss function be reconstructed image vs original
# image with no noise

# style transform: make this image picasso style

