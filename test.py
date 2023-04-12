from train import train
import torch
from torchsummary import summary
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,TensorDataset
from torchvision import models,datasets,transforms
import wandb
import os 
import numpy as np
from types import SimpleNamespace
import time

default_config = SimpleNamespace(
    data_dir = 'data',
    device = 'mps',
    
)

def preprocess(data_dir = 'data',img_size = 128):
    data_dir = data_dir

    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.25, 0.25, 0.25])

    #size of image to be resized to
    img_size = img_size

    #transformation to be performed on training an test images
    data_transforms = {
        'val': transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    # creating datasets dictionary from train and validation folders in data directory
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['val']}

    # splitting train dataset into train and validation dataset with random split of size [0.8,0.2]
    
    test_dataset =  image_datasets['val']

    # class names 
    class_names = image_datasets['val'].classes
    
    return test_dataset,class_names