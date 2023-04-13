from train import train as train_a
from train import default_config as config_a
from transfer_learning import default_config as config_b
from transfer_learning import train as train_b
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
import argparse

default_config = SimpleNamespace(
    data_dir = 'data',
    device = 'mps',
    part = 'a',
    )
def parse_args():
	argparser = argparse.ArgumentParser(description = 'Processing Hyperparameters')
	argparser.add_argument('-p','--part',type=str , default=default_config.part, help='part a or b to test the best model from there')
	argparser.add_argument('-d_dir','--data_dir',type=str , default=default_config.data_dir, help='path to the data folder')
	argparser.add_argument('-dev','--device',type=str , default=default_config.device, help='device used for training the model') 
	args = argparser.parse_args()
	vars(default_config).update(vars(args))
	return 

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

def test(config = default_config):
    
    device = config.device
    
    if config.part == 'a':
        model = train_a()
        model = model.to(device)
        img_size = config_a.input_size
    else:
        model = train_b()
        model = model.to(device)
        img_size = config_b.input_size
        
    test_dataset , class_names = preprocess(data_dir=config.data_dir,img_size=img_size)
    test_dataloader = DataLoader(test_dataset,batch_size=100)
    
    
    test_accuracy = 0
    model.eval()
    start = time.time()
    for x , test_labels in test_dataloader:
        x = x.to(device)
        test_labels = test_labels.to(device)
        
        outputs = model(x)
        _,preds = torch.max(outputs,1)
        test_accuracy += torch.sum(preds==test_labels)
        
    end = time.time()
    
    test_accuracy = test_accuracy/len(test_dataset)
    test_accuracy = test_accuracy.cpu().numpy()
    
    print('-------------------------------')
    print('test accuracy = ',np.round(test_accuracy,4))
    
    time_elapsed = start - end
    
    print('Prediction complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
if __name__ =='__main__':
    parse_args()
    test()
    
    
        
        
    
    
