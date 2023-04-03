import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import argparse
import random
import imageio
import cv2
import numpy as np
import os 
from sklearn.model_selection import train_test_split

class SimpleNamespace:
    def __init__(self, /, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        items = (f"{k}={v!r}" for k, v in self.__dict__.items())
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        if isinstance(self, SimpleNamespace) and isinstance(other, SimpleNamespace):
           return self.__dict__ == other.__dict__
        return NotImplemented

default_config = SimpleNamespace(
	wandb_project = 'sweeps',
	wandb_entity = 'uttakarsh05',
	dataset = 'fashion_mnist',
	epochs = 8,
	batch_size = 16,
	loss = 'cross_entropy',
	optimizer = 'adam',
	learning_rate = 0.000704160852345564,
	momentum = 0.9,
	beta = 0.9,
	beta1 = 0.9,
	beta2 = 0.99,
	epsilon = 1e-10,
	weight_decay = 0,
	weight_initialization = 'He',
	num_layers = 1,
	hidden_size = 128,
	activation = 'relu',
	keep_prob = 0.9,
)

def parse_args():
	argparser = argparse.ArgumentParser(description = 'Processing Hyperparameters')
	argparser.add_argument('-wp','--wandb_project',type = str,default = default_config.wandb_project,help = 'wandb project name')
	argparser.add_argument('-we','--wandb_entity',type = str,default = default_config.wandb_entity,help = 'wandb username/entity name')
	argparser.add_argument('-d','--dataset',type = str,default = default_config.dataset,help = 'dataset name')
	argparser.add_argument('-e','--epochs',type = int,default = default_config.epochs,help = 'no of epochs')
	argparser.add_argument('-b','--batch_size',type = int,default = default_config.batch_size,help = 'batch size')
	argparser.add_argument('-l','--loss',type = str,default = default_config.loss,help = 'loss function name')
	argparser.add_argument('-o','--optimizer',type = str,default = default_config.optimizer,help = 'optimizer name')
	argparser.add_argument('-lr','--learning_rate',type = float,default = default_config.learning_rate,help = 'learning rate')
	argparser.add_argument('-m','--momentum',type = float,default = default_config.momentum,help = 'beta value used for momentum optimizer')
	argparser.add_argument('-beta','--beta',type = float,default = default_config.beta,help = 'beta value used for rmsprop')
	argparser.add_argument('-beta1','--beta1',type = float,default = default_config.beta1,help = 'beta1 used by adam and nadam')
	argparser.add_argument('-beta2','--beta2',type = float,default = default_config.beta2,help = 'beta2 used by adam and nadam')
	argparser.add_argument('-eps','--epsilon',type = float,default = default_config.epsilon,help = 'epsilon value used by optimizers')
	argparser.add_argument('-w_d','--weight_decay',type = float,default = default_config.weight_decay,help = 'weight decay (lamda) value for l2 regularization')
	argparser.add_argument('-nhl','--num_layers',type = int,default = default_config.num_layers,help = 'number of hidden layers')
	argparser.add_argument('-sz','--hidden_size',type = int,default = default_config.hidden_size,help = 'size of every hidden layer')
	argparser.add_argument('-a','--activation',type = str,default = default_config.activation,help = 'activation name')
	argparser.add_argument('-kp','--keep_prob',type = float,default = default_config.keep_prob,help = 'probability of a neuron to be dropped')
	argparser.add_argument('-w_i','--weight_init',type = str,default = default_config.weight_initialization,help = 'activation name')

	args = argparser.parse_args()
	vars(default_config).update(vars(args))
	return 

# Define the labels for the Simpsons characters we're detecting
class_names = {0:'Amphibia', 1:'Animalia', 2:'Arachnida',3: 'Aves',4: 'Fungi',5: 'Insecta', 6:'Mammalia', 7:'Mollusca', 8:'Plantae',9: 'Reptilia'}
num_classes = 10
img_size = 128
dir = 'data/train'

# Load training data
'''X_train = []
y_train = []
for label, name in class_names.items():
   list_images = os.listdir(dir+'/'+name)
   for image_name in list_images:
       image = imageio.imread(dir+'/'+name+'/'+image_name)
       if np.ndim(image) == 3:
          X_train.append(cv2.resize(image, (img_size,img_size)))
          y_train.append(label)
print(X_train[0].shape)
        
np.save('X_train',np.array(X_train))
np.save('y_train',np.array(y_train))'''

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=42)








if __name__ == "__main__":
	parse_args()
	#train_wandb(default_config)