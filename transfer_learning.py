import torch
import torchvision 
from torchvision import models,datasets,transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os 
import matplotlib.pyplot as plt
import wandb
import argparse
import copy
import time
import torchvision.models as models 
from torchvision.models import list_models , get_model , get_model_weightsfrom types import SimpleNamespace
from types import SimpleNamespace

default_config = SimpleNamespace(
	wandb_project = 'sweeps',
	wandb_entity = 'uttakarsh05',
    model = 'resnet18',
	epochs = 5,
	batch_size = 64,
	loss = 'crossentropy',
	optimizer = 'adam',
	learning_rate = 1e-3,
	momentum = 0.9,
	beta = 0.9,
	beta1 = 0.9,
	beta2 = 0.99,
	epsilon = 1e-10,
	weight_decay = 0,
	input_size = 224,
	input_dim = 3,
	out_dim = 10 ,
    data_dir = 'data',
    device = 'mps'
)

def parse_args():
	argparser = argparse.ArgumentParser(description = 'Processing Hyperparameters')
	argparser.add_argument('-wp','--wandb_project',type = str,default = default_config.wandb_project,help = 'wandb project name')
	argparser.add_argument('-we','--wandb_entity',type = str,default = default_config.wandb_entity,help = 'wandb username/entity name')
	argparser.add_argument('-m','--model',type = str,default = default_config.model,help = 'pretrained model to be fine tuned')
	argparser.add_argument('-e','--epochs',type = int,default = default_config.epochs,help = 'no of epochs')
	argparser.add_argument('-b','--batch_size',type = int,default = default_config.batch_size,help = 'batch size')
	argparser.add_argument('-l','--loss',type = str,default = default_config.loss,help = 'loss function name')
	argparser.add_argument('-o','--optimizer',type = str,default = default_config.optimizer,help = 'optimizer name')
	argparser.add_argument('-lr','--learning_rate',type = float,default = default_config.learning_rate,help = 'learning rate')
	argparser.add_argument('-mom','--momentum',type = float,default = default_config.momentum,help = 'beta value used for momentum optimizer')
	argparser.add_argument('-beta','--beta',type = float,default = default_config.beta,help = 'beta value used for rmsprop')
	argparser.add_argument('-beta1','--beta1',type = float,default = default_config.beta1,help = 'beta1 used by adam and nadam')
	argparser.add_argument('-beta2','--beta2',type = float,default = default_config.beta2,help = 'beta2 used by adam and nadam')
	argparser.add_argument('-eps','--epsilon',type = float,default = default_config.epsilon,help = 'epsilon value used by optimizers')
	argparser.add_argument('-w_d','--weight_decay',type = float,default = default_config.weight_decay,help = 'weight decay (lamda) value for l2 regularization')
	argparser.add_argument('-ins','--input_size',type = int, default = default_config.input_size , help = 'resolution of input image')
	argparser.add_argument('-ind','--input_dim',type = int, default = default_config.input_dim , help = 'no of channels in the input image')
	argparser.add_argument('-outd','--out_dim',type = int, default = default_config.out_dim , help = 'no of classes for the classification task')
	argparser.add_argument('-d_dir','--data_dur',type=str , default=default_config.data_dir, help='path to the data folder')
	argparser.add_argument('-dev','--device',type=str , default=default_config.device, help='device used for training the model') 
	args = argparser.parse_args()
	vars(default_config).update(vars(args))
	return 
