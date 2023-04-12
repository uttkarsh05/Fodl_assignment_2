import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os 
import wandb
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from torchvision import models,datasets,transforms
import copy
import time
from types import SimpleNamespace


default_config = SimpleNamespace(
	wandb_project = 'sweeps',
	wandb_entity = 'uttakarsh05',
	epochs = 10,
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
	weight_initialization = 'He',
	input_size = 128,
	input_dim = 3,
	out_dim = 10 ,
    no_cnn = 4,
	filter_sizes = 3,
	feature_maps = 64,
	padding = 0,
	stride = 1,
	dropout = 0.2,
	filter_orientation = 'double',
	hidden_layers = 1 ,
	hidden_size = 64,
	activation = 'elu',
    batch_norm = 'True',
    patience = 2,
    min_delta = 0.005,
    data_dir = 'data',
    device = 'mps'
)

def parse_args():
	argparser = argparse.ArgumentParser(description = 'Processing Hyperparameters')
	argparser.add_argument('-wp','--wandb_project',type = str,default = default_config.wandb_project,help = 'wandb project name')
	argparser.add_argument('-we','--wandb_entity',type = str,default = default_config.wandb_entity,help = 'wandb username/entity name')
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
	argparser.add_argument('-cnn','--no_cnn', type = int , default = default_config.no_cnn,help = 'no of cnn layers')
	argparser.add_argument('-fs','--filter_sizes', type = int , default = default_config.filter_sizes,help = 'kernel sizes for every convolutional layer')
	argparser.add_argument('-fm','--feature_maps', type = int , default = default_config.feature_maps,help = 'starting no of feature maps in a convoutional layer')
	argparser.add_argument('-fo','--filter_orientation',type = str , default = default_config.filter_orientation , help = 'how no of feature maps increase or decrease')
	argparser.add_argument('-ins','--input_size',type = int, default = default_config.input_size , help = 'resolution of input image')
	argparser.add_argument('-ind','--input_dim',type = int, default = default_config.input_dim , help = 'no of channels in the input image')
	argparser.add_argument('-outd','--out_dim',type = int, default = default_config.out_dim , help = 'no of classes for the classification task')
	argparser.add_argument('-str','--stride',type = int , default = default_config.stride, help = 'stride values for the kernel in the convolutional layers')
	argparser.add_argument('-pad','--padding',type = int , default = default_config.padding, help = 'padding values for the convolutional layers')
	argparser.add_argument('-hs','--hidden_size',type = int,default = default_config.hidden_size,help = 'list of number of hidden layers')
	argparser.add_argument('-a','--activation',type = str,default = default_config.activation,help = 'activation name')
	argparser.add_argument('-d','--dropout',type = float,default = default_config.dropout,help = 'probability of a neuron to be dropped')
	argparser.add_argument('-w_i','--weight_init',type = str,default = default_config.weight_initialization,help = 'activation name')
	argparser.add_argument('-b_n','--batch_norm',type=str , default=default_config.batch_norm, help='batch normalization after every layer')
	argparser.add_argument('-hl','--hidden_layers',type=int , default=default_config.hidden_layers, help='no of hidden layers in feed forward layer')
	argparser.add_argument('-p','--patience',type=int , default=default_config.patience, help='size of early stopping window')
	argparser.add_argument('-m_d','--min_delta',type=int , default=default_config.min_delta, help='minimum value allowed from the min validation accuracy')
	argparser.add_argument('-d_dir','--data_dur',type=str , default=default_config.data_dir, help='path to the data folder')
	argparser.add_argument('-dev','--device',type=str , default=default_config.device, help='device used for training the model') 
	args = argparser.parse_args()
	vars(default_config).update(vars(args))
	return 

def preprocess_data(data_dir = 'data' , img_size = 128 ):
    
    # path to the data folder from the current directory
    data_dir = data_dir

    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.25, 0.25, 0.25])

    #size of image to be resized to
    img_size = img_size

    #transformation to be performed on training an test images
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size,img_size)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    # creating datasets dictionary from train and validation folders in data directory
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}

    # splitting train dataset into train and validation dataset with random split of size [0.8,0.2]
    train_size = int(0.8 * len(image_datasets['train']))
    test_size = len(image_datasets['train']) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(image_datasets['train'], [train_size, test_size])

    test_dataset =  image_datasets['val']

    # class names 
    class_names = image_datasets['train'].classes
    
    return train_dataset,val_dataset,test_dataset,class_names

class EarlyStopper:
    def __init__(self, patience=2, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_accuracy = 0

    def early_stop(self, validation_accuracy):
        if validation_accuracy > self.min_validation_accuracy:
            self.min_validation_accuracy = validation_accuracy
            self.counter = 0
        elif validation_accuracy < (self.min_validation_accuracy - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Model(nn.Module):
    def __init__(self,input_size,input_dim,out_dim,filter_sizes,feature_maps,activation,hidden_layers,hidden_size,padding = 0,stride = 1,max_pool = 'Half',dropout = 0,batch_norm = True):
        
        super().__init__()
        
        self.input_size = input_size
        
        self.input_dim = input_dim
        self.out_dim = out_dim
        
        self.filter_sizes = filter_sizes
        self.feature_maps = feature_maps
        
        self.activation =  self._get_activation(activation)
        self.max_pool = max_pool
        
        self.padding = padding
        self.stride = stride
        
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        
        self.droppout = dropout
        self.batch_norm = batch_norm
        
        self.flatten_layer = self._get_flatten_layer_size()
        
        self.network = self._create_network()
    
    def forward(self,X):
        return self.network(X)
    
    def predict(self,X):
        y_pred = self.network(X)
        y_hat = y_pred.argmax(1)
        return y_hat
    
    
        
        
    
        
        
    def _get_flatten_layer_size(self):
        
        N = self.input_size
        P = self.padding
        S = self.stride
        O = 0
        
        for i in range(len(self.filter_sizes)):
            
            F = self.filter_sizes[i]
            
            O = ((N-F+2*P)/(S)) + 1
            O = O//2
            
            N = O
            
        out_channels = self.feature_maps[-1]
        
        N = out_channels*N*N
        
        return int(N)
    
  

        
            
        
        
        
    def _get_activation(self,activation):
        if activation.lower() =='relu':
            g = nn.ReLU()
        
        if activation.lower() == 'tanh':
            g = nn.Tanh()
        
        return g
            
            

    
    def _create_network(self):
        
        
        network = []
        in_channels = self.input_dim
        pad = self.padding
        stride = self.stride
        
        for i in range(len(self.filter_sizes)):
            
            k = self.filter_sizes[i]
            out_channels = self.feature_maps[i]
            
            
            network.append(nn.Conv2d(in_channels,out_channels,kernel_size = k , padding = pad , stride = stride))
        
            network.append(self.activation)
            if self.batch_norm:
                network.append(nn.BatchNorm2d(num_features=out_channels))
            
            network.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
            
            in_channels = out_channels
        
        network.append(nn.Flatten())
        
        in_dim = self.flatten_layer
        if self.batch_norm:
            network.append(nn.BatchNorm1d(num_features=in_dim))
        
        for i in range(self.hidden_layers):
            
            hidden_size = int(self.hidden_size[i])
            
            network.append(nn.Linear(in_dim,hidden_size))
            network.append(self.activation)
            if self.batch_norm:
                network.append(nn.BatchNorm1d(num_features=hidden_size))
            
            
            in_dim = hidden_size
        
        network.append(nn.Linear(in_dim,self.out_dim))
        network.append(nn.LogSoftmax(dim=-1))
        
        network = nn.Sequential(*network)
        
        return network
            
        
        
        
        
        
def get_loss(loss):
    if loss.lower() =='crossentropy':
        loss_function = nn.CrossEntropyLoss().to(device)
    elif loss.lower()=='mse':
        loss_function = nn.MSELoss()
    
    return loss_function
        
def get_optimizer(optimizer,lr,momentum,beta,beta1,beta2,network):
    optimizer = optimizer.lower()
    
    if optimizer=='sgd':
        opt = optim.SGD(network.parameters(),lr = lr )
    
    elif optimizer=='momentum':
        opt = optim.SGD(network.parameters(),lr = lr,momentum = momentum )
    
    elif optimizer=='nesterov':
        opt = optim.SGD(network.parameters(),lr = lr , momentum = beta)
    
    elif optimizer == 'adam':
        opt = optim.Adam(network.parameters(),lr = lr , betas = (beta1,beta2))
    
    elif optimizer == 'nadam':
        opt = optim.NAdam(network.parameters(),lr = lr , betas = (beta1,beta2))
    
    elif optimizer == 'rmsprop':
        opt = optim.RMSprop(network.parameters(),lr = lr )
    
    return opt

'''input_size  = default_config.input_size
    
input_dim = default_config.input_dim
out_dim  = default_config.out_dim

activation = default_config.activation

filter_sizes = default_config.filter_sizes
feature_maps = default_config.feature_maps

padding = default_config.padding
stride  = default_config.stride

dropout = default_config.dropout
max_pool = default_config.max_pool



hidden_layers = default_config.hidden_layers
hidden_size = default_config.hidden_size

model = Model(input_size,input_dim,out_dim,filter_sizes,feature_maps,activation,hidden_layers,hidden_size,padding,stride,max_pool ,dropout)

network = model.network
network = network.to(device)
network[0].weight.shape



network(X_val[:100].view(100,3,128,128)).argmax(1)

accuracy_score(val_labels[:100],network(X_val[:100].view(100,3,128,128)).argmax(1).cpu().numpy())'''

def train(config = default_config):
    
    run  = wandb.init(project = config.wandb_project , entity = config.wandb_entity,config = config )
    
    config = wandb.config
    
    input_size  = config.input_size
    
    input_dim = config.input_dim
    out_dim  = config.out_dim
    
    activation = config.activation
    
    filter_sizes = config.filter_sizes
    feature_maps = config.feature_maps
    
    padding = config.padding
    stride  = config.stride
    
    dropout = config.dropout
    max_pool = config.max_pool
    batch_norm = config.batch_norm
    
    
    hidden_layers = config.hidden_layers
    hidden_size = config.hidden_size
    
    
    model = Model(input_size,input_dim,out_dim,filter_sizes,feature_maps,activation,hidden_layers,hidden_size,padding,stride,max_pool ,dropout)
    
    network = model.network
    network = network.to(device)
    
    loss_function = get_loss(config.loss).to(device)
    
    lr = config.learning_rate
    momentum = config.momentum
    beta = config.beta
    beta1 = config.beta1
    beta2 = config.beta2
    
    optimizer = get_optimizer(config.optimizer,lr,momentum,beta,beta1,beta2,network)
    
    epochs = config.epochs
    
    batch_size = config.batch_size 
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for i in range(epochs):
        j = 0
        for (x,y) in train_dataloader:
            #print('iteration no = ',j+1)
            
            x = x.view(x.shape[0],input_dim,input_size,input_size)
            
            optimizer.zero_grad()
            
            y_hat = network.forward(x)
            
            loss = loss_function(y_hat,y)
            
            loss.backward()
            
            optimizer.step()
            
            j+=1
        
        val_accuracy = 0
        batches = int(X_val.shape[0]/100)
        for k in range(batches):
            val = X_val[k*100:k*100+100]
            y_pred = model.predict(val)
            acc = torch.sum(val_labels[k*100:k*100+100]==y_pred)/100
            val_accuracy += acc.cpu().numpy()
            
        val_accuracy = val_accuracy/batches
            
        
        
        
        print('epoch = ',i+1, ' training loss = ',loss.item(),' val accuracy = ', np.round(val_accuracy,4)*100)
        wandb.log({'epochs':i+1,'training_loss':loss.item(),'val_accuracy':np.round(val_accuracy,4)*100})


 
    
    
    



if __name__ == "__main__":
    parse_args()
    train(default_config)
