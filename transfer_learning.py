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
    
def get_loss(loss):
    if loss.lower() =='crossentropy':
        loss_function = nn.CrossEntropyLoss()
    elif loss.lower()=='mse':
        loss_function = nn.MSELoss()
    
    return loss_function
        
def get_optimizer(optimizer,lr,momentum,beta,beta1,beta2,network,weight_decay):
    optimizer = optimizer.lower()
    
    if optimizer=='sgd':
        opt = optim.SGD(network.parameters(),lr = lr ,weight_decay=weight_decay)
    
    elif optimizer=='momentum':
        opt = optim.SGD(network.parameters(),lr = lr,momentum = momentum ,weight_decay=weight_decay)
    
    elif optimizer=='nesterov':
        opt = optim.SGD(network.parameters(),lr = lr , momentum = beta,weight_decay=weight_decay)
    
    elif optimizer == 'adam':
        opt = optim.Adam(network.parameters(),lr = lr , betas = (beta1,beta2),weight_decay=weight_decay)
    
    elif optimizer == 'nadam':
        opt = optim.NAdam(network.parameters(),lr = lr , betas = (beta1,beta2),weight_decay=weight_decay)
    
    elif optimizer == 'rmsprop':
        opt = optim.RMSprop(network.parameters(),lr = lr ,weight_decay=weight_decay)
    
    return opt


def preprocess_data(data_dir = 'data' , img_size = 224 ):
    
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

def train(config = default_config):
    
    #run  = wandb.init(project = config.wandb_project , entity = config.wandb_entity,config = config )
    
    #config = wandb.config
    
    device = torch.device(config.device)
    
    model = config.model
    network = get_model(model,weights = "DEFAULT")
    weights = get_model_weights(model)
    param = next(iter(weights))
    c = param.transforms()
    img_size = c.crop_size[0]
    data_dir = config.data_dir
    
    train_dataset , val_dataset , test_dataset , class_names = preprocess_data(img_size=img_size,data_dir=data_dir)
    
    val_loader = DataLoader(val_dataset,batch_size=100 )
    
    input_dim = config.input_dim
    out_dim  = config.out_dim
    
    #name = '_lr_'+str(config.learning_rate)+'_o_'+str(config.optimizer)+'_bs_'+str(config.batch_size)+'_m_'+str(model)
    #run.name = name
    
    for param in network.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = network.fc.in_features
    network.fc = nn.Linear(num_ftrs, out_dim)
    
    network = network.to(device)
    
    loss_function = get_loss(config.loss).to(device)
    
    lr = config.learning_rate
    momentum = config.momentum
    beta = config.beta
    beta1 = config.beta1
    beta2 = config.beta2
    
    weight_decay = config.weight_decay
    
    optimizer = get_optimizer(config.optimizer,lr,momentum,beta,beta1,beta2,network,weight_decay)
    
    epochs = config.epochs
    
    batch_size = config.batch_size 
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    best_model_wts = copy.deepcopy(network.state_dict())
    best_acc = 0.0
    
    early_stopper = EarlyStopper(patience=2,min_delta=0.01)
    start = time.time()
    for i in range(epochs):
        
        train_loss = 0
        train_accuracy = 0
        
        # training loop 
        network.train()
        for (x,labels) in train_dataloader:
            
            x = x.to(device)
            labels = labels.to(device)
            
            x = x.view(x.shape[0],input_dim,img_size,img_size)
            
            outputs = network(x)
            _ , preds = torch.max(outputs,1)
            loss = loss_function(outputs,labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*x.size(0)
            train_accuracy += torch.sum(preds==labels)
        
        # calculating training loss and training accuracy after every epoch 
        train_loss = train_loss/len(train_dataset)
        
        train_accuracy = train_accuracy/len(train_dataset)
        train_accuracy = train_accuracy.cpu().numpy()
        
        val_accuracy = 0
        val_loss = 0
        
        # validation loop
        network.eval() 
        for x_val,val_labels in val_loader:
            x_val = x_val.to(device)
            val_labels = val_labels.to(device)
            x_val = x_val.view(x_val.shape[0],input_dim,img_size,img_size)
            outputs = network(x_val)
            _ , preds = torch.max(outputs,1)
            loss = loss_function(outputs,val_labels)
            
            val_loss += loss.item()*x_val.size(0)
            val_accuracy+= torch.sum(preds == val_labels)
        
        
        # calculating validation loss and validation accuracy after every epoch     
        val_loss = val_loss/len(val_dataset)
        
        val_accuracy = val_accuracy/len(val_dataset)
        val_accuracy = val_accuracy.cpu().numpy()
        
        if val_accuracy>best_acc:
            best_acc = val_accuracy
            best_model_wts = copy.deepcopy(network.state_dict())

        print('epoch = ',i+1, ' training loss = ',np.round(train_loss,4),' training accuracy = ',np.round(train_accuracy,4)*100,' validation loss = ',np.round(val_loss,4),' val accuracy = ', np.round(val_accuracy,4)*100)
        
        #wandb.log({'epochs':i+1,'training_loss':np.round(train_loss,4),'training_accuracy':np.round(train_accuracy,4)*100,'validation_loss':np.round(val_loss,4),'val_accuracy':np.round(val_accuracy,4)*100})

        if early_stopper.early_stop(val_accuracy):
            break
            
    end = time.time()
    time_elapsed = start - end
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    #loading the best model weights 
    network.load_state_dict(best_model_wts)
    
    #returning the best model
    return network