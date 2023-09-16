# iNaturalist Classifier

## Instructions :

* Clone the repository using  `git clone uttkarsh05/Fodl_assignment_2`

* Create a data folder in the same directory as repository

* Add the inaturalist train and val dataset in that data folder (note : there should be two folders in the data directory named train and val)
  download the dataset from here : https://storage.googleapis.com/wandb_datasets/nature_12K.zip

* To test the best custom model on the test dataset (for full training of a cnn) run on the terminal : ` python test.py --part a `

* To test the best fine tuned model on the test dataset (for fine tuning a large pretrain model on ImageNet) run on the terminal : ` python test.py --part b `

* To use the best_model.pth file to load the best model in your code base to test by using ``` model = model.load_state_dict(torch.load('best_model.pth')) ```

* `train.py` also support command line arguments which can help in training a custom cnn model architecture . To see what all arguments are available in train.py run on the terminal :  ` python train.py --help ` 

* `transfer_learning.py` also support command line arguments which can help in fine tuning a pretrained ImageNet model . To see what all arguments are available in train.py run on the terminal :  ` python transfer_learning.py --help `

## Files 

`train.py` 
 
 This file contains the following :
 
 * `default_config` 
    
    It contains the best set of hyperparameter for the cnn model trained from scratch.   
 
 * `parse_args`
    
    Helper function to support and read the command line arguments . Default values are set to the values in the default_config dictionary
 
 * `preprocess_data` 
 
    The `preprocess_data` function is a utility function that loads image data from a directory, resizes and normalizes the images, and splits the data into training, validation, and testing datasets. The function takes as input the path to the input data directory and the size to which images should be resized.

    To perform the loading and preprocessing, the function uses PyTorch's `datasets.ImageFolder` function from the `torchvision` module, which loads images into two datasets: one for training and one for testing. The function then uses `torchvision.transforms` to resize and normalize the images.

    After preprocessing, the function randomly splits the training dataset into training and validation subsets, and returns a tuple containing four elements: the preprocessed training, validation, and testing datasets, and a list of the class names.
    
 * `EarlyStopper`

    The EarlyStopper class is a utility class for performing early stopping during model training. It has the following methods:

    * `__init__(self, patience=2, min_delta=0.01)` : Initializes the `EarlyStopper` object with the specified parameters:
       * `patience`: The number of epochs to wait before stopping training if validation accuracy does not improve.
       * `min_delta`: The minimum change in validation accuracy required to be considered an improvement.
    * `early_stop(self, validation_accuracy)` : Returns True if training should be stopped based on the current validation accuracy. Otherwise, returns False.
       * `validation_accuracy`: The current validation accuracy.
       
 * `Model` 
 
    * The `Model` class is a PyTorch `nn.Module` that defines a convolutional neural network architecture. It takes in various hyperparameters during initialization, including the size and dimension of the input, output dimension, filter sizes, feature maps, activation function, number of hidden layers, hidden layer size, padding, stride, dropout rate, and batch normalization.

    * The `Model` class has two main methods: `forward()` and `predict()`. The `forward()` method is used to pass the input tensor through the network, and the output is returned. The `predict()` method uses the `forward()` method to make predictions on the input tensor and returns the predicted class label.

    * The `_get_flatten_layer_size()` method calculates the size of the flattened output from the convolutional layers, which is used to define the input size of the fully connected layers. 
    
    * The `_get_activation()` method takes in the activation function as a string and returns the corresponding PyTorch activation function. 
    
    * The `_create_network()` method creates the convolutional neural network architecture by stacking layers in the following order: convolutional layer, activation function, batch normalization layer (if specified), max pooling layer, and fully connected layer. Finally, the `Model` class returns the fully connected layer with a LogSoftmax activation function for classification.
 
 * `get_loss`   

    Helper function to get the PyTorch loss function based on the input string.
  
 * `get_optimizers` 
 
    Helper function to get the PyTorch optimizer function based on the input string.
    
 * `train`
 
    It is a training function that creates a neural network based on the given parameters and trains the network using the provided data. The function takes a configuration object as an argument and initializes the wandb API for the experiment. The `preprocess_data` function is called to load the data and prepare it for training. Afterward, the hyperparameters are used to initialize the network and set up the optimization process, then the training and validation loops are executed. During the training process, the loss, accuracy, and other metrics are logged to the wandb project, and the best model weights are saved. Finally, the function prints the training time and returns the best model .
 
`transfer_learning.py` 
   
   The given code contains several functions and a class to perform image classification using a pre-trained CNN model, fine-tuned on a custom dataset. Below is the summary of each function and class in the code:

 *  `default_config`: This is a simple namespace that contains the default hyperparameters for the project, such as the wandb project name, wandb entity name, model name, number of epochs, batch size, optimizer name, learning rate, momentum, etc.

 * `parse_args()`: This function takes command line arguments using the argparse library, updates the default_config namespace with the values passed as arguments, and returns the updated namespace.

 * `EarlyStopper`: This is a class that implements early stopping based on the validation accuracy of the model during training. It takes two parameters, patience and min_delta, and has a single method called early_stop that returns True if the validation accuracy has not improved for 'patience' epochs, where the improvement is defined as an increase in accuracy of at least 'min_delta'.

 * `get_loss()`: This function takes a string argument that specifies the name of the loss function to be used and returns the corresponding loss function from the nn module of the PyTorch library. The function currently supports two loss functions, CrossEntropyLoss and MSELoss.

 * `get_optimizer()`: This function takes several arguments that specify the optimizer to be used, along with its hyperparameters, and returns the corresponding optimizer from the optim module of the PyTorch library. The function currently supports five optimizers, SGD, Momentum, Nesterov, Adam, and Nadam.

 * `train()`: This function is the entry point of the program and contains the main logic for training and evaluating the model. It loads the dataset using the torchvision.datasets module, preprocesses the data using the torchvision.transforms module, creates the data loader using the DataLoader class from the torch.utils.data module, loads the pre-trained CNN model using the torchvision.models module, replaces the last fully connected layer with a new layer that has the number of output classes specified in the hyperparameters, and trains the model using the specified optimizer and loss function. It also logs the training and validation accuracy using the wandb library, and saves the best model based on the validation accuracy using the torch.save function. 
   
`test.py`
 
 This is the script which can be used to train the best model from either `transfer_learning.py` or `train.py` on test dataset
 
 The given code consists of three functions - `preprocess()`, `test()`, and `parse_args()`, and a few import statements.

   * `preprocess(data_dir='data', img_size=128)`:
   
     This function is used to preprocess the input data. It takes two parameters as input - `data_dir`, which is the path to the data folder, and img_size,      which is the size of the image to be resized. It returns the preprocessed test dataset and the class names of the images in the dataset.

   * `test(config=default_config)`:
   
      This function is used to test the accuracy of the trained model on the test dataset. It takes one parameter as input - `config`, which is the    configuration of the model to be tested. If the `part` parameter in the `config` object is set to 'a', the `train_a()` function will be used to load the model. Otherwise, the `train_b()` function will be used. The `preprocess()` function is called to preprocess the test dataset. The accuracy of the model on the test dataset is computed and printed to the console.

   * `parse_args()`:
    This function is used to parse the arguments passed to the script. It uses the `argparse` library to parse the command-line arguments. The `part`, `data_dir`, and `device` arguments can be passed to the script.

   * Import Statements:
    
     The `train` function is imported from the `train` file. The `default_config` object is imported from the `train` and `transfer_learning` modules. The torch, torchsummary, torchvision, wandb, os, and numpy modules are imported for various tasks in the script.
      
  
