import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# custom imports
from logger import Logger

torch.manual_seed(1)

class Model(nn.Module):
  def __init__(self, name):
    super(Model, self).__init__()

    # internal logging default level is DEBUG
    self.logger = Logger(name)

    # model training and validation dataset
    self.train_dataset = None
    self.train_targets = None
    self.test_dataset  = None
    self.test_targets  = None

    # model layers, loss, and optimizer parameters
    self.layers = []
    self.classifier = nn.Linear(1, 1)

    self.criterion = None
    self.optimizer = None

    # model parameters
    if torch.cuda.is_available():
      self.device = torch.device("cuda")
    elif torch.backends.mps.is_available():
      self.device = torch.device("mps")
    else:
      self.device = torch.device("cpu")

    self.path = None

    self.logger.info(f"running PyTorch version {torch.__version__} on {self.device}")

  def forward(self, x):
    '''
      perform forward propagation
    '''
    for layer in self.layers:
      x = layer(x)
    return x

  def load_configuration(self, path = None):
    '''
      read in a model configuration from a specified path
    '''

    assert os.path.exists(path), f"configuration file path '{path}' does NOT exist"

    with open(path, 'r') as file:
      configuration = json.load(file)

    self.configure(
      layers=configuration['layers'],
      criterion=configuration['criterion'],
      optimizer=configuration['optimizer']
    )

  def build_layers(self, layers):
    '''
      create our layers with respective activations
    '''

    available_layers = {
      'convolution 2D': nn.Conv2d,
      'max pooling 2D': nn.MaxPool2d,
      'batch normalization': nn.BatchNorm2d,
      'fully connected': nn.Linear,
      'relu': nn.ReLU
    }

    for layer in layers:
      for key, value in layer.items():
        assert key in available_layers, f"'{key}' layer is NOT available for this model"
        layer_type = available_layers[key]
        self.layers.append(layer_type(*value))

    return nn.Sequential(*self.layers)

  def configure(self, layers, criterion, optimizer):
    '''
      store our model's tunable hyperparameters
    '''

    available_optimizers = {
      'SGD' : optim.SGD,
      'Adam': optim.Adam
    }

    available_criterion = {
      'MSE loss': nn.MSELoss,
      'cross entropy loss': nn.CrossEntropyLoss,
    }

    assert criterion in available_criterion, f"{criterion} NOT available for this model"
    assert optimizer['name'] in available_optimizers, f"{optimizer['name']} NOT available for this model"

    # self.layers    = self.build_layers(layers)
    self.criterion = criterion
    self.optimizer = available_optimizers[optimizer['name']](self.parameters(), lr = optimizer['learning rate'], momentum = optimizer['momentum'])

    self.logger.info(f"our criterion is {self.criterion}")
    self.logger.info(f"our optimizer is {self.optimizer}")
    self.logger.info(f"our layers are {self.layers}")

  def train(self, ):
    raise NotImplementedError

  def test(self):
    raise NotImplementedError

  def predict(self):
    raise NotImplementedError

  def __repr__(self):
    '''
      displays out our current model layers
    '''
    return f"instance layers\n\t{self.layers}"


#> example usage
model_test = Model('CIFAR10 Model')
model_test.load_configuration('./config.json')

#> load our training and testing data
