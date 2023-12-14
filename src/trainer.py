'''
  A training class to fit our training data to any model

  source: https://xylambda.github.io/blog/python/pytorch/machine-learning/2021/01/04/pytorch_trainer.html
'''
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

from logger import Logger

class Trainer:
  def __init__(self, model, criterion, optimizer, device, config_file = None) -> None:
    
    self.logger = Logger("trainer")
    
    assert model is not None, f"model cannot be empty!"

    self.model = model
    self.criterion = None
    self.optimizer = None
    self.device = None
    
    if config_file is not None:
      self._load_cofigure(config_file)
    else:
      self._configure(criterion, optimizer, device)
    
    self.logger.info(f"training model on a '{self.device}' device")


  def _configure(self, criterion, optimizer, device):
    '''
      Verify our user-selected criterion and optimizer are available
    '''

    available_criterion = {
      'MSE loss': nn.MSELoss,
      'cross entropy loss': nn.CrossEntropyLoss,
    }

    available_optimizers = {
      'SGD' : optim.SGD,
      'Adam': optim.Adam
    }

    available_devices = {
      'cuda' : torch.device('cuda'),
      'mps'  : torch.device('mps'),
      'cpu'  : torch.device('cpu')
    }

    self.logger.debug(f"our criterion is {criterion}")

    assert criterion in available_criterion, f"{criterion} NOT an available criterion"
    assert optimizer in available_optimizers, f"{optimizer} NOT an available optimizer"
    assert device in available_devices, f"{device} device NOT available to train on"

    self.criterion = available_criterion[criterion]
    self.optimizer = available_optimizers[optimizer['name']](self.parameters(), lr = optimizer['learning rate'], momentum = optimizer['momentum'])

    if torch.cuda.is_available():
      self.device = available_devices['cuda']
    elif torch.backends.mps.is_available():
      self.device = available_devices['mps']
    else:
      self.device = available_devices['cpu']


  def _train(self, train_loader):
    ...
    
  def _test(self, test_loader):
    ...
    
  def _compute_loss(self, predicted, target):
    ...


  def _load_configuration(self, path = None):
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

  def fit(self, train_loader, test_loader, epochs):
    '''
      fits our train and test loaders to run for `epochs` epochs
    '''

    for epoch in epochs:
      self._train(train_loader)
      
      self._test(test_loader)

      
#> example use
model = 1
debug_trainer = Trainer(
  model = model,
  criterion='cross entropy loss',
  optimizer='SGD',
  device='mps'
)

# note: alternate use with a json config file
debug_trainer = Trainer(model, config_path = './config.json')

train_loader = None
test_loader = None
epochs = 20

debug_trainer.fit(train_loader, test_loader, epochs)

debug_trainer.validate()

debug_trainer.