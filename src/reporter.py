
import sys
import logging as log

import matplotlib.pyplot as plt # todo: for use in `visualize`
import torch # todo: for use in `load_data`

from load import CustomDataset

class Reporter():
  def __init__(self, name = None, level = None):

    self.log_levels = {
      'DEBUG'  : log.DEBUG,
      'INFO'   : log.INFO,
      'WARNING': log.WARNING,
      'ERROR'  : log.ERROR,
    }

    # configure our logger object at instance creation
    if name is not None and level is not None:
      assert level in self.log_levels

      self.logger    = log.getLogger(f'{name}')
      self.formatter = log.Formatter('%(name)s - %(levelname)s - %(message)s')
      self.handler   = log.StreamHandler()

      self.handler.setFormatter(self.formatter)
      self.logger.addHandler(self.handler)
      self.logger.setLevel(self.log_levels[level])
    else:
      self.logger    = None
      self.formatter = None
      self.handler   = None

  def configure(self, name, level):
    '''
      configure Logger instance settings
    '''
    assert level in self.log_levels

    self.logger    = log.getLogger(f'{name}')
    self.formatter = log.Formatter('%(name)s - %(levelname)s - %(message)s')
    self.handler   = log.StreamHandler()

    self.handler.setFormatter(self.formatter)
    self.logger.addHandler(self.handler)
    self.logger.setLevel(self.log_levels[level])



  def debug(self, message):
    '''
      report a message of severity 'DEBUG'
    '''
    self.logger.debug(message)

  def info(self, message):
    '''
      report a message of severity 'INFO'
    '''
    self.logger.info(message)

  def warning(self, message):
    '''
      report a message of severity 'WARNING'
    '''
    self.logger.warning(message)

  def error(self, message):
    '''
      report a message of severity 'ERROR', exits program
    '''
    self.logger.error(message)
    sys.exit(1)



  def visualize(self, data, dest = None):
    '''
      plot our data out to a terminal or optional path
    '''
    raise NotImplementedError



  def load_data():
    '''
      load in our training, validation, or testing datasets
      returns our pytorch *_loader
    '''
    raise NotImplementedError

  def save_data():
    '''
      save our predicted labels for Kaggle submission
    '''
    raise NotImplementedError

