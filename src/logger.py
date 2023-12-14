
import sys
import logging as log
import numpy as np

import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class Logger():
  def __init__(self, name = None, level = 'DEBUG'):

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