import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# todo: refactor naming, technically a dataset class

class CustomDataset(Dataset):
  '''
    attributes:
      - data

      - targets

    methods:
      - __len__

      - __getitem__
  '''
  def __init__(self):
    self.data = None
    self.targets = None
    self.transforms = None

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    raise NotImplementedError
