import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class CustomDataset(Dataset):
  def __init__(self):
    raise NotImplementedError

  def __len__(self):
    raise NotImplementedError

  def __getitem__(self, idx):
    raise NotImplementedError
