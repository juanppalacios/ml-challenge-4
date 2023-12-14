
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def visualize(dataloader, n_images = None):
  '''
    plot our data out to a terminal or optional path
  '''

  images, labels = next(iter(dataloader))

  #> only show a single dataset image
  if n_images == 1:
    image = transforms.ToPILImage()(images[0])
    plt.imshow(image)
    plt.title(f"class: {labels[0]}")
    plt.show()
  else:
    images = torchvision.utils.make_grid(images)
    images = images / 2 + 0.5
    images = images.numpy()
    plt.imshow(np.transpose(images, (1, 2, 0)))
    plt.show()

def download_dataset(dataset_name = 'CIFAR10', batch_size = 32, transform = None):
  '''
    downloads and returns our PyTorch's dataset and data loader
  '''

  available_datasets = {
    'CIFAR10' : datasets.CIFAR10,
  }

  assert dataset_name in available_datasets, "selected database NOT in available databases"

  #> read in our training data set
  train_dataset = available_datasets[dataset_name](root = f'../data/{dataset_name}', train=True, download=True, transform=transform)
  train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

  #> read in our testing data set
  test_dataset = available_datasets[dataset_name](root = f'../data/{dataset_name}', train=False, download=True, transform=transform)
  test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

  return train_dataset, train_loader, test_dataset, test_loader
