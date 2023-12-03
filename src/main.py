# testing brevitas neural network architectures

import torch
import brevitas

from brevitas.quant_tensor import QuantTensor
from brevitas.core.scaling import ConstScaling


def debug(msg):
  print(f"debug - {msg}")

def main():
  torch_tensor = torch.randn(5)

  debug(f"before applying brevitas torch tensor function to {torch_tensor}\n")


  debug(f"after applying brevitas torch tensor function to {torch_tensor}\n")


if __name__ == "__main__":
  main()