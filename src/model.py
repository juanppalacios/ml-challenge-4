
import torch
from reporter import Reporter

# note: MacOS does NOT support CUDA, check MPS availability instead
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Running PyTorch version {torch.__version__} on a {device} device")

class Model(torch.nn.Module):
  def __init__(self):
    super(Model, self).__init__()

    self.reporter = Reporter(name = 'PyTorch Model', level = 'DEBUG')
    self.reporter.info("creating our PyTorch model")

  def forward(self, x):
    '''
      perform forward propagation
    '''
    raise NotImplementedError

  def configure(self):
    '''
      store our model's tunable hyperparameters
    '''
    raise NotImplementedError

  def train(self):
    raise NotImplementedError

  def test(self):
    raise NotImplementedError

  def predict(self):
    raise NotImplementedError


  def __repr__(self):
    '''
      displays out our current model architecture
    '''
    raise NotImplementedError
