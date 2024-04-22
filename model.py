import torch
import torch.nn as nn
import torchvision

"""
Notes: pixel size is 28x28 = 784
"""
class MnistClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    pixel_size = 784
    self.L1 = nn.Linear(pixel_size, 128, bias=False)
    self.L2 = nn.Linear(128, 32, bias=False)
    self.L3 = nn.Linear(32, 1, bias=False)