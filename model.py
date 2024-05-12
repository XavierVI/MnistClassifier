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
    # transforms each 2D (28x28) image into an array of 784 pixel values
    self.flatten = nn.Flatten()
    # nn.Sequential is an ordered container of modules
    self.linear_stack = nn.Sequential(
      nn.Linear(pixel_size, 128, bias=False),
      nn.Linear(128, 32, bias=False),
      nn.Linear(32, 10, bias=False)
    )
  
  def forward(self, x):
    x = self.flatten(x) # flatten inputs
    prediction = self.linear_stack(x) # make prediction
    return prediction
  
class NonLinearClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    pixel_size = 784
    # transforms each 2D (28x28) image into an array of 784 pixel values
    self.flatten = nn.Flatten()
    # nn.Sequential is an ordered container of modules
    self.linear_stack = nn.Sequential(
      nn.Linear(pixel_size, 128, bias=False),
      nn.ReLU(),
      nn.Linear(128, 32, bias=False),
      nn.ReLU(),
      nn.Linear(32, 10, bias=False)
    )
  
  def forward(self, x):
    x = self.flatten(x) # flatten inputs
    prediction = self.linear_stack(x) # make prediction
    return prediction
    