import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=2,
            stride=1
        )
        self.dropout1 = nn.Dropout(0.25)
        self.linear1 = nn.Linear(5408, 128)
        self.linear2 = nn.Linear(128, 10)
  
    def forward(self, x):
        x = self.conv_layer(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        output = F.softmax(x, dim=1)
        return output
  
    def __str__(self):
        return 'ConvolutionalNetwork'
    

class LinearNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        pixel_size = 784
        # transforms each 2D (28x28) image into an array of 784 pixel values
        self.flatten = nn.Flatten()
        # nn.Sequential is an ordered container of modules
        self.linear_stack = nn.Sequential(
            nn.Linear(pixel_size, 128),
            nn.Linear(128, 32),
            nn.Linear(32, 10)
        )
  
    def forward(self, x):
        x = self.flatten(x) # flatten inputs
        prediction = self.linear_stack(x) # make prediction
        return prediction
    
    def __str__(self):
        return 'LinearNetwork'
    
    
