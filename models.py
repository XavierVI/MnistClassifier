import torch.nn as nn

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(169, 169, bias=True),
            nn.ReLU(),
            nn.Linear(169, 10, bias=True)
        )
  
    def forward(self, x):
        prediction = self.conv_layers(x)
        return prediction
  
    def __str__(self):
        return 'ConvModel'
  
class LinearModel(nn.Module):
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
    
    def __str__(self):
        return 'LinearModel'