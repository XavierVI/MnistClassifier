import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from models import *
from Trainer import Trainer

import os
import sys

batch_size = 64
epochs = 10
momentum = 0.9
learning_rate = 0.001
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f'Using {device} device')
print(f'Device Name: {torch.cuda.get_device_name()}')

model_name = sys.argv[1]

# Defining the model
if model_name == 'ConvModel':
    model = ConvModel()
elif model_name == 'PyTorchExample':
    model = PytorchExample()
else:
    model = LinearModel()

model = model.to(device)

# Loading the dataset
training_data = datasets.MNIST(
    root="datasets",
    train=True,
    download=True,
    transform=ToTensor()
)
testing_data = datasets.MNIST(
    root="datasets",
    train=False,
    download=True,
    transform=ToTensor()
)
training_loader = DataLoader(training_data, batch_size=64, shuffle=True)
testing_loader = DataLoader(testing_data, batch_size=64, shuffle=True)

# Training
optimizer = torch.optim.SGD(
    model.parameters(), 
    momentum=momentum,
    lr=learning_rate   
)
loss_fn = nn.CrossEntropyLoss()
trainer = Trainer()
epochs = 5

print(f"Training {model} for {epochs} epochs")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    trainer.train(training_loader, model, loss_fn, optimizer, device)
    trainer.test(testing_loader, model, loss_fn, device)

directory = 'models/'+model.__str__()

if not os.path.exists(directory): os.makedirs(directory)

torch.save(model.state_dict(), f=directory+"/model.pth")
print("Saved PyTorch Model State to model.pth")
print("Done!")