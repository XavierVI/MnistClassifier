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
epochs = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f'Using {device} device')
print(f'Device Name: {torch.cuda.get_device_name()}')

network_name = sys.argv[1]

# Defining the network
if network_name == 'ConvolutionalNetwork':
    network = ConvolutionalNetwork()
else:
    network = LinearNetwork()

network = network.to(device)

trainer = Trainer(network)

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
training_loader = DataLoader(training_data,batch_size=batch_size,shuffle=True)
testing_loader = DataLoader(testing_data,batch_size=batch_size,shuffle=True)


print(f"Training {network} for {epochs} epochs")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    trainer.train(training_loader, device)
    trainer.test(testing_loader, device)

directory = 'models/'+network.__str__()

if not os.path.exists(directory): os.makedirs(directory)

torch.save(network.state_dict(), f=directory+f"/model.pth")
print("Saved PyTorch Model State to model.pth")

trainer.plot_training_losses(True, filename=f'{os.curdir}/figures/{network}.png')