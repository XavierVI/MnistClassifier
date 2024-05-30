import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from models import *
import Trainer

import os

print('PyTorch version: {torch.__version__}')
print('*'*40)
print('CUDA version: ')
print(torch.version.cuda)
print('*'*40)
print(f'CUDNN version: {torch.backends.cudnn.version()}')
print(f'Available GPU devices: {torch.cuda.device_count()}')
device = "cuda" if torch.cuda.is_available() else "cpu"
print('Using {device} device')
print(f'Device Name: {torch.cuda.get_device_name()}')

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

# Defining the model
model = ConvModel().to(device)

# Training
optimizer = torch.optim.SGD(model.parameters(), weight_decay=0.01, 
                            momentum=0.9, lr=0.1)
loss_fn = nn.CrossEntropyLoss()
trainer = Trainer()

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    trainer.train(training_loader, model, loss_fn, optimizer, device)
    trainer.test(testing_loader, model, loss_fn, device)

directory = 'models/'+model.__str__()

if not os.path.exists(directory): os.makedirs(directory)

torch.save(model.state_dict(), f=directory+"/model.pth")
print("Saved PyTorch Model State to model.pth")
print("Done!")