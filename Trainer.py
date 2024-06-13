import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Trainer():
    def __init__(self, network, momentum=0.9, learning_rate=1e-3):
        self.network = network
        self.training_losses = []
        self.avg_training_loss = []
        self.avg_testing_loss = []
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.optimizer = torch.optim.SGD(
            network.parameters(),
            momentum=momentum,
            lr=learning_rate
        )
        self.loss_fn = nn.CrossEntropyLoss()
        

    def train(self, dataloader, device):
        size = len(dataloader.dataset)
        self.network.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = self.network(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                self.training_losses.append(loss)
                

    def test(self, dataloader, device):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.network.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = self.network(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size

        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        self.avg_testing_loss.append(test_loss)
        mean = sum(self.training_losses) / len(self.training_losses)
        self.avg_training_loss.append(mean)
        self.training_losses = []

    def plot_training_testing_loss(self):
        epochs = range(1, len(self.avg_training_loss) + 1)
        plt.plot(epochs, self.avg_training_loss, 'r-+', label='Avg. Training loss')
        plt.plot(epochs, self.avg_testing_loss, 'b-', label='Avg. Test loss')
        plt.title(f'Avg. Training and Testing loss for {self.network}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
