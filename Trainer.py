import matplotlib.pyplot as plt
import torch


class Trainer():
    def __init__(self):
        self.training_losses = []
        self.training_losses_avg = []
        self.testing_losses = []

    def train(self, dataloader, model, loss_fn, optimizer, device):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                self.training_losses.append(loss)

    def test(self, dataloader, model, loss_fn, device):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size

        print(f"Test Error: \n Accuracy: {
              (100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        self.testing_losses.append(test_loss)
        self.training_losses_avg.append(correct)

    def plot_training_losses(self):
        epochs = range(1, len(self.training_losses_avg) + 1)
        plt.plot(epochs, self.training_losses_avg, 'bo', label='Training loss')
        plt.plot(epochs, self.testing_losses, 'b', label='Testing loss')
        plt.title('Training and testing loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        # plt.savefig('loss.png')
