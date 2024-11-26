import torch
from torchvision import datasets, transforms

import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split

n_epochs = 10
w_decay = 1e-05
learning_rate = 1e-03
batch_size_train = 200
batch_size_test = 1000


# CNN Architecture for MNIST
class CNNModel(nn.Module):
    def __init__(self, in_channels):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)

        return output


def CNN(device):
    train_loader, validation_loader, test_loader = load_MNIST()
    in_channels = 1  # MNIST dataset has 1 channel

    # Define the model and optimizer
    cnn_model = CNNModel(in_channels).to(device)
    optimizer = optim.Adam(
        cnn_model.parameters(), lr=learning_rate, weight_decay=w_decay
    )
    one_hot = torch.nn.functional.one_hot

    # Training function
    def train(data_loader, model, optimizer):
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)

            # loss = F.cross_entropy(output, target)
            loss = F.mse_loss(output, one_hot(target, num_classes=10).float())
            loss.backward()
            optimizer.step()

    # Evaluation function
    def eval(epoch, data_loader, model, dataset):
        loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                los = F.nll_loss(output, target)
                loss += los.item()
        loss /= len(data_loader.dataset)

        # Print the evaluation results in easily readable format
        print(
            "Epoch "
            + str(epoch)
            + ": "
            + dataset
            + " Set: | Average Loss: ({:.4f}) | Accuracy Raw: ({}/{}) | Accuracy Percentage: ({:.2f}%) |\n".format(
                loss,
                correct,
                len(data_loader.dataset),
                100.0 * correct / len(data_loader.dataset),
            )
        )

    # Training the model
    # eval("-", validation_loader, cnn_model, "Validation")
    for epoch in range(1, n_epochs + 1):
        train(train_loader, cnn_model, optimizer)
        eval(epoch, validation_loader, cnn_model, "Validation")

    # Testing the model
    # eval("-", test_loader, cnn_model, "Test")

    # Save the model
    results = dict(model=cnn_model)
    return results


# HELPER FUNCTIONS
def load_MNIST():
    # Load MNIST training set
    MNIST_data_set = datasets.MNIST(
        "/MNIST_dataset/",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    # Load MNIST test set
    MNIST_test_set = datasets.MNIST(
        "/MNIST_dataset/",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    # Create a training and a validation set
    MNIST_training_set, MNIST_validation_set = random_split(
        MNIST_data_set, [48000, 12000]
    )

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        MNIST_training_set, batch_size=batch_size_train, shuffle=True
    )
    validation_loader = torch.utils.data.DataLoader(
        MNIST_validation_set, batch_size=batch_size_train, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        MNIST_test_set, batch_size=batch_size_test, shuffle=False
    )

    return train_loader, validation_loader, test_loader