import torch
import timeit
import random
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split

n_epochs = 10
momentum = 0.9
random_seed = 1
w_decay = 1e-05
log_interval = 100
learning_rate = 1e-03
batch_size_train = 200
batch_size_test = 1000

image_dimension = 28 * 28

# CNN Architecture for MNIST
class Net(nn.Module): 
    def __init__(self, in_channels):
        super(Net, self).__init__()

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
        
        return F.log_softmax(x, dim=1)


def CNN(device):
    # Load MNIST dataset
    train_loader, validation_loader, test_loader = load_MNIST()
    
    in_channels = 1 # MNIST dataset has 1 channel

    # Define the model and optimizer
    cnn_model = Net(in_channels).to(device)
    optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate, weight_decay=w_decay)
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
            + " Set: | Average Loss: ({:.4f}) | Accuracy Raw: ({}/{}) | Accuracy Percentage: ({:.0f}%) |\n".format(
                loss,
                correct,
                len(data_loader.dataset),
                100.0 * correct / len(data_loader.dataset),
            )
        )

    # Training the model
    eval("-", validation_loader, cnn_model, "Validation")
    for epoch in range(1, n_epochs + 1):
        train(train_loader, cnn_model, optimizer)
        eval(epoch, validation_loader, cnn_model, "Validation")

    # Testing the model
    eval("-", test_loader, cnn_model, "Test")

    # Save the model
    results = dict(model=cnn_model)
    return results

# HELPER FUNCTIONS
def load_MNIST():
    # Load MNIST training set
    MNIST_training = torchvision.datasets.MNIST(
        "/MNIST_dataset/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    # Load MNIST test set
    MNIST_test_set = torchvision.datasets.MNIST(
        "/MNIST_dataset/",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    # Create a training and a validation set
    MNIST_training_set, MNIST_validation_set = random_split(
        MNIST_training, [48000, 12000]
    )

    # Create data loaders
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


def train_for_param(epoch, optimizer, model, data_loader):
    one_hot = torch.nn.functional.one_hot
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Training function
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.mse_loss(output, one_hot(target, num_classes=10).float())
        loss.backward()
        optimizer.step()



