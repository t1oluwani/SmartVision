import pbar

import torch
from torchvision import datasets, transforms

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split

n_epochs = 10
w_decay = 1e-05
learning_rate = 1e-03
batch_size_train = 200
batch_size_test = 1000

# FNN Architecture for MNIST
class FNNModel(nn.Module):
    def __init__(self, num_classes):
        super(FNNModel, self).__init__()

        self.num_classes = num_classes
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc1(x))
        x = F.relu(self.fc2(x))
        output =   self.fc3(x)

        return output
    
def FNN(device):
    train_loader, validation_loader, test_loader = load_MNIST()
    num_classes = 10 # 10 classes for MNIST

    # Define the model and optimizer
    fnn_model = FNNModel(num_classes).to(device)
    optimizer = optim.Adam(
        fnn_model.parameters(), lr=learning_rate, weight_decay=w_decay
    )
    one_hot = torch.nn.functional.one_hot

    # Training function
    def train(data_loader, model, optimizer):
        model.train()
        
        # pbar = tqdm(data_loader, ncols=100, position=0, leave=True)
        # for batch_idx, (data, target) in enumerate(pbar):
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)

            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            # pbar.set_description("Loss: {:.4f}".format(loss.item()))

    # Evaluation function
    def eval(epoch, data_loader, model, dataset):
        model.eval()
        
        correct = 0
        eval_loss = 0
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                loss = F.nll_loss(output, target)
                eval_loss += loss.item()
        eval_loss /= len(data_loader.dataset)

        # Print the evaluation results in easily readable format
        print(
            "Epoch "
            + str(epoch)
            + ": "
            + dataset
            + " Set: | Average Loss: ({:.4f}) | Accuracy Raw: ({}/{}) | Accuracy Percentage: ({:.2f}%) |\n".format(
                eval_loss,
                correct,
                len(data_loader.dataset),
                100.0 * correct / len(data_loader.dataset),
            )
        )

    # Training the model
    for epoch in range(1, n_epochs + 1):
        train(train_loader, cnn_model, optimizer)
        eval(epoch, validation_loader, cnn_model, "Validation")

    # Testing the model
    # eval("T", test_loader, cnn_model, "Test")

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