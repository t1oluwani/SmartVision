from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split
from torchvision import datasets, transforms

random_seed = 1
torch.manual_seed(random_seed)
one_hot = torch.nn.functional.one_hot

batch_size_train = 256
batch_size_test = 1024
image_dimension = 28 * 28


def load_MNIST():
    MNIST_data_set = datasets.MNIST(
        "/MNIST_dataset/",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]
        ),
    )
    MNIST_test_set = datasets.MNIST(
        "/MNIST_dataset/",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]
        ),
    )
    MNIST_training_set, MNIST_validation_set = random_split(
        MNIST_data_set, [48000, 12000]
    )

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


# Load MNIST dataset into train, validation, and test sets
train_loader, validation_loader, test_loader = load_MNIST()


# CNN Architecture for MNIST
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # CNN Hyperparameters
        self.n_epochs = 10
        self.w_decay = 1e-05
        self.learning_rate = 1e-03

        # CNN Layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
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


# FNN Architecture for MNIST
class FNNModel(nn.Module):
    def __init__(self):
        super(FNNModel, self).__init__()

        # FNN Hyperparameters
        self.n_epochs = 10
        self.momentum = 0.5
        self.learning_rate = 1e-01

        # FNN Layers
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)

        return output

    # Logistic regression model


class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()

        # Logistic Regression Hyperparameters
        self.n_epochs = 10
        self.momentum = 0.9
        self.random_seed = 1
        self.w_decay = 1e-05
        self.log_interval = 100
        self.learning_rate = 1e-03

        # Logistic Regression Layer
        self.fc = nn.Linear(image_dimension, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        output = nn.functional.softmax(x, dim=1)

        return output

def CNN(device):
    cnn_model = CNNModel().to(device)
    cnn_optimizer = optim.Adam(
        cnn_model.parameters(),
        lr=cnn_model.learning_rate,
        weight_decay=cnn_model.w_decay,
    )

    # Train the model
    for epoch in range(1, cnn_model.n_epochs + 1):
        train(device, cnn_model, cnn_optimizer, train_loader, loss_type="mse")
        _, validation_accuracy = evaluate(
            device,
            cnn_model,
            validation_loader,
            epoch,
            loss_type="nll",
            dataset="Validation",
        )
    
    # Test the model
    evaluation_loss, test_accuracy = evaluate(device, cnn_model, test_loader, epoch="T", loss_type="ce", dataset="Test")
    
    # Save the model
    results = dict(
        model=cnn_model,
        avg_loss=evaluation_loss,
        test_accuracy=test_accuracy,
        validation_accuracy=validation_accuracy,
    )
    return results

def FNN(device):
    fnn_model = FNNModel().to(device)
    fnn_optimizer = optim.SGD(
        fnn_model.parameters(), lr=fnn_model.learning_rate, momentum=fnn_model.momentum
    )

    # Train the model
    for epoch in range(1, fnn_model.n_epochs + 1):
        train(device, fnn_model, fnn_optimizer, train_loader, loss_type="ce")
        _, validation_accuracy = evaluate(
            device,
            fnn_model,
            validation_loader,
            epoch,
            loss_type="ce",
            dataset="Validation",
        )

    # Test the model
    evaluation_loss, test_accuracy = evaluate(device, fnn_model, test_loader, epoch="T", loss_type="ce", dataset="Test")
    
    # Save the model
    results = dict(
        model=fnn_model,
        avg_loss=evaluation_loss,
        test_accuracy=test_accuracy,
        validation_accuracy=validation_accuracy,
    )
    return results

def LogisticRegression(device):
    logistic_model = LogisticRegressionModel().to(device)
    logistic_optimizer = optim.Adam(
        logistic_model.parameters(),
        lr=logistic_model.learning_rate,
        weight_decay=logistic_model.w_decay,
    )

    # Train the model
    for epoch in range(1, logistic_model.n_epochs + 1):
        train(device, logistic_model, logistic_optimizer, train_loader, loss_type="mse")
        _, validation_accuracy = evaluate(
            device,
            logistic_model,
            validation_loader,
            epoch,
            loss_type="mse",
            dataset="Validation",
        )

    # Test the model
    evaluation_loss, test_accuracy = evaluate(device, logistic_model, test_loader, epoch="T", loss_type="ce", dataset="Test")
    
    # Save the model
    results = dict(
        model=logistic_model,
        avg_loss=evaluation_loss,
        test_accuracy=test_accuracy,
        validation_accuracy=validation_accuracy,
    )
    return results

# HELPER FUNCTIONS: ====================================================================================================

# Training function
def train(device, model, optimizer, data_loader, loss_type):
    model.train()

    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = get_loss(loss_type, output, target)
        loss.backward()
        optimizer.step()


# Evaluation function
def evaluate(device, model, data_loader, epoch, loss_type, dataset):
    model.eval()

    loading_bar = tqdm(data_loader, ncols=100, position=0, leave=True)

    correct = 0
    eval_loss = 0
    with torch.no_grad():
        for data, target in loading_bar:
            data = data.to(device)
            target = target.to(device)
            output = model(data)

            loss = get_loss(loss_type, output, target)
            eval_loss += loss.item()

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

            loading_bar.set_description("Epoch " + str(epoch) + ": ")
    eval_loss /= len(data_loader.dataset)
    acc_percentage = 100.0 * correct / len(data_loader.dataset)
    acc_percentage = acc_percentage.item() # detach tensor

    print(
        dataset,
        "Set: | Average Loss: ({:.4f}) | Accuracy Raw: ({}/{}) | Accuracy Percentage: ({:.2f}%) |\n".format(
            eval_loss,
            correct,
            len(data_loader.dataset),
            acc_percentage,
        ),
    )
    
    return eval_loss, acc_percentage 

# Loss function
def get_loss(loss_type, output, target):
    if loss_type == "ce":
        return F.cross_entropy(output, target)
    elif loss_type == "mse":
        return F.mse_loss(output, one_hot(target, num_classes=10).float())
    elif loss_type == "nll":
        return F.nll_loss(output, target)
    else:
        print("Invalid loss function")
        return None
