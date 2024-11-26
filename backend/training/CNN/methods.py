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
                loss += F.mse_loss(
                    output, one_hot(target, num_classes=10).float(), size_average=False
                ).item()
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


# def tune_hyper_parameter(target_metric, device):
#     start = timeit.default_timer()  # start the timer

#     # Initialize variables
#     best_params = None
#     best_metric = 0

#     # Define hyperparameters to search over
#     learning_rates = np.linspace(1e-04, 1e-03, num=10)
#     weight_decays = np.linspace(1e-05, 1e-04, num=10)
#     num_epochs = list(range(5, 10))

#     # Load MNIST dataset
#     image_dimension = 28 * 28
#     train_loader, validation_loader, test_loader = load_MNIST()

#     for _ in range(25):
#         # Randomly sample hyperparameters
#         lr_val = random.choice(learning_rates)
#         wd_val = random.choice(weight_decays)
#         ne_val = random.choice(num_epochs)
#         print(
#             f"Training with [learning_rate={lr_val}, weight_decay={wd_val}, epochs={ne_val}]"
#         )

#         # Define the model and optimizer
#         logistic_model = LogisticRegressionModel().to(device)
#         optimizer = optim.Adam(
#             logistic_model.parameters(), lr=lr_val, weight_decay=wd_val
#         )

#         # Train model and evaluate validation based on the target metric (accuracy or loss)
#         if target_metric == "acc":
#             metric = evaluate_accuracy(logistic_model, validation_loader)
#             for epoch in range(1, ne_val + 1):
#                 train_for_param(epoch, optimizer, logistic_model, validation_loader)
#                 metric = evaluate_accuracy(logistic_model, validation_loader)
#         elif target_metric == "loss":
#             metric = evaluate_loss(logistic_model, validation_loader)
#             for epoch in range(1, ne_val + 1):
#                 train_for_param(epoch, optimizer, logistic_model, validation_loader)
#                 metric = evaluate_loss(logistic_model, validation_loader)
#         print(f"Validation {target_metric}: {metric:.4f}")

#         # Check if the current hyperparameters yield a better accuracy
#         if (target_metric == "acc" and metric > best_metric) or (
#             target_metric == "loss" and metric < best_metric
#         ):
#             best_metric = metric
#             best_params = {
#                 "learning_rate": lr_val,
#                 "weight_decay": wd_val,
#                 "epochs": ne_val,
#             }

#     stop = timeit.default_timer()  # stop the timer
#     run_time = stop - start
#     print(f"Ran with a runtime of {run_time:.3f} seconds")

#     return best_params, best_metric


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
        MNIST_test_set, batch_size=batch_size_test, shuffle=True
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


def evaluate_accuracy(model, data_loader):
    loss = 0
    correct = 0
    dataset = "Validation"
    one_hot = torch.nn.functional.one_hot
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Evaluation function (accuracy)
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            loss += F.mse_loss(
                output, one_hot(target, num_classes=10).float(), size_average=False
            ).item()
    loss /= len(data_loader.dataset)
    accuracy = 100.0 * correct / len(data_loader.dataset)
    return accuracy


def evaluate_loss(model, data_loader):
    loss = 0
    correct = 0
    dataset = "Validation"
    one_hot = torch.nn.functional.one_hot
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Evaluation function (loss)
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            loss += F.mse_loss(
                output, one_hot(target, num_classes=10).float(), size_average=False
            ).item()
    loss /= len(data_loader.dataset)
    return loss
