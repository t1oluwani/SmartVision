from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split
from torchvision import datasets, transforms

random_seed = 1
torch.manual_seed(random_seed)

batch_size_train = 256
batch_size_test = 1024

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


def FNN(device):
    
    fnn_model = FNNModel().to(device)
    fnn_optimizer = optim.SGD(fnn_model.parameters(), lr=fnn_model.learning_rate, momentum=fnn_model.momentum)
    
    train_loader, validation_loader, test_loader = load_MNIST()

    # Training function
    def train(data_loader, model, optimizer):
        model.train()

        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)

            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()


    # Evaluation function
    def evaluate(epoch, data_loader, model, dataset):
        model.eval()
        
        correct = 0
        eval_loss = 0
        loading_bar = tqdm(data_loader, ncols=100, position=0, leave=True)
        
        with torch.no_grad():
            for data, target in loading_bar:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                loss = F.cross_entropy(output, target)
                eval_loss += loss.item()
                loading_bar.set_description("Epoch " + str(epoch) + ": ")
        eval_loss /= len(data_loader.dataset)

        print(dataset, "Set: | Average Loss: ({:.4f}) | Accuracy Raw: ({}/{}) | Accuracy Percentage: ({:.2f}%) |\n".format(
                eval_loss, correct, len(data_loader.dataset), 100.0 * correct / len(data_loader.dataset)))

    # Train the model
    for epoch in range(1, fnn_model.n_epochs + 1):
        train(train_loader, fnn_model, fnn_optimizer)
        evaluate(epoch, validation_loader, fnn_model, "Validation")
        
    # Test the model
    evaluate("T", test_loader, fnn_model, "Test")

    # Save the model
    results = dict(model=fnn_model)
    return results


# HELPER FUNCTIONS:

def load_MNIST():
    # Load MNIST dataset
    MNIST_data_set = datasets.MNIST("/MNIST_dataset/", train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))])
    )
    MNIST_test_set = datasets.MNIST(
        "/MNIST_dataset/", train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))])
    )
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
