import timeit
from collections import OrderedDict

import torch
from torchvision import transforms, datasets

import os, sys
sys.path.append(os.path.abspath('../')) # add the path to the directory with methods
from methods import logistic_regression

torch.multiprocessing.set_sharing_strategy('file_system')

# Function to test the model
def test(model, device):
    # Load the test dataset
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]))

    # Create a test dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True)

    # Set the model to evaluation mode
    model.eval()
    num_correct = 0
    total = 0
    
    # Iterate over the test data and generate predictions
    for batch_idx, (data, targets) in enumerate(test_loader):
        data = data.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            output = model(data)
            predicted = torch.argmax(output, dim=1)
            total += targets.size(0)
            num_correct += (predicted == targets).sum().item()

    # Compute the accuracy
    acc = float(num_correct) / total
    return acc

# Main function: Run the logistic regression model
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start = timeit.default_timer() # start the timer

    # Run the logistic regression model
    results = logistic_regression(device) # triggers logistic regression function
    model = results['model']
    if model is None:
        print('No Model was Trained')
        return
    else:
        print("Attempting to save the model")
        torch.save(model.state_dict(), 'LR_model.pth') # Saves only the model parameters
        print("Model saved successfully")
        # torch.save(model, 'LR_model_full.pth') # Saves the full model

    # Compute the run time
    stop = timeit.default_timer() # stop the timer
    run_time = stop - start

    # Test the model and get the accuracy
    accuracy = test(
        model,
        device,
    )
    result = OrderedDict(
        accuracy=accuracy,
        run_time=run_time
    )
    # Print the results
    print(f"Testing Results:")
    for key in result:
        print(f"\t{key}: {result[key]}")

if __name__ == "__main__":
    main()
