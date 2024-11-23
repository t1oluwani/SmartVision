
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.path.abspath('../')) # add the path to the directory with methods
from methods import logistic_regression


def process_and_predict(image_path, model, device):
    # Load the image
    img = Image.open(image_path).convert("L")  # Convert to grayscale

    # Preprocess the image
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),  # Resize to 28x28
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize((0.1307,), (0.3081,)),  # Normalize with MNIST stats
        ]
    )
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension

    # Model inference
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        output = model(img_tensor)
        predicted = torch.argmin(output, dim=1).item()  # Get the predicted class

    plt.imshow(img_tensor.squeeze().cpu().numpy(), cmap="gray")
    plt.show()  # Display the image
    print(img_tensor.shape)
    img.show()
    print(output)

    return predicted


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model (use your logistic regression training code)
    results = logistic_regression(device)
    model = results["model"]

    # Path to the uploaded image (replace with your image path)
    image_path = "test_images/image_2(0)"
    # image_path = "test_images/image_5(0)"
    # image_path = "test_images/image_5(1)"

    # Predict the class of the uploaded image
    predicted_class = process_and_predict(image_path, model, device)
    print(f"The model predicts the uploaded image as: {predicted_class}")
