
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.path.abspath('../')) # add the path to the directory with methods
from methods import LogisticRegressionModel

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
    print(output)

    return predicted

# HELPER FUNCTION - Load the trained model
def load_model(model_path, device):
    model = LogisticRegressionModel().to(device)  
    model.load_state_dict(torch.load(model_path, weights_only=True))  
    model.eval()
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # results = logistic_regression(device)
    # model = results["model"]
    
    # Load pre-trained model 
    model_path = 'LR_model.pth' 
    model = load_model(model_path, device)

    # Path to the uploaded image
    # image_path = "test_images/image_2(0).png"
    # image_path = "test_images/image_5(0).jpeg"
    # image_path = "test_images/image_5(1).png"
    image_path = "test_images/image_7(0).jpg"

    # Predict the class of the uploaded image
    predicted_class = process_and_predict(image_path, model, device)
    print(f"The model predicts the uploaded image as: {predicted_class}")
