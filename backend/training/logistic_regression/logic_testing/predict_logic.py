
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.path.abspath('../')) # add the path to the directory with methods
from methods import LogisticRegressionModel

def process_and_predict(image_path, model, device):
    # Load the image
    img = Image.open(image_path)

    # Preprocess the image
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),    # Convert to grayscale
            transforms.Resize((28, 28)),                    # Resize to 28x28
            transforms.ToTensor(),                          # Convert to tensor
            transforms.Lambda(lambda x: 1 - x),             # Invert colours (black background) 
            transforms.Normalize((0.1307,), (0.3081,)),     # Normalize with MNIST stats
        ]
    )
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension

    # Model inference
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        output = model(img_tensor)
        predicted = torch.argmax(output, dim=1).item()  # Get the predicted class

    # Debug Lines
    # plt.imshow(img_tensor.squeeze().cpu().numpy(), cmap="gray") # Convert back to visible image
    # plt.show()  # Display the image
    # print(img_tensor.shape)
    # print(output)

    return predicted

# HELPER FUNCTION - Load the trained model
def load_model(model_path, device):
    model = LogisticRegressionModel().to(device)  
    model.load_state_dict(torch.load(model_path, weights_only=True))  
    model.eval()
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pre-trained model 
    model_path = 'LR_model.pth' 
    model = load_model(model_path, device)
    
    
    score = 0
    for num in range(10):
        image_path = f"test_images/image_{num}.jpg"
        predicted_class = process_and_predict(image_path, model, device)
        score += 1 if num == predicted_class else 0
        print(f"The model predicts the uploaded image of {num} as: {predicted_class}")
    print(f"Model Accuracy: {score}/10")
        
