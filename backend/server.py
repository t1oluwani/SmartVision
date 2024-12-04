import os
import timeit
from PIL import Image

import torch
from torchvision import transforms, datasets

from flask_cors import CORS
from flask import Flask, request, jsonify

from models_util import CNN, FNN, LogisticRegression
from models_util import CNNModel, FNNModel, LogisticRegressionModel

# Initialize Flask app
app = Flask(__name__)

# Enable CORS
CORS(
    app,
    resources={
        r"/*": {
            "origins": ["http://localhost:8080"], # Frontend URL
            "methods": ["GET", "POST"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True,
        }
    },
)

# Device and multiprocessing configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy("file_system")

# HELPER FUNCTIONS

# Loads the model according to the model type
def load_model(model_path, model_type):
    if model_type == "CNN":
        model = CNNModel().to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
    elif model_type == "FNN":
        model = FNNModel().to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
    elif model_type == "LR":
        model = LogisticRegressionModel().to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
    else:
        return (
            jsonify(
                {"error": "Invalid model type to load. Choose from CNN, FNN, or LR."}
            ),
            400,
        )

    return model

# ROUTES AND ENDPOINTS


# Test Route
@app.route("/", methods=["GET"])
def test():
    """
    Test route to check if the server is running.
    """
    return jsonify({"message": "Server is up and running!"}), 200


# Route: Train or retrain the model
@app.route("/train/<model_type>", methods=["POST"])
def train_and_save(model_type):
    """
    Endpoint to train and save selected model.
    """
    # Check if model type is provided
    if not model_type:
        return jsonify({"error": "Please provide a model type."}), 400

    # Check if model type is valid
    if model_type not in ["CNN", "FNN", "LR"]:
        return (
            jsonify(
                {"error": "Invalid model type to train. Choose from CNN, FNN, or LR."}
            ),
            400,
        )
    try:
        start_timer = timeit.default_timer()
        model_path = f"ml_models/{model_type}_model.pth"

        # Train the model
        if model_type == "CNN":
            training_results = CNN(device)
        elif model_type == "FNN":
            training_results = FNN(device)
        elif model_type == "LR":
            training_results = LogisticRegression(device)

        # Save the model and stats
        trained_model = training_results["model"]
        average_loss = training_results["avg_loss"]
        test_accuracy = training_results["test_accuracy"]
        training_accuracy = training_results["validation_accuracy"]
        
        # Save trained model to disk
        torch.save(trained_model.state_dict(), model_path)

        # Calculate total runtime
        stop_timer = timeit.default_timer()
        total_runtime = stop_timer - start_timer

        return (
            jsonify(
                {
                    "train_accuracy": training_accuracy,
                    "test_accuracy": test_accuracy,
                    "average_loss": average_loss,
                    "run_time": total_runtime,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route: Use a model to predict an images classification
@app.route("/predict/<model_type>", methods=["POST"])
def preprocess_and_predict(model_type):
    """
    Endpoint to predict the class of an uploaded image using selected model.
    """
    print("Predicting image class...")
    # Check if model type is provided
    if not model_type or model_type not in ["CNN", "FNN", "LR"]:
        return jsonify({"error": "Invalid Model Type."}), 400

    # Check if image is provided
    if "canvas_drawing" not in request.files:
        return jsonify({"error": "No Image Found with Request"}), 400

    # Preprocess the image for prediction
    unprocessed_image = request.files["canvas_drawing"]
    unprocessed_image = Image.open(unprocessed_image)

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.Resize((28, 28)),  # Resize to 28x28
            transforms.ToTensor(),  # Convert to tensor
            transforms.Lambda(lambda x: 1 - x),  # Invert colours (black background)
            transforms.Normalize((0.1307,), (0.3081,)),  # Normalize with MNIST stats
        ]
    )
    processed_image = transform(unprocessed_image).unsqueeze(0).to(device)

    try:
        if model_type == "CNN":
            model_path = "ml_models/CNN_model.pth"
        elif model_type == "FNN":
            model_path = "ml_models/FNN_model.pth"
        elif model_type == "LR":
            model_path = "ml_models/LR_model.pth"
        else:
            return (
                jsonify(
                    {
                        "error": "Invalid model type to predict. Choose from CNN, FNN, or LR."
                    }
                ),
                400,
            )

        model = load_model(model_path, model_type)
        model.eval()

        # Make a prediction
        with torch.no_grad():
            output = model(processed_image)
            predicted_class = torch.argmax(output, dim=1).item()

        return jsonify({"predicted_class": predicted_class}), 200
    except Exception as e:
        print("Failed to predict image class.")
        return jsonify({"error": str(e)}), 500


# Route: Clear saved model by type
@app.route("/clear/<model_type>", methods=["GET"])
def clear_model(model_type):
    """
    Endpoint to clear the saved model by type.
    """
    if model_type == "CNN":
        os.remove("ml_models/CNN_model.pth")
    elif model_type == "FNN":
        os.remove("ml_models/FNN_model.pth")
    elif model_type == "LR":
        os.remove("ml_models/LR_model.pth")

    else:
        return (
            jsonify(
                {"error": "Invalid model type to clear. Choose from CNN, FNN, or LR."}
            ),
            400,
        )

    return jsonify({"message": f"{model_type} model cleared successfully!"}), 200


if __name__ == "__main__":
    app.run(port=5000, debug=True)
