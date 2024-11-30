import os
from PIL import Image

import torch
from torchvision import transforms

from flask_cors import CORS
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Enable CORS
CORS(app, resources={r"/api/*": {
    "origins": ["http://localhost:8080"],
    "methods": ["GET", "POST"],
    "allow_headers": ["Content-Type", "Authorization"],
    "supports_credentials": True
}})

# Model placeholder (load your trained model here)
model = None

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device Availale: ", device)


# Test Route
@app.route('/', methods=['GET'])
def test():
    """
    Test route to check if the server is running.
    """
    return jsonify({"message": "Server is up and running!"}), 200

# Route: Upload and preprocess an image
@app.route('/upload', methods=['POST'])
def upload_and_preprocess():
    """
    Endpoint to upload and preprocess an image.
    """
    try:
        
        return jsonify({"message": "Image uploaded and preprocessed successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Route: Train or retrain the model
@app.route('/train', methods=['POST'])
def train_and_save():
    """
    Endpoint to train or retrain the model.
    """
    try:
        # Example training code (replace with your training logic)
        global model
        # Call your model training function here and return status
        # Example: `model = train_function()`
        return jsonify({"message": "Model trained/retrained successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Route: Use model to predict an image class
@app.route('/predict', methods=['POST'])
def predict_image():
    """
    Endpoint to predict the class of an uploaded image using the model.
    """
    try:
        global model
        if model is None:
            return jsonify({"error": "Model not loaded. Train the model first."}), 400

        return jsonify({"predicted_class"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
