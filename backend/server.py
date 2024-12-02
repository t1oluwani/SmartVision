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

# Directory to save uploaded images
UPLOAD_FOLDER = 'canvas_image'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model placeholders (load your trained model here)
CNN_model = None
FNN_model = None
LR_model = None

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

# # Route: Upload and preprocess an image
# @app.route('/upload', methods=['POST'])
# def upload_image():
#     """
#     Endpoint to upload an image.
#     """
#     # Clear canvas_images directory
#     for file in os.listdir("canvas_image"):
#         os.remove(os.path.join("canvas_image", file))
        
#     try:
#         # Upload the image
        
        
#         return jsonify({"message": "Image uploaded successfully!"}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


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


# Route: Use a model to predict an image classification
@app.route('/predict/<model_type>', methods=['POST'])
def preprocess_and_predict(model_type):
    """
    Endpoint to predict the class of an uploaded image using the model.
    """
    if not model_type: 
        return jsonify({"error": "Please provide a model type."}), 
    
    if 'canvas_drawing' not in request.files:
        return jsonify({"error": "No Image Found"}), 400
    
    
    
    try:
        if model_type == "CNN":
            pass
        elif model_type == "FNN":
            pass
        elif model_type == "LR":
            pass
        else:
            return jsonify({"error": "Invalid model type. Choose from CNN, FNN, or LR."}), 400
        

        return jsonify({"predicted_class"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
