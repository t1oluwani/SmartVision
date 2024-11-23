from flask import Flask, request, jsonify
import os
from PIL import Image
import torch
from torchvision import transforms

# Initialize Flask app
app = Flask(__name__)

# Model placeholder (load your trained model here)
model = None

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Route: Upload and preprocess an image
@app.route('/upload', methods=['POST'])
def upload_and_preprocess():
    """
    Endpoint to upload and preprocess an image.
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        # Save and preprocess the image
        image_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(image_path)

        # Preprocess the image
        img = Image.open(image_path).convert("L")  # Convert to grayscale
        transform = transforms.Compose([
            transforms.Resize((28, 28)),                # Resize to 28x28
            transforms.ToTensor(),                     # Convert to tensor
            transforms.Normalize((0.1307,), (0.3081,)) # MNIST normalization
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)
        return jsonify({"message": "Image uploaded and preprocessed successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Route: Train or retrain the model
@app.route('/train', methods=['POST'])
def train_model():
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
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        # Save and preprocess the image
        image_path = os.path.join("uploads", file.filename)
        file.save(image_path)

        # Preprocess the image
        img = Image.open(image_path).convert("L")
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Predict
        global model
        if model is None:
            return jsonify({"error": "Model not loaded. Train the model first."}), 400

        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            predicted = torch.argmax(output, dim=1).item()

        return jsonify({"predicted_class": predicted}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
