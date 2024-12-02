import requests

score = 0
# Loop over 10 images
for num in range(10):
    # Prepare the image file
    image_path = f"test_images/image_{num}.jpg"
    
    # Make a POST request to the Flask API with the image file
    with open(image_path, 'rb') as image_file:
        response = requests.post(
            f'http://localhost:5000/predict/CNN',  # Change model type here, e.g., CNN, FNN, LR
            files={'canvas_drawing': image_file}
        )

    # Handle the response
    if response.status_code == 200:
        predicted_class = response.json()[0]  # Assuming the response contains the predicted class
        score += 1 if num == predicted_class else 0
        print(f"The model predicts the uploaded image of {num} as: {predicted_class}")
    else:
        print(f"Error: {response.json().get('error')}")

print(f"Model Accuracy: {score}/10")
