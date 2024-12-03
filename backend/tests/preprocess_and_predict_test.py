import requests
import os

score = 0
# Loop over 10 images
for num in range(1):
    # Prepare the image file
    image_path = f"./test_images/image_{num}.jpg"

    # Make a POST request to the Flask API with the image file
    with open(image_path, "rb") as image_file:
        response1 = requests.post(
            f"http://localhost:5000/predict/CNN"
        )
        
    # Make a POST request to the Flask API with the image file
    with open(image_path, "rb") as image_file:
        response2 = requests.post(
            f"http://localhost:5000/predict/FNN"
        )
        
    # Make a POST request to the Flask API with the image file
    with open(image_path, "rb") as image_file:
        response3 = requests.post(
            f"http://localhost:5000/predict/LR"
        )

    # Handle the response
    print(f"Performance - CNN: {response1.json()}, FNN: {response2.json()}, LR: {response3.json()}")
    # if response.status_code == 200:
    #     predicted_class = response.json()[
    #         0
    #     ]  # Assuming the response contains the predicted class
    #     score += 1 if num == predicted_class else 0
    #     print(f"The model predicts the uploaded image of {num} as: {predicted_class}")
    # else:
    #     print(f"Error: {response.json().get('error')}")

print(f"Model Accuracy: {score}/10")
