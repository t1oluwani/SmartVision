import requests
import os

cnn_score = 0
fnn_score = 0
lr_score = 0
# Loop over 10 images
for num in range(10):
    # Prepare the image file
    image_path = f"./test_images/image_{num}.jpg"

    # Make a POST request to the Flask API with the image file
    with open(image_path, "rb") as image_file:
        response1 = requests.post(
            f"http://localhost:5000/predict/CNN", files={"canvas_drawing": image_file}
        )
        
    # Make a POST request to the Flask API with the image file
    with open(image_path, "rb") as image_file:
        response2 = requests.post(
            f"http://localhost:5000/predict/FNN", files={"canvas_drawing": image_file}
        )
        
    # Make a POST request to the Flask API with the image file
    with open(image_path, "rb") as image_file:
        response3 = requests.post(
            f"http://localhost:5000/predict/LR", files={"canvas_drawing": image_file}
        )

    # Handle scores
    if num == response1.json()['predicted_class']:
        cnn_score += 1
    if num == response2.json()['predicted_class']:
        fnn_score += 1
    if num == response3.json()['predicted_class']:
        lr_score += 1
        
    print(f"Performance - CNN: {response1.json()}, FNN: {response2.json()}, LR: {response3.json()}")
print(f"Final Scores - CNN: {cnn_score}, FNN: {fnn_score}, LR: {lr_score}")
   
    
