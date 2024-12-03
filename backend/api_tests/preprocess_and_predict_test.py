import requests

cnn_score = 0
fnn_score = 0
lr_score = 0

for num in range(10):
    image_path = f"./test_images/image_{num}.jpg"

    with open(image_path, "rb") as image_file:
        cnn_response = requests.post(
            f"http://localhost:5000/predict/CNN", files={"canvas_drawing": image_file}
        )
    with open(image_path, "rb") as image_file:
        fnn_response = requests.post(
            f"http://localhost:5000/predict/FNN", files={"canvas_drawing": image_file}
        )
    with open(image_path, "rb") as image_file:
        lr_response = requests.post(
            f"http://localhost:5000/predict/LR", files={"canvas_drawing": image_file}
        )

    if num == cnn_response.json()['predicted_class']:
        cnn_score += 1
    if num == fnn_response.json()['predicted_class']:
        fnn_score += 1
    if num == lr_response.json()['predicted_class']:
        lr_score += 1
        
    print(f"Performance - CNN: {cnn_response.json()}, FNN: {fnn_response.json()}, LR: {lr_response.json()}")
print(f"Final Scores - CNN: {cnn_score}, FNN: {fnn_score}, LR: {lr_score}")
   
    
