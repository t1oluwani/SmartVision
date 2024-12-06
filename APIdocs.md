# API Documentation

## Base URL
The base URL for this API is `http://localhost:5000`.

## Test Route
- **Route**: `/`
- **Method**: `GET`
- **Description**: This is a test route to check if the server is running.
- **Response**: 
    ```json
    {
        "message": "Server is up and running!"
    }
    ```

## Train or Retrain the Model
- **Route**: `/train/<model_type>`
- **Method**: `POST`
- **Parameters**: 
    - `model_type`: Type of the model to train. Acceptable values are `CNN`, `FNN`, and `LR`.
- **Description**: This endpoint allows you to train and save the selected model.
- **Response**: 
    ```json
    {
        "train_accuracy": <training_accuracy>,
        "test_accuracy": <test_accuracy>,
        "average_loss": <average_loss>,
        "run_time": <total_runtime>
    }
    ```

    - `train_accuracy`: Accuracy of the model on the training dataset.
    - `test_accuracy`: Accuracy of the model on the test dataset.
    - `average_loss`: Average loss during training.
    - `run_time`: Total time taken for training.

## Predict the Class of an Image
- **Route**: `/predict/<model_type>`
- **Method**: `POST`
- **Parameters**: 
    - `model_type`: Type of the model to use for prediction. Acceptable values are `CNN`, `FNN`, and `LR`.
    - The image should be uploaded with the key `canvas_drawing` in the request body.
- **Description**: This endpoint allows you to predict the class of an uploaded image using the selected model.
- **Request Example**: 
    - Upload an image with the key `canvas_drawing`.
- **Response**: 
    ```json
    {
        "predicted_class": <predicted_class>
    }
    ```

    - `predicted_class`: The predicted class of the image (e.g., a number corresponding to the digit in a handwritten image).

## Clear Saved Model by Type
- **Route**: `/clear/<model_type>`
- **Method**: `GET`
- **Parameters**: 
    - `model_type`: Type of the model to clear. Acceptable values are `CNN`, `FNN`, and `LR`.
- **Description**: This endpoint clears the saved model for the given type.
- **Response**: 
    ```json
    {
        "message": "<model_type> model cleared successfully!"
    }
    ```

    - `model_type`: The type of model that was successfully cleared (e.g., `CNN`, `FNN`, `LR`).
