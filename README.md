# SmartVision

## Overview

Welcome to **SmartVision**, a cutting-edge visual recognition playground! This platform lets you explore and interact with advanced visual recognition models for handwritten numbers.<!-- and object recognition. --> The main features include:

- An interactive canvas where users can draw handwritten numbers and click a button to receive predictions generated by various trained models, each offering its own output.
- A feature that enables users to explore different machine learning models such as Logistic Regression, Convolutional Neural Networks (CNN), and Feedforward Neural Networks (FNN), helping them understand their use cases and capabilities.
<!-- - An option to upload an image for object recognition, classifying the objects into categories based on the FashionMNIST dataset.  -->

# Deployment

Visit the live deployment to explore SmartVision:<br><br>
[Deployment Link]

## Installation

To get started with StudyAI, follow these steps:

1. Clone or fork the repository:
   ```bash
   git clone https://github.com/t1oluwani/StudyAI.git
   ```
   
2. **Create and Activate Virtual Environment** (Optional, but Highly Recommended):
   - Navigate to the project directory:
     ```bash
     cd StudyAI
     ```
   - Create a virtual environment using `venv`:
     ```bash
     python -m venv venv
     ```
   - Activate the virtual environment:
     - On **Windows**:
       ```bash
       venv\Scripts\activate
       ```
     - On **macOS/Linux**:
       ```bash
       source venv/bin/activate
       ```
       
3. Navigate to the `backend` folder and install the backend dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. After installing the backend dependencies, navigate back to the root folder:
   ```bash
   cd ..
   ```

5. Start the backend server and frontend by running the following shell script:
   ```bash
   chmod +x start.sh
   ./start.sh
   ```

## Usage

1. **Toggle Between Models**
   - Select a model (Logistic Regression, CNN, or FNN) to compare their performance and results.
   - Each model's accuracy and processing details are displayed to help you understand their strengths and limitatio

2. **Handwriting Recognition**
   - Draw a handwritten number (0-9) directly on the provided canvas.
   - Click "Identify" to send your image to the backend.
   - The app processes the image and predicts the number class using your selected model (Logistic Regression, CNN, or FNN).

<!-- 3. **Object Recognition**
   - Upload an image to classify objects into categories based on the **FashionMNIST** dataset.
   - The application uses trained models to identify and predict the object class. -->

## Technologies Used

- **Frontend**: Vue.js.  
- **Backend**: Flask.
- **APIs**: Canvas API for drawing.
- **Machine Learning**: PyTorch and Torchvision.  
  - Models: Logistic Regression, Convolutional Neural Network (CNN), and Feedforward Neural Network (FNN).  
  - Datasets: MNIST for handwritten numbers, FashionMNIST for object recognition.

## API Documentation

API documentation will be available at: [https://github.com/t1oluwani/SmartVision/blob/main/APIdocs.md]

The API includes:
- **POST /train/<model_type>**: Train and save the selected model (CNN, FNN, Logistic Regression).
- **POST /predict/<model_type>**: Upload an image for classification using the selected model.
- **GET /clear/<model_type>**: Delete the saved model of the specified type (CNN, FNN, Logistic Regression).
- **GET /**: Test the server to ensure it is running.

## Testing

To ensure the backend is functioning correctly, you can run the provided test files:
```bash
python api_tests/<test_file>
```

# References

### PyTorch Website:
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Recipes](https://pytorch.org/tutorials/recipes/)
- [Getting Started Locally](https://pytorch.org/get-started/locally/)
- [PyTorch Basics (Beginner)](https://pytorch.org/tutorials/beginner/basics/)
- [Introduction to PyTorch (YouTube)](https://pytorch.org/tutorials/beginner/introyt/)

### Understanding of Logistic Regression, FNN, and CNN:
- [UofA CMPUT 328 Visual Recognition Class Notes](https://apps.ualberta.ca/catalogue/course/cmput/328)

### CSS Loading Loaders:
- [CSS Loaders](https://css-loaders.com/)

### Getting a `<Canvas>` as a Blob:
- [Stack Overflow - Access Blob Value](https://stackoverflow.com/questions/42458849/access-blob-value-outside-of-canvas-toblob-async-function)

## Contributions and Contact

Feel free to contribute to the project! You can submit issues or pull requests directly through the repository.
For questions or support, please reach out through the links on my profile or on LinkedIn.

## License

This project is licensed under the [MIT License](https://opensource.org/license/mit).

