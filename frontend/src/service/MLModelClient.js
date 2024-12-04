import axios from "axios";

const API_URL = "http://localhost:5000"; // Backend URL

async function modelTrain(model_type) {
  try {
    const response = await axios.post(`${API_URL}/train/${model_type}`);

    const formatted_data = {
      train_accuracy: response.data.train_accuracy.toFixed(2),
      test_accuracy: response.data.test_accuracy.toFixed(2),
      average_loss: response.data.average_loss.toFixed(4),
      run_time: response.data.run_time.toFixed(1),
    };

    if (response && response.data) {
      console.log("Model training result:", formatted_data);
      return formatted_data;
    }

  } catch (error) {
    console.error("Model training failed:", error);
    alert("Error: Model Training Failed.");
  }
}

async function modelClear(model_type) {
  try {
    const response = await axios.get(`${API_URL}/clear/${model_type}`);

    if (response) {
      return response.status;
    }

  } catch (error) {
    console.error("Model clearing failed:", error);
    alert("Error: Model Clearing Failed.");
  }
}

async function modelPredict(model_type, img_data) {
  try {
    const response = await axios.post(`${API_URL}/predict/${model_type}`, img_data,
      {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

    if (response && response.data) {
      console.log("Model prediction result:", response.data);
      return response.data;
    }

  } catch (error) {
    console.error("Model prediction failed:", error);
    alert("Error: Model Prediction Failed.");
  }
}

export { modelTrain, modelClear, modelPredict };