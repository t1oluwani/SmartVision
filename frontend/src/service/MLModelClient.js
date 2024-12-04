import axios from "axios";

const API_URL = "http://localhost:5000"; // Backend URL

async function modelTrain(model_type) {
  try {
    const response = await axios.post(`${API_URL}/train/${model_type}`);
    return response.data;

  } catch (error) {
    alert("Error: Model Training Failed.");
    console.error(error);
  }
}

async function modelClear(model_type) {
  try {
    const response = await axios.get(`${API_URL}/clear/${model_type}`);
    return response.status;

  } catch (error) {
    alert("Error: Model Clearing Failed.");
    console.error(error);
  }
}

async function modelPredict(model_type, img_data) {
  try {
    const response = await axios.post(`${API_URL}/predict/${model_type}`, {
      img_data: img_data,
    });
    return response.data;

  } catch (error) {
    alert("Error: Model Prediction Failed.");
    console.error(error);
  }
}

export { modelTrain, modelClear, modelPredict };