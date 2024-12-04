import axios from "axios";

const API_URL = "http://localhost:5000"; // Backend URL

async function trainModel(model_type) {
  const response = await axios.post(`${API_URL}/train/${model_type}`);
  return response.data;
}

async function clearModel(model_type) {
  const response = await axios.post(`${API_URL}/clear/${model_type}`);
  return response.data;
}

async function modelPredict(model_type, img_data) {
  const response = await axios.post(`${API_URL}/predict/${model_type}`, img_data);
  return response.data;
}

export { trainModel, clearModel, modelPredict };