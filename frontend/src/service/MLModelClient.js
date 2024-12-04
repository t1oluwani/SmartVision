import axios from "axios";

const API_URL = "http://localhost:5000"; // Backend URL

async function trainModel(model_type) {
  const response = await axios.post(`${API_URL}/...`);
  return response.data;
}

async function clearModel(model_type) {
  const response = await axios.post(`${API_URL}/...`);
  return response.data;
}

async function modelPredict(model_type, img_data) {
  const response = await axios.post(`${API_URL}/...`, img_data);
  return response.data;
}

export { trainModel, clearModel, modelPredict };