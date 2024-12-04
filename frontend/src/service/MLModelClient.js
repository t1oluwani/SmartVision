import axios from "axios";

const API_URL = "http://localhost:5000"; // Backend URL

async function trainModel(model_type) {
  try {
    const response = await axios.post(`${API_URL}/train/${model_type}`);
    return response.data;
  } catch (error) {
    console.error(error);
  }
}

async function clearModel(model_type) {
  try {
    const response = await axios.post(`${API_URL}/clear/${model_type}`);
    return response.data;
  } catch (error) {
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
    console.error(error);
  }
}

export { trainModel, clearModel, modelPredict };