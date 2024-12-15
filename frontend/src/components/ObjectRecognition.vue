<template>
  <div class="OR-Scope">
    <div class="tab-content">

      <div class="left-container">
        <div class="upload-box">
          <input class="upload-image" type="file" accept="image/*" @change="updateSelectedImage" />
          <label class="upload-label" >{{ imageSelected || 'No Image Selected' }}</label>
        </div>
        <button @click="identify">Identify Item</button>
      </div>

      <div class="middle-container">
        <div class="training-status">
          <h3>Visual Recognition Model Status</h3>
          <p>{{ modelTrained ? 'Model is trained' : 'Model not trained' }}</p>
          <button @click="trainModel">Train / Retrain Model</button>
        </div>

        <div v-if="modelTrained" class="model-specs">
          <h3>Model Specifications</h3>
          <p><strong>Avg Loss:</strong> {{ modelSpecs.avgLoss }}</p>
          <p><strong>Training Accuracy:</strong> {{ modelSpecs.trainAccuracy }}%</p>
          <p><strong>Test Accuracy:</strong> {{ modelSpecs.testAccuracy }}%</p>
          <p><strong>Time Spent to Train:</strong> {{ modelSpecs.trainingTime }}s</p>
        </div>

        <div class="model-prediction">
          <h3>Model Predictions</h3>
          <p>The Model predicts the uploaded image to belong to the category of:</p>
          <div class="predicted-type">
            <strong>{{ modelPrediction ? modelPrediction : "?" }}</strong>
          </div>
        </div>
      </div>

      <div class="right-container">
        <button @click="generate">Generate Item</button>
        <div class="generate-box">
          <div class="generate-placeholder">
            <label class="generate-label">Generated Image will appear here</label>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      imageSelected: '',
      loadingState: false,
      modelTrained: false,
      modelSpecs: {
        avgLoss: 0,
        trainAccuracy: 0,
        testAccuracy: 0,
        trainingTime: 0,
      },
      modelPrediction: null,
    };
  },
  methods: {
    updateSelectedImage(event) {
      this.imageSelected = event.target.files[0].name;
    },
    identify() {
      // Placeholder for identify function (no logic)
      alert(`Identifying image`);
    },
    trainModel() {
      // Placeholder for trainModel function (no logic)
      alert('Training model');

      // Simulate training process for testing
      this.modelTrained = true;
      this.modelSpecs = {
        avgLoss: 0.3,
        trainAccuracy: 92,
        testAccuracy: 88,
        trainingTime: 120, // in seconds
      }
    },
    generate() {
      // Placeholder for random item generation (no logic)
      alert('Generating a random item from the category');
    },
  },
};
</script>
