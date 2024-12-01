<template>
  <div class="OR-Scope">
    <h1>Coming Soon...</h1>
    <div class="tab-content">

      <div class="left-container">
        <div class="upload-box">
          <!-- Placeholder for image upload: White rectangle -->
          <div class="upload-placeholder">Upload Image Area</div>
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
          <p>The Model predicts the handwritten number to have a value of</p>
          <div class="predicted-value">
            <strong>{{ modelPrediction ? modelPrediction : "?" }}</strong>
          </div>
        </div>
      </div>

      <div class="right-container">
        <button @click="generate">Generate Item</button>
        <div class="generate-box">
          <!-- Placeholder for image generation: White rectangle -->
          <div class="generate-placeholder">Generated Image Area</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
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
