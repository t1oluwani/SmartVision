<template>
  <div class="HN-Scope">
    <div class="tab-content">
      <div class="tab-container">
        <DrawingCanvas />

        <div class="model-switch">
          <label for="model">Choose Model:</label>
          <select v-model="selectedModel" id="model">
            <option value="logistic-regression">Logistic Regression</option>
            <option value="fneural-network">FeedFoward Neural Network (FNN)</option>
            <option value="cneural-network">Convolutional Neural Network (CNN)</option>
          </select>
        </div>

        <button id="identifybtn" @click="identify">Identify</button>
        <h3>The handwritten number has a value of __ with __% confidence</h3>

      </div>
    </div>
  </div>
</template>

<script>
import DrawingCanvas from './subcomponents/DrawingCanvas.vue';

export default {
  components: {
    DrawingCanvas,
  },
  data() {
    return {
      selectedModel: 'logistic-regression', // Default model
    };
  },
  methods: {
    identify() {
      // Placeholder for identify function (no logic)
      alert(`Identifying with ${this.selectedModel}`);
    },
  },
};
</script>
