<template>
  <div class="HN-Scope">
    <div class="tab-content">

      <div class="left-container">
        <div class="model-info">
          <h3>{{
            selectedModel === 'logistic-regression' ? 'Logistic Regression' :
              selectedModel === 'fneural-network' ? 'FeedForward Neural Network' :
                'Convolutional Neural Network'
          }}</h3>

          <p v-if="selectedModel === 'logistic-regression'">
            Logistic Regression is a linear model used for binary classification tasks.
            Blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah
            blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah
            blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah
            blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah
            blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah
          </p>
          <p v-if="selectedModel === 'fneural-network'">
            Feedforward Neural Networks (FNN) are deep learning models with fully connected layers for classification.
            Blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah
            blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah
            blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah
            blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah
            blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah
          </p>
          <p v-if="selectedModel === 'cneural-network'">
            Convolutional Neural Networks (CNN) are a class of deep learning models commonly used for image recognition.
            Blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah
            blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah
            blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah
            blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah
            blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah
          </p>
        </div>
      </div>

      <div class="middle-container">
        <DrawingCanvas />

        <div class="choose-model">
          <label for="model">Choose Model: </label>
          <select v-model="selectedModel" id="model">
            <option value="logistic-regression">Logistic Regression</option>
            <option value="fneural-network">FeedFoward Neural Network (FNN)</option>
            <option value="cneural-network">Convolutional Neural Network (CNN)</option>
          </select>

          <button id="identifybtn" @click="identify">Identify</button>
        </div>
      </div>

      <div class="right-container">
        <div class="training-status">
          <h3>Visual Recognition Model Status</h3>
          <p>{{ modelTrained ? 'Model is trained' : 'Model not trained' }}</p>
          <button @click="trainModel">Train / Retrain Model</button>
        </div>

        <div v-if="selectedModel === 'cneural-network'">
          <div v-if="model.modelTrained" class="model-specs">
            <h3>Model Specifications</h3>
            <p><strong>Avg Loss:</strong> {{ modelSpecs.avgLoss }}</p>
            <p><strong>Training Accuracy:</strong> {{ modelSpecs.trainAccuracy }}%</p>
            <p><strong>Test Accuracy:</strong> {{ modelSpecs.testAccuracy }}%</p>
            <p><strong>Time Spent to Train:</strong> {{ modelSpecs.trainingTime }}s</p>
          </div>
        </div>

        <div v-if="selectedModel === 'fneural-network'">
          <div v-if="model.modelTrained" class="model-specs">
            <h3>Model Specifications</h3>
            <p><strong>Avg Loss:</strong> {{ modelSpecs.avgLoss }}</p>
            <p><strong>Training Accuracy:</strong> {{ modelSpecs.trainAccuracy }}%</p>
            <p><strong>Test Accuracy:</strong> {{ modelSpecs.testAccuracy }}%</p>
            <p><strong>Time Spent to Train:</strong> {{ modelSpecs.trainingTime }}s</p>
          </div>
        </div>

        <div v-if="selectedModel === 'logistic-regression'">
          <div v-if="model.modelTrained" class="model-specs">
            <h3>Model Specifications</h3>
            <p><strong>Avg Loss:</strong> {{ modelSpecs.avgLoss }}</p>
            <p><strong>Training Accuracy:</strong> {{ modelSpecs.trainAccuracy }}%</p>
            <p><strong>Test Accuracy:</strong> {{ modelSpecs.testAccuracy }}%</p>
            <p><strong>Time Spent to Train:</strong> {{ modelSpecs.trainingTime }}s</p>
          </div>
        </div>

        <div class="model-prediction">
          <h3>Model Predictions</h3>
          <p>The Model predicts the handwritten number to have a value of</p>
          <div class="predicted-value">
            <strong>{{ modelPrediction ? modelPrediction : "?" }}</strong>
          </div>
        </div>
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
      cnn_model: {
        modelTrained: false,
        modelSpecs: {
          avgLoss: 0,
          trainAccuracy: 0,
          testAccuracy: 0,
          trainingTime: 0,
        },
        modelPrediction: null,
      },
      fnn_model: {
        modelTrained: false,
        modelSpecs: {
          avgLoss: 0,
          trainAccuracy: 0,
          testAccuracy: 0,
          trainingTime: 0,
        },
        modelPrediction: null,
      },
      lr_model: {
        modelTrained: false,
        modelSpecs: {
          avgLoss: 0,
          trainAccuracy: 0,
          testAccuracy: 0,
          trainingTime: 0,
        },
        modelPrediction: null,
      },
    };
  },
  methods: {
    identify() {
      // Placeholder for identify function (no logic)
      alert(`Identifying with ${this.selectedModel}`);

      // Simulate prediction process for testing
      this.modelPrediction = Math.floor(Math.random() * 10);
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
    }
  }
};
</script>
