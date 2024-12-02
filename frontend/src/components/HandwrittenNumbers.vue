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
          <div v-if="cnn_model.modelTrained" class="model-specs">
            <h3>Model Specifications</h3>
            <p><strong>Avg Loss:</strong> {{ cnn_model.modelSpecs.avgLoss }}</p>
            <p><strong>Training Accuracy:</strong> {{ cnn_model.modelSpecs.trainAccuracy }}%</p>
            <p><strong>Test Accuracy:</strong> {{ cnn_model.modelSpecs.testAccuracy }}%</p>
            <p><strong>Time Spent to Train:</strong> {{ cnn_model.modelSpecs.trainingTime }}s</p>
          </div>
        </div>

        <div v-if="selectedModel === 'fneural-network'">
          <div v-if="fnn_model.modelTrained" class="model-specs">
            <h3>Model Specifications</h3>
            <p><strong>Avg Loss:</strong> {{ fnn_model.modelSpecs.avgLoss }}</p>
            <p><strong>Training Accuracy:</strong> {{ fnn_model.modelSpecs.trainAccuracy }}%</p>
            <p><strong>Test Accuracy:</strong> {{ fnn_model.modelSpecs.testAccuracy }}%</p>
            <p><strong>Time Spent to Train:</strong> {{ fnn_model.modelSpecs.trainingTime }}s</p>
          </div>
        </div>

        <div v-if="selectedModel === 'logistic-regression'">
          <div v-if="lr_model.modelTrained" class="model-specs">
            <h3>Model Specifications</h3>
            <p><strong>Avg Loss:</strong> {{ lr_model.modelSpecs.avgLoss }}</p>
            <p><strong>Training Accuracy:</strong> {{ lr_model.modelSpecs.trainAccuracy }}%</p>
            <p><strong>Test Accuracy:</strong> {{ lr_model.modelSpecs.testAccuracy }}%</p>
            <p><strong>Time Spent to Train:</strong> {{ lr_model.modelSpecs.trainingTime }}s</p>
          </div>
        </div>

        <div class="model-prediction">
          <h3>Model Predictions</h3>
          <p>The Model predicts the handwritten number to have a value of</p>
          <div v-if="selectedModel === 'cneural-network'" class="predicted-value">
            <strong>{{ cnn_model.modelPrediction ? cnn_model.modelPrediction : "?" }}</strong>
          </div>
          <div v-if="selectedModel === 'fneural-network'" class="predicted-value">
            <strong>{{ fnn_model.modelPrediction ? fnn_model.modelPrediction : "?" }}</strong>
          </div>
          <div v-if="selectedModel === 'logistic-regression'" class="predicted-value">
            <strong>{{ lr_model.modelPrediction ? lr_model.modelPrediction : "?" }}</strong>
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
      if (this.selectedModel === 'cneural-network') {
        this.cnn_model.modelPrediction = Math.floor(Math.random() * 10);
      } else if (this.selectedModel === 'fneural-network') {
        this.fnn_model.modelPrediction = Math.floor(Math.random() * 10);
      } else {
        this.lr_model.modelPrediction = Math.floor(Math.random() * 10);
      }
    },
    trainModel() {
      // Placeholder for trainModel function (no logic)
      alert('Training model');

      // Simulate training process for testing
      if (this.selectedModel === 'cneural-network') {
        this.cnn_model.modelTrained = true;
        this.cnn_model.modelSpecs = {
          avgLoss: 0.3,
          trainAccuracy: 92,
          testAccuracy: 88,
          trainingTime: 120, // in seconds
        }
      } else if (this.selectedModel === 'fneural-network') {
        this.fnn_model.modelTrained = true;
        this.fnn_model.modelSpecs = {
          avgLoss: 0.2,
          trainAccuracy: 94,
          testAccuracy: 90,
          trainingTime: 180, // in seconds
        }
      } else {
        this.lr_model.modelTrained = true;
        this.lr_model.modelSpecs = {
          avgLoss: 0.1,
          trainAccuracy: 96,
          testAccuracy: 92,
          trainingTime: 60, // in seconds
        }
      }
    }
  }
};
</script>
