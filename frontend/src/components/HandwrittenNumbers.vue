<template>
  <div class="HN-Scope">
    <div class="tab-content">

      <div class="left-container">
        <div class="model-info">
          <h3>{{
            selectedModel === 'LR' ? 'Logistic Regression' :
              selectedModel === 'FNN' ? 'FeedForward Neural Network' :
                'Convolutional Neural Network'
          }}</h3>

          <p v-if="selectedModel === 'LR'">
            Logistic Regression is a linear model used for binary classification tasks.
            Blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah
            blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah
            blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah
            blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah
            blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah
          </p>
          <p v-if="selectedModel === 'FNN'">
            Feedforward Neural Networks (FNN) are deep learning models with fully connected layers for classification.
            Blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah
            blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah
            blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah
            blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah
            blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah
          </p>
          <p v-if="selectedModel === 'CNN'">
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
            <option value="LR">Logistic Regression</option>
            <option value="FNN">FeedFoward Neural Network (FNN)</option>
            <option value="CNN">Convolutional Neural Network (CNN)</option>
          </select>

          <button id="identifybtn" @click="identify">Identify</button>
        </div>
      </div>

      <div class="right-container">
        <div class="training-status">
          <h3>Visual Recognition Model Status</h3>
          <p v-if="selectedModel === 'CNN'">{{ cnn_model.modelTrained ? 'CNN Model is trained' : 'No CNN Model is trained' }}</p>
          <p v-if="selectedModel === 'FNN'">{{ fnn_model.modelTrained ? 'FNN Model is trained' : 'No FNN Model is trained' }}</p>
          <p v-if="selectedModel === 'LR'">{{ lr_model.modelTrained ? 'LR Model is trained' : 'No LR Model is trained' }}</p>

          <button v-if="!cnn_model.modelTrained && selectedModel === 'CNN'" @click="trainModel">Train CNN Model</button>
          <button v-if="cnn_model.modelTrained && selectedModel === 'CNN'" @click="clearModel">Clear CNN Model</button>
          <button v-if="!fnn_model.modelTrained && selectedModel === 'FNN'" @click="trainModel">Train FNN Model</button>
          <button v-if="fnn_model.modelTrained && selectedModel === 'FNN'" @click="clearModel">Clear FNN Model</button>
          <button v-if="!lr_model.modelTrained && selectedModel === 'LR'" @click="trainModel">Train LR Model</button>
          <button v-if="lr_model.modelTrained && selectedModel === 'LR'" @click="clearModel">Clear LR Model</button>
        </div>


        <div v-if="selectedModel === 'CNN'">
          <div v-if="cnn_model.modelTrained" class="model-specs">
            <h3>Model Specifications</h3>
            <p><strong>Avg Loss:</strong> {{ cnn_model.modelSpecs.avgLoss }}</p>
            <p><strong>Training Accuracy:</strong> {{ cnn_model.modelSpecs.trainAccuracy }}%</p>
            <p><strong>Test Accuracy:</strong> {{ cnn_model.modelSpecs.testAccuracy }}%</p>
            <p><strong>Time Spent to Train:</strong> {{ cnn_model.modelSpecs.trainingTime }}s</p>
          </div>
        </div>

        <div v-if="selectedModel === 'FNN'">
          <div v-if="fnn_model.modelTrained" class="model-specs">
            <h3>Model Specifications</h3>
            <p><strong>Avg Loss:</strong> {{ fnn_model.modelSpecs.avgLoss }}</p>
            <p><strong>Training Accuracy:</strong> {{ fnn_model.modelSpecs.trainAccuracy }}%</p>
            <p><strong>Test Accuracy:</strong> {{ fnn_model.modelSpecs.testAccuracy }}%</p>
            <p><strong>Time Spent to Train:</strong> {{ fnn_model.modelSpecs.trainingTime }}s</p>
          </div>
        </div>

        <div v-if="selectedModel === 'LR'">
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
          <div v-if="selectedModel === 'CNN'" class="predicted-value">
            <strong>{{ cnn_model.modelPrediction ? cnn_model.modelPrediction : "?" }}</strong>
          </div>
          <div v-if="selectedModel === 'FNN'" class="predicted-value">
            <strong>{{ fnn_model.modelPrediction ? fnn_model.modelPrediction : "?" }}</strong>
          </div>
          <div v-if="selectedModel === 'LR'" class="predicted-value">
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
      selectedModel: 'LR', // Default model
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
      if (this.selectedModel === 'CNN') {
        this.cnn_model.modelPrediction = Math.floor(Math.random() * 10);
      } else if (this.selectedModel === 'FNN') {
        this.fnn_model.modelPrediction = Math.floor(Math.random() * 10);
      } else {
        this.lr_model.modelPrediction = Math.floor(Math.random() * 10);
      }
    },
    trainModel() {
      // Placeholder for trainModel function (no logic)
      alert('Training model');

      // Simulate training process for testing
      if (this.selectedModel === 'CNN') {
        this.cnn_model.modelTrained = true;
        this.cnn_model.modelSpecs = {
          avgLoss: 0.3,
          trainAccuracy: 92,
          testAccuracy: 88,
          trainingTime: 120, // in seconds
        }
      } else if (this.selectedModel === 'FNN') {
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
    },
    clearModel() {
      // Placeholder for clearModel function (no logic)
      alert('Clearing model');

      // Simulate clearing process for testing
      if (this.selectedModel === 'CNN') {
        this.cnn_model.modelTrained = false;
      } else if (this.selectedModel === 'FNN') {
        this.fnn_model.modelTrained = false;
      } else {
        this.lr_model.modelTrained = false;
      }
    },
  }
};
</script>
