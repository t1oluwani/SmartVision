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
        <DrawingCanvas ref="drawingCanvas"/>

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
          <p v-if="selectedModel === 'CNN'">{{ cnn_model.modelTrained ? 'CNN Model' : 'No Model is trained' }}</p>
          <p v-if="selectedModel === 'FNN'">{{ fnn_model.modelTrained ? 'FNN Model' : 'No Model is trained' }}</p>
          <p v-if="selectedModel === 'LR'">{{ lr_model.modelTrained ? 'LR Model' : 'No Model is trained' }}</p>

          <button v-if="!cnn_model.modelTrained && selectedModel === 'CNN'" @click="trainModel">Train Convolutional-NN
            Model</button>
          <button v-if="cnn_model.modelTrained && selectedModel === 'CNN'" @click="clearModel">Clear Model</button>
          <button v-if="!fnn_model.modelTrained && selectedModel === 'FNN'" @click="trainModel">Train Feedfoward-NN
            Model</button>
          <button v-if="fnn_model.modelTrained && selectedModel === 'FNN'" @click="clearModel">Clear Model</button>
          <button v-if="!lr_model.modelTrained && selectedModel === 'LR'" @click="trainModel">Train Logistic-Regression
            Model</button>
          <button v-if="lr_model.modelTrained && selectedModel === 'LR'" @click="clearModel">Clear Model</button>
        </div>

        <div v-if="selectedModel === 'CNN'">
          <div v-if="cnn_model.modelTrained" class="model-specs">
            <h3>Model Specifications</h3>
            <p><strong>Average Loss:</strong> {{ cnn_model.modelSpecs.averageLoss }}</p>
            <p><strong>Training Accuracy:</strong> {{ cnn_model.modelSpecs.trainAccuracy }}%</p>
            <p><strong>Test Accuracy:</strong> {{ cnn_model.modelSpecs.testAccuracy }}%</p>
            <p><strong>Training and Testing Time:</strong> {{ cnn_model.modelSpecs.trainingTime }}s</p>
          </div>
        </div>

        <div v-if="selectedModel === 'FNN'">
          <div v-if="fnn_model.modelTrained" class="model-specs">
            <h3>Model Specifications</h3>
            <p><strong>Average Loss:</strong> {{ fnn_model.modelSpecs.averageLoss }}</p>
            <p><strong>Training Accuracy:</strong> {{ fnn_model.modelSpecs.trainAccuracy }}%</p>
            <p><strong>Test Accuracy:</strong> {{ fnn_model.modelSpecs.testAccuracy }}%</p>
            <p><strong>Training and Testing Time:</strong> {{ fnn_model.modelSpecs.trainingTime }}s</p>
          </div>
        </div>

        <div v-if="selectedModel === 'LR'">
          <div v-if="lr_model.modelTrained" class="model-specs">
            <h3>Model Specifications</h3>
            <p><strong>Average Loss:</strong> {{ lr_model.modelSpecs.averageLoss }}</p>
            <p><strong>Training Accuracy:</strong> {{ lr_model.modelSpecs.trainAccuracy }}%</p>
            <p><strong>Test Accuracy:</strong> {{ lr_model.modelSpecs.testAccuracy }}%</p>
            <p><strong>Training and Testing Time:</strong> {{ lr_model.modelSpecs.trainingTime }}s</p>
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
import FormData from 'form-data';
import DrawingCanvas from './subcomponents/DrawingCanvas.vue';
import { modelTrain, modelClear, modelPredict } from '@/service/MLModelClient';

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
          averageLoss: 0,
          trainAccuracy: 0,
          testAccuracy: 0,
          trainingTime: 0,
        },
        modelPrediction: null,
      },
      fnn_model: {
        modelTrained: false,
        modelSpecs: {
          averageLoss: 0,
          trainAccuracy: 0,
          testAccuracy: 0,
          trainingTime: 0,
        },
        modelPrediction: null,
      },
      lr_model: {
        modelTrained: false,
        modelSpecs: {
          averageLoss: 0,
          trainAccuracy: 0,
          testAccuracy: 0,
          trainingTime: 0,
        },
        modelPrediction: null,
      },
    };
  },
  methods: {
    async identify() {
      // Placeholder for identify function (no logic)
      alert(`Identifying with ${this.selectedModel}`);

      // if (this.selectedModel.modelTrained) {
      //   this.modelPredict();
      // }

      // Get the image data from the canvas
      const imgData = new FormData();
      const imgAsBlob = await this.$refs.drawingCanvas.getCanvasImageAsBlob();
      imgData.append("canvas_drawing", imgAsBlob);

      const predictionResponse = await modelPredict(this.selectedModel, imgData);
      console.log("Prediction Response:", predictionResponse);
      
      // Add Loading State and Spinner here

      // Update model prediction
      if (this.selectedModel === 'CNN') {
        this.cnn_model.modelPrediction = predictionResponse.predicted_class;
      } else if (this.selectedModel === 'FNN') {
        this.fnn_model.modelPrediction = predictionResponse.predicted_class;
      } else {
        this.lr_model.modelPrediction = predictionResponse.predicted_class;
      }
    },

    async trainModel() {
      let modelResponse = await modelTrain(this.selectedModel);
      console.log(modelResponse);

      // Add Loading State and Spinner here (take about 5-6 mins for CNN, 2-3 mins for FNN, and 3-4 mins for LR)

      let modelSpecs = {
        trainAccuracy: modelResponse.train_accuracy,
        testAccuracy: modelResponse.test_accuracy,
        averageLoss: modelResponse.average_loss,
        trainingTime: modelResponse.run_time,
      }
      console.log(modelSpecs);

      // Update model status and specs
      if (this.selectedModel === 'CNN') {
        this.cnn_model.modelTrained = true;
        this.cnn_model.modelSpecs = modelSpecs;

      } else if (this.selectedModel === 'FNN') {
        this.fnn_model.modelTrained = true;
        this.fnn_model.modelSpecs = modelSpecs;

      } else {
        this.lr_model.modelTrained = true;
        this.lr_model.modelSpecs = modelSpecs;
      }
    },

    async clearModel() {
      let clearStatus = await modelClear(this.selectedModel);
      console.log(clearStatus);

      // Add Loading State and Spinner here

      if (clearStatus == 200) {
        if (this.selectedModel === 'CNN') {
          this.cnn_model.modelTrained = false;
        } else if (this.selectedModel === 'FNN') {
          this.fnn_model.modelTrained = false;
        } else {
          this.lr_model.modelTrained = false;
        }
      }
    },
  }
};
</script>
