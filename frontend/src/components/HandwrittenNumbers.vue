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

          <p v-if="selectedModel === 'CNN'">
            Convolutional Neural Networks (CNNs) are a type of deep learning computer models designed to work with images. 
            They are great at recognizing objects, identifying faces, video analysis and Image segmentation.<br><br> </p>
          <p v-if="selectedModel === 'CNN'">
            <strong>Pros:</strong>
          <ul>
            <li>Good at detecting complex patterns.</li>
            <li>Handles high-dimensional input well.</li>
            <li>Excellent for images and spatial data.</li>
          </ul>
          <strong>Cons:</strong>
          <ul>
            <li>Requires strong computation.</li>
            <li>Longer training than simpler models.</li>
            <li>Potential overfitting without proper regularization.</li>
          </ul>
          </p>

          <p v-if="selectedModel === 'FNN'">
            Feedforward Neural Networks (FNNs) are a type of deep learning models that have fully connected layers. They are flexible 
            and often used for tasks involving organized data, like spreadsheets, where relationships between features are prioritized.<br><br></p>
          <p v-if="selectedModel === 'FNN'">
            <strong>Pros:</strong>
          <ul>
            <li>Simpler architecture, trains quick.</li>
            <li>Versatile & suitable for various tasks.</li>
            <li>Good for patterns and structured data</li>
          </ul>
          <strong>Cons:</strong>
          <ul>
            <li>Bad for images/unstructured data.</li>
            <li>Efficiiency suffers on complex datasets.</li>
            <li>Requires feature engineering for optimal results.</li>
          </ul>
          </p>

          <p v-if="selectedModel === 'LR'">
            Logistic Regression is a simple model used for tasks where outcomes only have two choices (eg. yes/no). It works by 
            analyzing the relationship between input data and the outcome, predicting the likelihood of each possibility.<br><br></p>
          <p v-if="selectedModel === 'LR'">
            <strong>Pros:</strong>
          <ul>
            <li>Efficient and quick to train.</li>
            <li>Easy to interpret and implement.</li>
            <li>Great for small/medium datasets.</li>
          </ul>
          <strong>Cons:</strong>
          <ul>
            <li>Limited to linear boundaries.</li>
            <li>Does not handle non-linear or complex patterns well.</li>
            <li>Sensitive to outliers in the data.</li>
          </ul>
          </p>
        </div>
      </div>

      <div class="middle-container">
        <DrawingCanvas ref="drawingCanvas" />

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

      <div v-if="loadingState" class="loading-overlay">
        <div class="loading-content">
          <div class="loader"></div>
          <div class="loading-text">
            <p v-if="selectedModel === 'CNN'">Training Convolutional Neural Network Model...</p>
            <p v-if="selectedModel === 'CNN'">(Estimated time to train: 5-6 minutes)</p>

            <p v-if="selectedModel === 'FNN'">Training Feedforward Neural Network Model...</p>
            <p v-if="selectedModel === 'FNN'">(Estimated time to train: 2-3 minutes)</p>

            <p v-if="selectedModel === 'LR'">Training Logistic Regression Model...</p>
            <p v-if="selectedModel === 'LR'">(Estimated time to train: 3-4 minutes)</p>
          </div>
        </div>
      </div>

    </div>
  </div>
</template>

<script>
import FormData from 'form-data';
import DrawingCanvas from './subcomponents/DrawingCanvas.vue';
import { modelTrain, modelClear, modelPredict } from '../services/MLModelClient';

export default {
  components: {
    DrawingCanvas,
  },

  data() {
    return {
      loadingState: false,
      selectedModel: 'LR', // Default model
      lr_model: {
        modelTrained: true,
        modelSpecs: {
          averageLoss: 0.0016,
          trainAccuracy: 92.78,
          testAccuracy: 92.94,
          trainingTime: 70.1,
        },
        modelPrediction: null,
      },
      fnn_model: {
        modelTrained: true,
        modelSpecs: {
          averageLoss: 0.0001,
          trainAccuracy: 96.76,
          testAccuracy: 97.22,
          trainingTime: 69.4,
        },
        modelPrediction: null,
      },
      cnn_model: {
        modelTrained: true,
        modelSpecs: {
          averageLoss: 0.0021,
          trainAccuracy: 98.82,
          testAccuracy: 98.85,
          trainingTime: 116.2,
        },
        modelPrediction: null,
      },
    };
  },

  methods: {
    async identify() {
      const modelTrained = this.modelIsTrained();
      if (modelTrained) {

        // Get the image data from the canvas
        const imgData = new FormData();
        const imgAsBlob = await this.$refs.drawingCanvas.getCanvasImageAsBlob();
        imgData.append("canvas_drawing", imgAsBlob);

        // Make a prediction request to the model
        const predictionResponse = await modelPredict(this.selectedModel, imgData);

        // Update model prediction
        switch (this.selectedModel) {
          case 'CNN':
            this.cnn_model.modelPrediction = predictionResponse.predicted_class;
            break;
          case 'FNN':
            this.fnn_model.modelPrediction = predictionResponse.predicted_class;
            break;
          default: // LR
            this.lr_model.modelPrediction = predictionResponse.predicted_class;
            break;
        }
      } else {
        alert("Please train the model first before making a prediction.");
      }
    },

    async trainModel() {
      this.loadingState = true;
      try {
        // Make a training request to the model
        const modelResponse = await modelTrain(this.selectedModel);
        console.log(modelResponse);

        // Get model specs from training response
        const modelSpecs = {
          trainAccuracy: modelResponse.train_accuracy,
          testAccuracy: modelResponse.test_accuracy,
          averageLoss: modelResponse.average_loss,
          trainingTime: modelResponse.run_time,
        }
        console.log(modelSpecs);

        // Update model status and specs
        switch (this.selectedModel) {
          case 'CNN':
            this.cnn_model.modelTrained = true;
            this.cnn_model.modelSpecs = modelSpecs;
            break;
          case 'FNN':
            this.fnn_model.modelTrained = true;
            this.fnn_model.modelSpecs = modelSpecs;
            break;
          default: // LR
            this.lr_model.modelTrained = true;
            this.lr_model.modelSpecs = modelSpecs;
            break;
        }
      } finally {
        this.loadingState = false;
      }
    },

    async clearModel() {
      // Display a confirmation dialog
      const userConfirmed = confirm(
        "READ, IMPORTANT!\n\nThe free tier of deployment services do not provide enough memory to train datasets, so if this is the live site, clearing it would mean you would not be able to retrain the model. However, if you have set the applications up locally then you are able to retrain each model however many times you want.\n\nDo you still want to clear the model?"
      );

      if (!userConfirmed) {
        // User chose not to clear the model
        console.log("Model Not Cleared!")
        return;
      } else {
        console.log("Model Cleared!")
      }

      // Make a request to clear the model
      const clearStatus = await modelClear(this.selectedModel);
      console.log(clearStatus);

      // Update model status to not trained
      if (clearStatus == 200) {
        switch (this.selectedModel) {
          case 'CNN':
            this.cnn_model.modelTrained = false;
            break;
          case 'FNN':
            this.fnn_model.modelTrained = false;
            break;
          default: // LR
            this.lr_model.modelTrained = false;
            break;
        }
      }
    },

    modelIsTrained() {
      switch (this.selectedModel) {
        case 'CNN':
          return this.cnn_model.modelTrained;
        case 'FNN':
          return this.fnn_model.modelTrained;
        default: // LR
          return this.lr_model.modelTrained;
      }
    }
  }
};
</script>
