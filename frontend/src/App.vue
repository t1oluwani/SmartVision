<template>
  <div id="app">
      <img id="logo" @click="showTab('default')" src="smartvision_logo.png" alt="SmartVision Logo" />

    <div id="tabs">
      <button @click="showTab('numbers')">Handwritten Numbers</button>
      <button @click="showTab('object')">Object Recognition</button>
    </div>

    <div id="default" v-if="activeTab === 'default'">
      <h1>Welcome to SmartVision</h1>
      <h2>A Cutting-Edge Visual Recognition Playground</h2>
      <p>
        This site is a playground for visual recognition models. You can test
        different models on handwritten numbers and object recognition. This site
        will teach you about each model, what they are best used for and showcase
        their abilities in action through hands-on testing.
      </p>
      <p>
        The models are trained using the MNIST dataset for handwritten numbers and
        the FashionMNIST dataset for object recognition. The models are trained using
        PyTorch and are run in the browser.
      </p>
      <p>
        <strong>Click a tab above to get started!</strong>
      </p>
    </div>

    <div v-if="activeTab === 'numbers'">
      <HandwrittenNumbers />
    </div>
    <div v-if="activeTab === 'object'">
      <ObjectRecognition />
    </div>
  </div>
</template>

<script>

import './assets/styles/App.css';
import './assets/styles/HandwrittenNumbers.css';
import './assets/styles/ObjectRecognition.css';

// import SVLogo from ;

import HandwrittenNumbers from './components/HandwrittenNumbers.vue';
import ObjectRecognition from './components/ObjectRecognition.vue';

export default {
  components: {
    HandwrittenNumbers,
    ObjectRecognition,
  },
  data() {
    return {
      activeTab: 'default', // TODO: Set the default tab, make information about the site and features
    };
  },
  methods: {
    showTab(tab) {
      this.activeTab = tab;
    },
  },
};
</script>
