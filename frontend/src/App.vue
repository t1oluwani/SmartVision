<template>
  <div id="app">
    <div id="tabs">
      <button @click="showTab('numbers')">Handwritten Numbers</button>
      <button @click="showTab('object')">Object Recognition</button>
    </div>

    <div id="default">
      <h1>Welcome to SmartVision, a Visual Recognition Playground</h1>
      <p>
        This site is a playground for visual recognition models. You can test
        different models on handwritten numbers and object recognition. This site
        will teach you about each model, what they're best used for and give you the 
        ability to see their abilities first hand.
      </p>
      <p>
        Select a tab above to get started.
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
