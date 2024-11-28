<template>
  <div id="app">
    <!-- <div class="tabs">
      <button @click="showTab('numbers')">Handwritten Numbers</button>
      <button @click="showTab('object')">Object Recognition</button>
    </div>

    <div v-if="activeTab === 'numbers'">
      <HandwrittenNumbers />
    </div>
    <div v-if="activeTab === 'object'">
      <ObjectRecognition />
    </div> -->
    <DrawingCanvas />
  </div>
</template>

<script>
import './assets/styles/App.css';
import './assets/styles/HandwrittenNumbers.css';
import './assets/styles/ObjectRecognition.css';

// import HandwrittenNumbers from './components/HandwrittenNumbers.vue';
// import ObjectRecognition from './components/ObjectRecognition.vue';
import DrawingCanvas from './components/subcomponents/DrawingCanvas.vue';

export default {
  components: {
    // HandwrittenNumbers,
    // ObjectRecognition,
    DrawingCanvas,
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
