<template>
  <div id="canvas-container">
    <div id="canvas-box">
      <div id="image-guideline"></div>
      <canvas id="drawingCanvas"></canvas>
    </div>
    <div id="buttons">
      <button>Clear</button>
      <button>Pen</button>
      <input id="colorpicker" type="color">
      <button>Eraser</button>
    </div>
  </div>
</template>

<script>
  // Setup Canvas and Graphics Content
  let cnv = document.getElementById("myCanvas");
  let ctx = cnv.getContext("2d");

  // Global Variables
  let mouseIsPressed = false;
  let mouseX, mouseY, pmouseX, pmouseY;
  let penSize = 55; // Porportionate to canvas size for MNIST Dataset
  let penColor = "black"; // Default pen color
  let eraserSize = 75; 
  let eraserColor = "white"; // Default eraser color

  



</script>

<style scoped>
#canvas-container {
  display: flex;
  flex-direction: column;
  align-items: center;
}

#canvas-box {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}

#image-guideline {
  position: fixed; /* Overlaps with canvas */
  width: 280px;
  height: 420px;
  border-radius: 50%;
  border: dotted 2px rgba(128, 128, 128, 0.5);

}

canvas {
  width: 560px ;
  height: 560px;
  border: 2px solid black;
  background-color: white;
  /* cursor: url('pen-cursor.png'), auto; Replace 'pen-cursor.png' with your cursor image */
}

#buttons {
  display: flex;
  justify-content: center;
  margin: 10px;
  gap: 1rem;
}
</style>