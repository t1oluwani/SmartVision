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
export default {
  mounted() {
    // Setup Canvas and Graphics Context
    let cnv = document.getElementById("drawingCanvas");
    if (!cnv) {
      console.error("Canvas element not found!");
      return;
    }
    let ctx = cnv.getContext("2d");

    // Global Variables
    let mouseIsPressed = false;
    let mouseX, mouseY, pmouseX, pmouseY;
    let penSize = 55; // Proportionate to canvas size for MNIST Dataset
    let penColor = "black"; // Default pen color

    // Main Program Loop (60 FPS)
    requestAnimationFrame(loop);

    function loop() {
      // Draw a circle if mouseIsPressed
      if (mouseIsPressed) {
        ctx.strokeStyle = penColor;
        ctx.lineWidth = penSize;
        ctx.beginPath();
        ctx.moveTo(pmouseX, pmouseY);
        ctx.lineTo(mouseX, mouseY);
        ctx.stroke();
      }
      requestAnimationFrame(loop);
    }

    // Mouse Events
    document.addEventListener("mousedown", mousedownHandler);
    document.addEventListener("mouseup", mouseupHandler);
    document.addEventListener("mousemove", mousemoveHandler);

    function mousedownHandler() {
      mouseIsPressed = true;
    }

    function mouseupHandler() {
      mouseIsPressed = false;
    }

    function mousemoveHandler(event) {
      // Save previous mouseX and mouseY
      pmouseX = mouseX;
      pmouseY = mouseY;

      // Update mouseX and mouseY
      let cnvRect = cnv.getBoundingClientRect();
      mouseX = event.clientX - cnvRect.left;
      mouseY = event.clientY - cnvRect.top;
    }
  },
};
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