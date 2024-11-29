<template>
  <div id="canvas-container">
    <div id="canvas-box">
      <div id="image-guideline"></div>
      <canvas id="drawingCanvas"></canvas>
    </div>
    <div id="buttons">
      <button id="clearbtn">Clear</button>
      <button id="penbtn">Pen</button>
      <input  id="colorpicker" type="color">
      <button id="eraserbtn">Eraser</button>
    </div>
  </div>
</template>

<script>
import { is } from 'core-js/core/object';

export default {
  mounted() {
    // Setup Canvas and Graphics Context
    let cnv = document.getElementById("drawingCanvas");
    let ctx = cnv.getContext("2d");

    // Global Variables
    let mouseIsPressed = false;
    let mouseX, mouseY, pmouseX, pmouseY;
    let isPen = true;
    let penSize = 5; // change to 55
    let penColor = "black"; 

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

    // Button Events
    document.querySelector("#clearbtn").addEventListener("click", clearCanvas);
    document.querySelector("#penbtn").addEventListener("click", changeToPen);
    document.querySelector("#colorpicker").addEventListener("input", changeColor);
    document.querySelector("#eraserbtn").addEventListener("click", changeToEraser);

    // Mouse Event Functions
    function mousedownHandler() {
      mouseIsPressed = true;
    }
    function mouseupHandler() {
      mouseIsPressed = false;
    }
    function mousemoveHandler(event) {
      pmouseX = mouseX;
      pmouseY = mouseY;

      let cnvRect = cnv.getBoundingClientRect()
      mouseX = event.x - cnvRect.x;
      mouseY = event.y - cnvRect.y;
    }

    // Button Event Functions
    function clearCanvas() {
      ctx.clearRect(0, 0, cnv.width, cnv.height);
    }

    function changeToPen() {
      isPen = true;
      penColor = document.querySelector("#colorpicker").value;
    }

    function changeColor() {
      penColor = document.querySelector("#colorpicker").value;
    }

    function changeToEraser() {
      isPen = false;
      penColor = "white";
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