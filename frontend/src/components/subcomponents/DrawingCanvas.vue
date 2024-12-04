<template>
  <div id="canvas-container">
    <div id="canvas-box">
      <div id="image-guideline"></div>
      <canvas id="drawingCanvas"></canvas>
    </div>
    <div id="buttons">
      <button id="clearbtn">Clear</button>
      <button id="eraserbtn">Eraser</button>
      <button id="penbtn">Pen</button>
      <input id="colorpicker" type="color">
    </div>
  </div>
</template>

<script>
export default {
  methods: {
    getCanvasImageAsBlob() {
      let cnv = document.getElementById("drawingCanvas");
      return new Promise((resolve) => {
        cnv.toBlob(blob => {
            resolve(blob);  
        });
      });
    }

  },
  mounted() {
    // Setup Canvas and Graphics Context
    let cnv = document.getElementById("drawingCanvas");
    cnv.width = 560;
    cnv.height = 560;
    let ctx = cnv.getContext("2d");
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, cnv.width, cnv.height);

    // Global Variables
    let mouseIsPressed = false;
    let mouseX, mouseY, pmouseX, pmouseY;
    // let isPen = true;
    let penSize = 50;
    let penColor = "black";

    // Main Program Loop (60 FPS)
    requestAnimationFrame(loop);

    function loop() {
      // Draw only when mouse is pressed and inside the canvas
      if (mouseIsPressed && mouseInCanvas()) {
        // Combining them leads to a smoother streak (don't ask me why it works, it just does :D)
        drawViaLine();
        drawViaCircle();
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
    document.querySelector("#eraserbtn").addEventListener("click", changeToEraser);
    document.querySelector("#colorpicker").addEventListener("input", changeColor);
    // document.querySelector("#identifybtn").addEventListener("click", saveCanvas);

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
      mouseX = event.clientX - cnvRect.left;
      mouseY = event.clientY - cnvRect.top;
    }
    function mouseInCanvas() {
      return mouseX >= 0 &&
        mouseX <= cnv.width &&
        mouseY >= 0 &&
        mouseY <= cnv.height;
    }

    // Button Event Functions
    function clearCanvas() {
      ctx.clearRect(0, 0, cnv.width, cnv.height);
    }
    function changeToPen() {
      // isPen = true;
      penColor = document.querySelector("#colorpicker").value;
    }
    function changeToEraser() {
      // isPen = false;
      penSize = 75;
      penColor = "white";
    }
    function changeColor() {
      penColor = document.querySelector("#colorpicker").value;
    }

    // Drawing Functions
    function drawViaLine() {
      ctx.strokeStyle = penColor;
      ctx.lineWidth = penSize;
      ctx.beginPath();
      ctx.moveTo(pmouseX, pmouseY);
      ctx.lineTo(mouseX, mouseY);
      ctx.stroke();
    }
    function drawViaCircle() {
      ctx.fillStyle = penColor;
      ctx.beginPath();
      ctx.arc(mouseX, mouseY, penSize / 2, 0, Math.PI * 2);
      ctx.fill();
    }

    // TODO: Implement pen/eraser toggle functionality
    // When isPen is true, clicker is a pen; when isPen is false, clicker is an eraser

    // Save Canvas as Image
    // function saveCanvas() {
    //   let link = document.createElement("a");
    //   link.download = "canvas_drawing.png";
    //   link.href = cnv.toDataURL("image/png").replace("image/png", "image/octet-stream");
    //   // link.click();
    // }
  }
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
  position: absolute;
  /* Overlaps with canvas */
  width: 280px;
  height: 420px;
  border-radius: 50%;
  border: dashed 6px rgba(128, 128, 128, 0.5);
}

canvas {
  width: 560px;
  height: 560px;
  border: 4px solid black;
  background-color: white;
  /* cursor: url('pen-cursor.png'), auto; Replace 'pen-cursor.png' with your cursor image */
}

#buttons {
  display: flex;
  justify-content: center;
  align-items: center;
  margin: 10px;
  gap: 2rem;
  border-radius: 5px;
  background-color: gray;
}

button,
#colorpicker {
  width: 100px;
  height: 40px;
  font-size: 18px;
  font-family: 'Trebuchet MS', Arial, sans-serif;
  border: none;
  border-radius: 5px;
  padding: 6px;
  margin: 0;
  background-color: transparent;
  cursor: pointer;
}

button:hover,
#colorpicker:hover {
  background-color: darkgray;
}

button:active,
#colorpicker:active {
  background-color: lightgray;
}

#savebtn {
  background-color: aquamarine;
}
</style>