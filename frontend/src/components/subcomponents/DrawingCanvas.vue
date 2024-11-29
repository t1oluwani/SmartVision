<template>
  <div id="canvas-container">
    <div id="canvas-box">
      <div id="image-guideline"></div>
      <canvas id="drawingCanvas"></canvas>
      <!-- Brush stroke should be  -->
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
// export default {
//   mounted() {
//     const canvas = document.getElementById('drawingCanvas');
//     const ctx = canvas.getContext('2d');
//     let isDrawing = false;
//     let lastX = 0;
//     let lastY = 0;

//     canvas.addEventListener('mousedown', (e) => {
//       isDrawing = true;
//       [lastX, lastY] = [e.offsetX, e.offsetY];
//     });

//     canvas.addEventListener('mousemove', (e) => {
//       if (!isDrawing) return;
//       ctx.beginPath();
//       ctx.moveTo(lastX, lastY);
//       ctx.lineTo(e.offsetX, e.offsetY);
//       ctx.stroke();
//       [lastX, lastY] = [e.offsetX, e.offsetY];
//     });

//     canvas.addEventListener('mouseup', () => {
//       isDrawing = false;
//     });

//     canvas.addEventListener('mouseout', () => {
//       isDrawing = false;
//     });
//   },
// };
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