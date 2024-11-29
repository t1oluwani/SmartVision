<template>
  <div id="canvas_container">
    <canvas id="drawingCanvas" width="560" height="560"></canvas>
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
#canvas_container {
  display: flex;
  flex-direction: column;
  align-items: center;
}
canvas {
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