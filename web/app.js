/**
 * Virtual Makeup Studio — Web In-Browser Real-Time Try-On
 * Natural skin-aware cosmetic blending, pore-diffused blush, evenly blended eyeliner,
 * seamless zero-edge hairline foundation, and velvet matte lips.
 */

// ---------------------------------------------------------------------------
// Preset Color Swatches
// ---------------------------------------------------------------------------
const SWATCHES = {
  foundation: [
    { name: "Fair Vanilla", hex: "#f5decc" },
    { name: "Light Ivory", hex: "#f0d2ba" },
    { name: "Medium Sand", hex: "#e2ba99" },
    { name: "Warm Honey", hex: "#d0a480" },
    { name: "Sweet Caramel", hex: "#b2845f" },
  ],
  blush: [
    { name: "Peach Blossom", hex: "#e8a090" },
    { name: "Dusty Rose", hex: "#e07a8a" },
    { name: "Sweet Rose", hex: "#d46a7a" },
    { name: "Warm Coral", hex: "#e09078" },
    { name: "Soft Peony", hex: "#e8b4b8" },
    { name: "Apricot Flush", hex: "#e49676" },
  ],
  eyeshadow: [
    { name: "Velvet Nude", hex: "#c4a484" },
    { name: "Soft Taupe", hex: "#8b7355" },
    { name: "Golden Bronze", hex: "#b87333" },
    { name: "Rose Gold", hex: "#b76e79" },
    { name: "Dusty Plum", hex: "#6b3a5d" },
    { name: "Cocoa Brown", hex: "#6b4423" },
  ],
  eyeliner: [
    { name: "Carbon Black", hex: "#121214" },
    { name: "Mocha Brown", hex: "#362620" },
    { name: "Midnight Blue", hex: "#182036" },
  ],
  brow: [
    { name: "Soft Brown", hex: "#4b342a" },
    { name: "Dark Chocolate", hex: "#30221c" },
    { name: "Blonde Taupe", hex: "#826c58" },
    { name: "Ebony", hex: "#1c1816" },
  ],
  lipstick: [
    { name: "Classic Red", hex: "#c41e3a" },
    { name: "Rose Petal", hex: "#c94c6d" },
    { name: "Soft Nude", hex: "#c08081" },
    { name: "Sweet Berry", hex: "#7b2d4e" },
    { name: "Peach Coral", hex: "#e07060" },
    { name: "Soft Mauve", hex: "#a45d7a" },
    { name: "Baby Pink", hex: "#e88aa4" },
  ],
};

// ---------------------------------------------------------------------------
// Landmark Index Constants
// ---------------------------------------------------------------------------
const OUTER_LIP = [
  61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
  291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
];

const INNER_LIP = [
  78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
  308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
];

const LEFT_UPPER_LID = [33, 246, 161, 160, 159, 158, 157, 173, 133];
const RIGHT_UPPER_LID = [263, 466, 388, 387, 386, 385, 384, 398, 362];

const LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
const RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398];

const LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46];
const RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276];

const LEFT_CHEEK = [345, 346, 347, 330, 266, 425, 411, 352];
const RIGHT_CHEEK = [116, 117, 118, 101, 36, 205, 187, 123];

const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
  379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
  234, 127, 162, 21, 54, 103, 67, 109,
];
const FOREHEAD_TOP = [103, 67, 109, 10, 338, 297, 332];

// ---------------------------------------------------------------------------
// App State (Defaults: Velvet Matte Lips, Diffused Blush, Soft Brows)
// ---------------------------------------------------------------------------
const state = {
  isCameraRunning: false,
  mirror: true,
  showOriginal: false,
  activeLandmarks: null,

  products: {
    foundation: { enabled: false, color: "#f0d2ba", intensity: 0.28 },
    blush: { enabled: true, color: "#e8a090", intensity: 0.28 },
    eyeshadow: { enabled: true, color: "#c4a484", intensity: 0.32 },
    eyeliner: { enabled: false, color: "#121214", intensity: 0.80, shape: "classic_wing" },
    brow: { enabled: true, color: "#4b342a", intensity: 0.25 },
    lipstick: { enabled: true, color: "#c41e3a", intensity: 0.60, blend: "multiply" },
  },
};

// ---------------------------------------------------------------------------
// DOM Elements
// ---------------------------------------------------------------------------
const canvas = document.getElementById("outputCanvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const video = document.getElementById("webcamVideo");
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingMsg = document.getElementById("loadingMsg");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const fpsBadge = document.getElementById("fpsBadge");

const btnCamera = document.getElementById("btnCamera");
const btnSave = document.getElementById("btnSave");
const chkMirror = document.getElementById("chkMirror");
const chkOriginal = document.getElementById("chkOriginal");
const btnResetAll = document.getElementById("btnResetAll");

let faceMesh = null;
let cameraInstance = null;
let lastFpsTime = performance.now();
let frameCounter = 0;

// ---------------------------------------------------------------------------
// High-DPI Responsive Canvas Setup
// ---------------------------------------------------------------------------
function setupHighDpiCanvas(width, height) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.aspectRatio = `${width} / ${height}`;
  ctx.resetTransform();
  ctx.scale(dpr, dpr);
}

// ---------------------------------------------------------------------------
// MediaPipe Initialization
// ---------------------------------------------------------------------------
async function initMediaPipe() {
  try {
    loadingMsg.textContent = "Loading FaceMesh model...";
    faceMesh = new FaceMesh({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
    });

    faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    faceMesh.onResults(onFaceMeshResults);

    await faceMesh.initialize();
    loadingOverlay.classList.add("hidden");
    statusText.textContent = "Camera ready";
    startCamera();
  } catch (err) {
    console.error("MediaPipe initialization error:", err);
    loadingMsg.textContent = "Failed to load model. Please check network connection.";
    statusDot.className = "status-indicator waiting";
    statusText.textContent = "Model load failed";
  }
}

// ---------------------------------------------------------------------------
// Camera & Video Feed
// ---------------------------------------------------------------------------
async function startCamera() {
  if (state.isCameraRunning) return;

  try {
    statusDot.className = "status-indicator waiting";
    statusText.textContent = "Connecting to webcam...";

    if (!cameraInstance) {
      cameraInstance = new Camera(video, {
        onFrame: async () => {
          if (faceMesh) {
            await faceMesh.send({ image: video });
          }
        },
        width: 1280,
        height: 720,
      });
    }

    await cameraInstance.start();
    state.isCameraRunning = true;
    statusDot.className = "status-indicator";
    statusText.textContent = "Camera active";
    btnCamera.classList.add("btn-primary");
    btnCamera.classList.remove("btn-secondary");
  } catch (err) {
    console.warn("Could not open camera:", err);
    statusDot.className = "status-indicator waiting";
    statusText.textContent = "Camera unavailable";
    state.isCameraRunning = false;
  }
}

function stopCamera() {
  if (cameraInstance && state.isCameraRunning) {
    cameraInstance.stop();
    state.isCameraRunning = false;
    btnCamera.classList.remove("btn-primary");
    btnCamera.classList.add("btn-secondary");
  }
}

// ---------------------------------------------------------------------------
// FaceMesh Results & Compositing Pipeline
// ---------------------------------------------------------------------------
function onFaceMeshResults(results) {
  const source = results.image;
  if (!source) return;

  const srcW = source.videoWidth || source.width;
  const srcH = source.videoHeight || source.height;

  if (canvas.width !== (srcW * (window.devicePixelRatio || 1))) {
    setupHighDpiCanvas(srcW, srcH);
  }

  const w = srcW;
  const h = srcH;

  ctx.save();
  ctx.clearRect(0, 0, w, h);

  if (state.mirror) {
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
  }

  ctx.drawImage(source, 0, 0, w, h);

  if (state.showOriginal || !results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
    ctx.restore();
    updateFps();
    if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
      statusText.textContent = "No face detected";
    }
    return;
  }

  const landmarks = results.multiFaceLandmarks[0];
  state.activeLandmarks = landmarks;

  // Apply Makeup Layers
  applyAllMakeupLayers(ctx, landmarks, w, h);

  ctx.restore();
  updateFps();
  statusText.textContent = "Face detected - Real-time tracking";
}

// ---------------------------------------------------------------------------
// Layered Compositor
// ---------------------------------------------------------------------------
function applyAllMakeupLayers(context, lm, w, h) {
  const faceLeft = getPoint(lm, 234, w, h);
  const faceRight = getPoint(lm, 454, w, h);
  const faceW = Math.hypot(faceRight.x - faceLeft.x, faceRight.y - faceLeft.y);

  // 1. Foundation / Skin Tint (Full Forehead with seamless hairline falloff)
  if (state.products.foundation.enabled && state.products.foundation.intensity > 0) {
    renderFoundation(context, lm, w, h, faceW, state.products.foundation);
  }

  // 2. Blushy Cheeks (Ultra-Soft, Naturally Diffused into Skin)
  if (state.products.blush.enabled && state.products.blush.intensity > 0) {
    renderBlush(context, lm, w, h, faceW, state.products.blush);
  }

  // 3. Eyeshadow Wash (Full Lid Crease)
  if (state.products.eyeshadow.enabled && state.products.eyeshadow.intensity > 0) {
    renderEyeshadow(context, lm, w, h, faceW, state.products.eyeshadow);
  }

  // 4. Cute Eyeliner (Smooth & Evenly Blended)
  if (state.products.eyeliner.enabled && state.products.eyeliner.intensity > 0) {
    renderEyeliner(context, lm, w, h, faceW, state.products.eyeliner);
  }

  // 5. Brow Fluff (Soft, Natural Hair Tint)
  if (state.products.brow.enabled && state.products.brow.intensity > 0) {
    renderBrow(context, lm, w, h, faceW, state.products.brow);
  }

  // 6. Pouty Matte Lips (Velvet Matte, Teeth Protected, No Gloss)
  if (state.products.lipstick.enabled && state.products.lipstick.intensity > 0) {
    renderLipstick(context, lm, w, h, faceW, state.products.lipstick);
  }
}

// ---------------------------------------------------------------------------
// 1. Foundation / Skin Tint (Seamless Hairline)
// ---------------------------------------------------------------------------
function renderFoundation(c, lm, w, h, faceW, p) {
  c.save();
  c.globalCompositeOperation = "soft-light";
  c.globalAlpha = p.intensity * 0.38;

  const midEyes = getPoint(lm, 168, w, h);
  const top10 = getPoint(lm, 10, w, h);
  const foreheadH = Math.hypot(top10.x - midEyes.x, top10.y - midEyes.y);
  const extDist = foreheadH * 0.48;

  let hairlineY = Infinity;
  for (const idx of FOREHEAD_TOP) {
    const pt = getPoint(lm, idx, w, h);
    if (pt.y - extDist < hairlineY) hairlineY = pt.y - extDist;
  }
  const browCenter = getCentroid(lm, [...LEFT_BROW, ...RIGHT_BROW], w, h);

  c.beginPath();
  for (let i = 0; i < FACE_OVAL.length; i++) {
    const pt = getPoint(lm, FACE_OVAL[i], w, h);
    if (i === 0) c.moveTo(pt.x, pt.y);
    else c.lineTo(pt.x, pt.y);
  }

  for (let i = FOREHEAD_TOP.length - 1; i >= 0; i--) {
    const pt = getPoint(lm, FOREHEAD_TOP[i], w, h);
    const radVec = { x: pt.x - midEyes.x, y: pt.y - midEyes.y };
    const rLen = Math.hypot(radVec.x, radVec.y) || 1;
    const extPt = {
      x: pt.x + (radVec.x / rLen) * extDist,
      y: pt.y - extDist * 0.9,
    };
    c.lineTo(extPt.x, extPt.y);
  }
  c.closePath();

  drawPolygonHole(c, lm, LEFT_EYE, w, h);
  drawPolygonHole(c, lm, RIGHT_EYE, w, h);
  drawPolygonHole(c, lm, OUTER_LIP, w, h);

  const grad = c.createLinearGradient(0, hairlineY, 0, browCenter.y);
  const hex = p.color;
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  grad.addColorStop(0, `rgba(${r}, ${g}, ${b}, 0.0)`);
  grad.addColorStop(0.35, `rgba(${r}, ${g}, ${b}, 0.55)`);
  grad.addColorStop(1, `rgba(${r}, ${g}, ${b}, 1.0)`);

  c.fillStyle = grad;
  c.filter = `blur(${Math.max(20, Math.round(faceW * 0.11))}px)`;
  c.fill("evenodd");
  c.restore();
}

// ---------------------------------------------------------------------------
// 2. Blushy Cheeks (Pore-Diffused Soft Glow)
// ---------------------------------------------------------------------------
function renderBlush(c, lm, w, h, faceW, p) {
  c.save();
  c.globalCompositeOperation = "soft-light";  // Seamlessly diffuses into skin
  c.globalAlpha = p.intensity * 0.60;

  const radiusX = Math.max(15, faceW * 0.15);
  const radiusY = Math.max(11, faceW * 0.11);

  const hex = p.color;
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);

  for (const cheekIndices of [LEFT_CHEEK, RIGHT_CHEEK]) {
    const center = getCentroid(lm, cheekIndices, w, h);
    c.beginPath();
    c.ellipse(center.x, center.y, radiusX, radiusY, 0, 0, Math.PI * 2);

    // Continuous radial Gaussian falloff to melt into cheek skin
    const radGrad = c.createRadialGradient(center.x, center.y, 0, center.x, center.y, radiusX);
    radGrad.addColorStop(0, `rgba(${r}, ${g}, ${b}, 1.0)`);
    radGrad.addColorStop(0.40, `rgba(${r}, ${g}, ${b}, 0.55)`);
    radGrad.addColorStop(0.80, `rgba(${r}, ${g}, ${b}, 0.15)`);
    radGrad.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0.0)`);

    c.fillStyle = radGrad;
    c.filter = `blur(${Math.max(18, Math.round(faceW * 0.11))}px)`;
    c.fill();
  }
  c.restore();
}

// ---------------------------------------------------------------------------
// 3. Eyeshadow Wash (Full Lid Crease)
// ---------------------------------------------------------------------------
function renderEyeshadow(c, lm, w, h, faceW, p) {
  c.save();
  c.globalCompositeOperation = "overlay";
  c.globalAlpha = p.intensity * 0.55;

  const pairs = [
    { lid: LEFT_UPPER_LID, brow: LEFT_BROW, eye: LEFT_EYE },
    { lid: RIGHT_UPPER_LID, brow: RIGHT_BROW, eye: RIGHT_EYE },
  ];

  for (const { lid, brow, eye } of pairs) {
    const lidPts = lid.map(idx => getPoint(lm, idx, w, h));
    const browPts = brow.map(idx => getPoint(lm, idx, w, h));

    const upperBoundary = lidPts.map(pt => {
      let bestBrow = browPts[0];
      let minDiff = Infinity;
      for (const bp of browPts) {
        const diff = Math.abs(bp.x - pt.x);
        if (diff < minDiff) {
          minDiff = diff;
          bestBrow = bp;
        }
      }
      return {
        x: (1.0 - 0.52) * pt.x + 0.52 * bestBrow.x,
        y: (1.0 - 0.52) * pt.y + 0.52 * bestBrow.y,
      };
    });

    c.beginPath();
    c.moveTo(lidPts[0].x, lidPts[0].y);
    for (let i = 1; i < lidPts.length; i++) c.lineTo(lidPts[i].x, lidPts[i].y);
    for (let i = upperBoundary.length - 1; i >= 0; i--) c.lineTo(upperBoundary[i].x, upperBoundary[i].y);
    c.closePath();

    drawPolygonHole(c, lm, eye, w, h);

    c.fillStyle = p.color;
    c.filter = `blur(${Math.max(6, Math.round(faceW * 0.038))}px)`;
    c.fill("evenodd");
  }

  c.restore();
}

// ---------------------------------------------------------------------------
// 4. Cute Eyeliner (Smooth & Evenly Blended Along Lash Line)
// ---------------------------------------------------------------------------
function renderEyeliner(c, lm, w, h, faceW, p) {
  c.save();
  c.globalCompositeOperation = "multiply";
  c.globalAlpha = p.intensity;
  c.fillStyle = p.color;
  c.strokeStyle = p.color;
  c.lineCap = "round";
  c.lineJoin = "round";

  const scale = Math.max(1.0, faceW / 200.0);
  const style = p.shape || "classic_wing";

  const eyes = [
    { lid: LEFT_UPPER_LID, outer: 33, inner: 133, browTail: 70 },
    { lid: RIGHT_UPPER_LID, outer: 263, inner: 362, browTail: 300 },
  ];

  // Soft anti-aliased blur for even blending
  c.filter = `blur(${Math.max(1.5, scale * 1.6)}px)`;

  for (const { lid, outer, inner, browTail } of eyes) {
    const lidPts = lid.map(idx => getPoint(lm, idx, w, h));
    const outerPt = getPoint(lm, outer, w, h);
    const innerPt = getPoint(lm, inner, w, h);
    const browPt = getPoint(lm, browTail, w, h);

    const eyeLen = Math.hypot(outerPt.x - innerPt.x, outerPt.y - innerPt.y);
    const wingVec = { x: browPt.x - outerPt.x, y: browPt.y - outerPt.y };
    const wingLen = Math.hypot(wingVec.x, wingVec.y) || 1;
    const wingDir = { x: wingVec.x / wingLen, y: wingVec.y / wingLen };

    // Draw smoothed quadratic curve along lash line
    c.beginPath();
    c.moveTo(lidPts[0].x, lidPts[0].y);
    for (let i = 1; i < lidPts.length - 1; i++) {
      const xc = (lidPts[i].x + lidPts[i + 1].x) / 2;
      const yc = (lidPts[i].y + lidPts[i + 1].y) / 2;
      c.quadraticCurveTo(lidPts[i].x, lidPts[i].y, xc, yc);
    }
    c.lineTo(lidPts[lidPts.length - 1].x, lidPts[lidPts.length - 1].y);

    if (style === "tightline") {
      c.lineWidth = Math.max(1.6, 2.0 * scale);
      c.stroke();
    } else if (style === "classic_wing") {
      c.lineWidth = Math.max(2.2, 2.8 * scale);
      c.stroke();

      const wingTip = {
        x: outerPt.x + wingDir.x * (eyeLen * 0.28),
        y: outerPt.y + wingDir.y * (eyeLen * 0.28),
      };
      c.beginPath();
      c.moveTo(outerPt.x, outerPt.y);
      c.lineTo(wingTip.x, wingTip.y);
      c.lineTo(outerPt.x - (outerPt.x - innerPt.x) * 0.16, outerPt.y);
      c.closePath();
      c.fill();
    } else if (style === "cat_eye") {
      c.lineWidth = Math.max(2.6, 3.4 * scale);
      c.stroke();

      const wingTip = {
        x: outerPt.x + wingDir.x * (eyeLen * 0.38),
        y: outerPt.y + wingDir.y * (eyeLen * 0.38),
      };
      c.beginPath();
      c.moveTo(outerPt.x, outerPt.y);
      c.lineTo(wingTip.x, wingTip.y);
      c.lineTo(outerPt.x - (outerPt.x - innerPt.x) * 0.24, outerPt.y);
      c.closePath();
      c.fill();

      // Inner corner flick
      c.beginPath();
      c.moveTo(innerPt.x, innerPt.y);
      c.lineTo(innerPt.x - (outerPt.x - innerPt.x) * 0.08, innerPt.y + 2);
      c.lineTo(innerPt.x + 2, innerPt.y);
      c.closePath();
      c.fill();
    } else { // bold / smoky kohl
      c.lineWidth = Math.max(3.2, 4.2 * scale);
      c.stroke();

      const wingTip = {
        x: outerPt.x + wingDir.x * (eyeLen * 0.22),
        y: outerPt.y + wingDir.y * (eyeLen * 0.22),
      };
      c.beginPath();
      c.moveTo(outerPt.x, outerPt.y);
      c.lineTo(wingTip.x, wingTip.y);
      c.lineTo(outerPt.x - (outerPt.x - innerPt.x) * 0.15, outerPt.y);
      c.closePath();
      c.fill();
    }
  }

  c.restore();
}

// ---------------------------------------------------------------------------
// 5. Brow Fluff (Soft & Natural)
// ---------------------------------------------------------------------------
function renderBrow(c, lm, w, h, faceW, p) {
  c.save();
  c.globalCompositeOperation = "multiply";
  c.globalAlpha = p.intensity * 0.35;
  c.fillStyle = p.color;
  c.filter = `blur(${Math.max(5, Math.round(faceW * 0.032))}px)`;

  for (const browIndices of [LEFT_BROW, RIGHT_BROW]) {
    c.beginPath();
    for (let i = 0; i < browIndices.length; i++) {
      const pt = getPoint(lm, browIndices[i], w, h);
      if (i === 0) c.moveTo(pt.x, pt.y);
      else c.lineTo(pt.x, pt.y);
    }
    c.closePath();
    c.fill();
  }

  c.restore();
}

// ---------------------------------------------------------------------------
// 6. Pouty Matte Lips (Velvet Matte, Teeth-Safe, No Gloss)
// ---------------------------------------------------------------------------
function renderLipstick(c, lm, w, h, faceW, p) {
  c.save();

  c.globalCompositeOperation = p.blend === "color" ? "color" : p.blend;
  c.globalAlpha = p.intensity;

  c.beginPath();
  for (let i = 0; i < OUTER_LIP.length; i++) {
    const pt = getPoint(lm, OUTER_LIP[i], w, h);
    if (i === 0) c.moveTo(pt.x, pt.y);
    else c.lineTo(pt.x, pt.y);
  }
  c.closePath();

  // Subtract inner mouth opening strictly so teeth/tongue never receive lipstick
  drawPolygonHole(c, lm, INNER_LIP, w, h);

  c.fillStyle = p.color;
  c.filter = `blur(${Math.max(2, Math.round(faceW * 0.012))}px)`;
  c.fill("evenodd");

  c.restore();
}

// ---------------------------------------------------------------------------
// Geometric Helpers
// ---------------------------------------------------------------------------
function getPoint(lm, idx, w, h) {
  const p = lm[idx];
  return { x: p.x * w, y: p.y * h };
}

function getCentroid(lm, indices, w, h) {
  let sumX = 0;
  let sumY = 0;
  for (const idx of indices) {
    const pt = getPoint(lm, idx, w, h);
    sumX += pt.x;
    sumY += pt.y;
  }
  return { x: sumX / indices.length, y: sumY / indices.length };
}

function drawPolygonHole(c, lm, indices, w, h) {
  if (!indices || indices.length < 3) return;
  const first = getPoint(lm, indices[0], w, h);
  c.moveTo(first.x, first.y);
  for (let i = 1; i < indices.length; i++) {
    const pt = getPoint(lm, indices[i], w, h);
    c.lineTo(pt.x, pt.y);
  }
  c.closePath();
}

// ---------------------------------------------------------------------------
// UI Binding & Controls
// ---------------------------------------------------------------------------
function setupControls() {
  document.querySelectorAll(".swatches").forEach(container => {
    const product = container.dataset.product;
    const swatches = SWATCHES[product] || [];

    swatches.forEach((s, idx) => {
      const btn = document.createElement("button");
      btn.className = `swatch-btn ${idx === 0 ? "active" : ""}`;
      btn.style.backgroundColor = s.hex;
      btn.title = `${s.name} (${s.hex})`;
      btn.addEventListener("click", () => {
        container.querySelectorAll(".swatch-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        state.products[product].color = s.hex;
        const colorInput = document.getElementById(`color-${product}`);
        if (colorInput) colorInput.value = s.hex;
      });
      container.appendChild(btn);
    });
  });

  Object.keys(state.products).forEach(product => {
    const chk = document.getElementById(`chk-${product}`);
    if (chk) {
      chk.addEventListener("change", (e) => {
        state.products[product].enabled = e.target.checked;
      });
    }

    const colorInput = document.getElementById(`color-${product}`);
    if (colorInput) {
      colorInput.addEventListener("input", (e) => {
        state.products[product].color = e.target.value;
        const container = document.querySelector(`.swatches[data-product="${product}"]`);
        if (container) {
          container.querySelectorAll(".swatch-btn").forEach(b => b.classList.remove("active"));
        }
      });
    }

    const intensityInput = document.getElementById(`intensity-${product}`);
    const valDisplay = document.getElementById(`val-${product}`);
    if (intensityInput && valDisplay) {
      intensityInput.addEventListener("input", (e) => {
        const val = parseInt(e.target.value, 10);
        valDisplay.textContent = `${val}%`;
        state.products[product].intensity = val / 100.0;
      });
    }
  });

  const shapeSelect = document.getElementById("select-eyeliner-shape");
  if (shapeSelect) {
    shapeSelect.addEventListener("change", (e) => {
      state.products.eyeliner.shape = e.target.value;
    });
  }

  const blendSelect = document.getElementById("select-lipstick-blend");
  if (blendSelect) {
    blendSelect.addEventListener("change", (e) => {
      state.products.lipstick.blend = e.target.value;
    });
  }

  btnCamera.addEventListener("click", () => {
    if (state.isCameraRunning) {
      stopCamera();
    } else {
      startCamera();
    }
  });

  btnSave.addEventListener("click", () => {
    const link = document.createElement("a");
    link.download = `makeup_snapshot_${Date.now()}.jpg`;
    link.href = canvas.toDataURL("image/jpeg", 0.95);
    link.click();
  });

  chkMirror.addEventListener("change", (e) => {
    state.mirror = e.target.checked;
  });

  chkOriginal.addEventListener("change", (e) => {
    state.showOriginal = e.target.checked;
  });

  btnResetAll.addEventListener("click", () => {
    Object.keys(state.products).forEach(p => {
      state.products[p].enabled = false;
      const chk = document.getElementById(`chk-${p}`);
      if (chk) chk.checked = false;
    });

    state.products.lipstick.enabled = true;
    document.getElementById("chk-lipstick").checked = true;
    state.products.lipstick.blend = "multiply";

    state.products.blush.enabled = true;
    document.getElementById("chk-blush").checked = true;
    state.products.blush.intensity = 0.28;
    const blushInput = document.getElementById("intensity-blush");
    const blushVal = document.getElementById("val-blush");
    if (blushInput) blushInput.value = "28";
    if (blushVal) blushVal.textContent = "28%";

    state.products.brow.enabled = true;
    document.getElementById("chk-brow").checked = true;
    state.products.brow.intensity = 0.25;
    const browInput = document.getElementById("intensity-brow");
    const browVal = document.getElementById("val-brow");
    if (browInput) browInput.value = "25";
    if (browVal) browVal.textContent = "25%";

    state.products.eyeshadow.enabled = true;
    document.getElementById("chk-eyeshadow").checked = true;
    state.products.eyeshadow.intensity = 0.32;
  });
}

// ---------------------------------------------------------------------------
// FPS Counter
// ---------------------------------------------------------------------------
function updateFps() {
  frameCounter++;
  const now = performance.now();
  const elapsed = now - lastFpsTime;
  if (elapsed >= 500) {
    const fps = Math.round((frameCounter / elapsed) * 1000);
    fpsBadge.textContent = `${fps} FPS`;
    frameCounter = 0;
    lastFpsTime = now;
  }
}

// ---------------------------------------------------------------------------
// App Bootstrap
// ---------------------------------------------------------------------------
window.addEventListener("DOMContentLoaded", () => {
  setupControls();
  initMediaPipe();
});