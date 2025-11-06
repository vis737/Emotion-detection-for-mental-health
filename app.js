const video = document.getElementById("video");
const btnCapture = document.getElementById("btnCapture");
const btnAuto = document.getElementById("btnStartAuto");
const btnStopAuto = document.getElementById("btnStopAuto");
const status = document.getElementById("status");
const result = document.getElementById("result");
const btnListen = document.getElementById("btnListen");
const btnBreath = document.getElementById("btnBreath");

let autoInterval = null;

async function initCamera(){
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    video.srcObject = stream;
    status.innerText = "Camera ready";
  } catch (e) {
    status.innerText = "Camera error: " + e.message;
  }
}

function captureFrame(){
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 320;
  canvas.height = video.videoHeight || 240;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.9));
}

async function analyzeOnce(){
  status.innerText = "Capturing...";
  const blob = await captureFrame();
  const fd = new FormData();
  fd.append("frame", blob, "frame.jpg");
  status.innerText = "Sending to server...";
  try {
    const res = await fetch("/predict", { method: "POST", body: fd });
    const data = await res.json();
    status.innerText = "Got result";
    result.innerText = `Emotion: ${data.label}  (conf ${data.confidence.toFixed(2)})`;
    if (data.label === "stressed" && data.confidence > 0.7) {
      speak("I notice signs of stress. Would you like a short breathing exercise? Say yes or press the breathing button.");
    }
  } catch (e) {
    status.innerText = "Error: " + e.message;
  }
}

btnCapture.onclick = analyzeOnce;
btnAuto.onclick = () => { if (!autoInterval) autoInterval = setInterval(analyzeOnce, 3000); };
btnStopAuto.onclick = () => { if (autoInterval) { clearInterval(autoInterval); autoInterval = null; } };

initCamera();

// Browser TTS
function speak(text) {
  if (!("speechSynthesis" in window)) return;
  const ut = new SpeechSynthesisUtterance(text);
  ut.rate = 0.95;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(ut);
}

btnBreath.onclick = () => {
  speak("Let's do a short breathing exercise. Breathe in for four. Hold for four. Breathe out for six. Repeat three times.");
  // optionally implement a guided loop with timers
};

// Basic browser ASR (Web Speech API)
btnListen.onclick = () => {
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    alert("SpeechRecognition not supported in this browser. Use Chrome.");
    return;
  }
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recog = new SpeechRecognition();
  recog.lang = 'en-US';
  recog.onstart = () => { status.innerText = "Listening..."; };
  recog.onerror = (e) => { status.innerText = "ASR error"; };
  recog.onresult = (ev) => {
    const txt = ev.results[0][0].transcript.toLowerCase();
    status.innerText = "You said: " + txt;
    if (txt.includes("yes")) {
      speak("Okay, starting breathing exercise.");
      btnBreath.click();
    } else {
      speak("Alright. I'm here if you change your mind.");
    }
  };
  recog.start();
};
