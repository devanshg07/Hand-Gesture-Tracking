const video = document.getElementById('video');
const counter = document.getElementById('counter');
const speechIcon = document.getElementById('speech-icon');
const muteBtn = document.getElementById('mute-btn');
let lastFingers = null, lastHands = null;
let isMuted = false;
let speakingTimeout = null;

// Ask for webcam
navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
  video.srcObject = stream;
}).catch(() => {
  showToast('Camera access denied. Please allow camera to use this app.');
});

// Animate counter on change
function animateCounter() {
  counter.classList.remove('counter-animate');
  void counter.offsetWidth; // trigger reflow
  counter.classList.add('counter-animate');
}

// Show toast for errors
function showToast(msg) {
  let toast = document.createElement('div');
  toast.textContent = msg;
  toast.style.position = 'fixed';
  toast.style.bottom = '30px';
  toast.style.left = '50%';
  toast.style.transform = 'translateX(-50%)';
  toast.style.background = 'rgba(0,0,0,0.8)';
  toast.style.color = '#fff';
  toast.style.padding = '16px 32px';
  toast.style.borderRadius = '12px';
  toast.style.fontSize = '1.1em';
  toast.style.zIndex = 1000;
  toast.style.boxShadow = '0 2px 16px #00fff7aa';
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3500);
}

// Mute/unmute button
muteBtn.addEventListener('click', () => {
  isMuted = !isMuted;
  muteBtn.classList.toggle('muted', isMuted);
  muteBtn.innerHTML = isMuted ? '🔇' : '🔊';
});

// Animate speech icon
function animateSpeechIcon() {
  speechIcon.classList.add('speaking');
  clearTimeout(speakingTimeout);
  speakingTimeout = setTimeout(() => speechIcon.classList.remove('speaking'), 1200);
}

// Capture and send frame every 500ms
setInterval(() => {
  if (video.readyState === 4) {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(blob => {
      const formData = new FormData();
      formData.append('frame', blob, 'frame.jpg');
      fetch('/detect', { method: 'POST', body: formData })
        .then(res => res.json())
        .then(data => {
          const hands = data.hands;
          const fingers = data.fingers;
          // Only update and animate if changed
          if (hands !== lastHands || fingers !== lastFingers) {
            counter.textContent = `Hands: ${hands} | Fingers: ${fingers}`;
            animateCounter();
            // Speak only if finger count changes and not muted
            if (fingers !== lastFingers && !isMuted) {
              window.speechSynthesis.cancel();
              const utter = new SpeechSynthesisUtterance(fingers.toString());
              utter.rate = 1.1;
              utter.pitch = 1.1;
              utter.volume = 1;
              utter.lang = 'en-US';
              window.speechSynthesis.speak(utter);
              animateSpeechIcon();
            }
            lastFingers = fingers;
            lastHands = hands;
          }
        }).catch(() => showToast('Lost connection to server.'));
    }, 'image/jpeg');
  }
}, 500); 