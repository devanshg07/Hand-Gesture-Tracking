const video = document.getElementById('video');
const counter = document.getElementById('counter');
let lastFingers = null, lastHands = null;

// Ask for webcam
navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
  video.srcObject = stream;
});

// Animate counter on change
function animateCounter() {
  counter.classList.remove('counter-animate');
  void counter.offsetWidth; // trigger reflow
  counter.classList.add('counter-animate');
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
            // Speak only if finger count changes
            if (fingers !== lastFingers) {
              window.speechSynthesis.cancel();
              const utter = new SpeechSynthesisUtterance(fingers.toString());
              utter.rate = 1.1;
              utter.pitch = 1.1;
              utter.volume = 1;
              utter.lang = 'en-US';
              window.speechSynthesis.speak(utter);
            }
            lastFingers = fingers;
            lastHands = hands;
          }
        });
    }, 'image/jpeg');
  }
}, 500); 