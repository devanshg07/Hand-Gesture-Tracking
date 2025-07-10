from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from handTracking import HandTrackerApp

app = Flask(__name__)
tracker = HandTrackerApp()

@app.route('/')
def index():#function
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    tracker.process_frame(frame)
    hand_count, finger_count = tracker.get_counts(frame)
    return jsonify({'hands': hand_count, 'fingers': finger_count})

if __name__ == '__main__':
    app.run(debug=True) 