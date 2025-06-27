import cv2
import mediapipe as mp
import time
import numpy as np
import pyautogui
from collections import deque

class HandTracker:
    def __init__(self, static_mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        """
        Initialize MediaPipe hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hands detector
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        
        # Hand landmarks
        self.results = None
        self.landmarks = []
        
        # Gesture detection
        self.last_gesture = None
        self.gesture_cooldown = 0
        
        # Colors for visualization
        self.colors = {
            'landmark': (255, 0, 255),      # Pink
            'connection': (0, 255, 0),      # Green
            'text': (255, 255, 255),        # White
            'fps': (0, 255, 255),           # Yellow
            'thumbs_up': (0, 255, 0),       # Green
            'thumbs_down': (0, 0, 255)      # Red
        }
    
    def detect_hands(self, image, draw=True):
        """
        Detect hands in the image and optionally draw landmarks
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        self.results = self.hands.process(image_rgb)
        
        # Draw landmarks if requested
        if draw and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return image
    
    def get_landmarks(self, image, hand_no=0, draw_points=True):
        """
        Get landmark positions for a specific hand
        Returns: list of [id, x, y] coordinates
        """
        landmarks = []
        
        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[hand_no]
                
                for id, lm in enumerate(hand.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append([id, cx, cy])
                    
                    # Draw individual points if requested
                    if draw_points:
                        cv2.circle(image, (cx, cy), 5, self.colors['landmark'], cv2.FILLED)
        
        return landmarks
    
    def detect_thumbs_gesture(self, landmarks):
        """
        Detect thumbs up or thumbs down gesture
        Returns: 'up', 'down', or None
        """
        if len(landmarks) < 21:
            return None
        
        # Get key landmarks
        thumb_tip = landmarks[4]      # Thumb tip
        thumb_ip = landmarks[3]       # Thumb interphalangeal joint
        thumb_mcp = landmarks[2]      # Thumb metacarpophalangeal joint
        index_tip = landmarks[8]      # Index finger tip
        index_pip = landmarks[6]      # Index finger pip
        middle_tip = landmarks[12]    # Middle finger tip
        middle_pip = landmarks[10]    # Middle finger pip
        ring_tip = landmarks[16]      # Ring finger tip
        ring_pip = landmarks[14]      # Ring finger pip
        pinky_tip = landmarks[20]     # Pinky tip
        pinky_pip = landmarks[18]     # Pinky pip
        
        # Check if fingers are extended (except thumb)
        fingers_extended = []
        
        # Index finger
        fingers_extended.append(index_tip[2] < index_pip[2])
        # Middle finger
        fingers_extended.append(middle_tip[2] < middle_pip[2])
        # Ring finger
        fingers_extended.append(ring_tip[2] < ring_pip[2])
        # Pinky
        fingers_extended.append(pinky_tip[2] < pinky_pip[2])
        
        # Check thumb position relative to index finger
        thumb_above_index = thumb_tip[2] < index_tip[2]
        thumb_below_index = thumb_tip[2] > index_tip[2]
        
        # Thumbs up: all fingers extended, thumb above index
        if all(fingers_extended) and thumb_above_index:
            return 'up'
        
        # Thumbs down: all fingers extended, thumb below index
        if all(fingers_extended) and thumb_below_index:
            return 'down'
        
        return None
    
    def process_gesture(self, image):
        """
        Process gesture and simulate key press
        """
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
            return None
        
        landmarks = self.get_landmarks(image, hand_no=0, draw_points=False)
        if not landmarks:
            return None
        
        gesture = self.detect_thumbs_gesture(landmarks)
        
        if gesture and gesture != self.last_gesture:
            self.last_gesture = gesture
            self.gesture_cooldown = 30  # Prevent rapid repeated presses
            
            # Simulate key press
            if gesture == 'up':
                pyautogui.press('up')
                return 'UP ARROW'
            elif gesture == 'down':
                pyautogui.press('down')
                return 'DOWN ARROW'
        
        return None
    
    def draw_gesture_info(self, image, gesture_text):
        """
        Draw gesture information on the image
        """
        if gesture_text:
            # Draw gesture text
            cv2.putText(image, f"Pressed: {gesture_text}", (10, image.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['thumbs_up'], 3)
            
            # Draw gesture indicator
            center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
            
            if gesture_text == 'UP ARROW':
                cv2.arrowedLine(image, (center_x, center_y + 50), (center_x, center_y - 50), 
                               self.colors['thumbs_up'], 5, tipLength=0.3)
            elif gesture_text == 'DOWN ARROW':
                cv2.arrowedLine(image, (center_x, center_y - 50), (center_x, center_y + 50), 
                               self.colors['thumbs_down'], 5, tipLength=0.3)
    
    def get_hand_count(self):
        """
        Get the number of hands detected
        """
        if self.results.multi_hand_landmarks:
            return len(self.results.multi_hand_landmarks)
        return 0
    
    def get_hand_bbox(self, image, hand_no=0):
        """
        Get bounding box for a specific hand
        Returns: (x, y, w, h) or None if no hand detected
        """
        landmarks = self.get_landmarks(image, hand_no, draw_points=False)
        
        if landmarks:
            x_coords = [lm[1] for lm in landmarks]
            y_coords = [lm[2] for lm in landmarks]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Add padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(image.shape[1], x_max + padding)
            y_max = min(image.shape[0], y_max + padding)
            
            return (x_min, y_min, x_max - x_min, y_max - y_min)
        
        return None

class FaceTracker:
    def __init__(self):
        """
        Initialize face detection and tracking
        """
        # Load pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize MediaPipe face detection for better accuracy
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for short-range, 1 for full-range
            min_detection_confidence=0.5
        )
        
        # Colors for visualization
        self.colors = {
            'face_bbox': (0, 255, 0),       # Green
            'face_landmarks': (255, 0, 0),  # Blue
            'text': (255, 255, 255)         # White
        }
        
        # Tracking variables
        self.face_tracked = False
        self.tracker = None
        self.face_bbox = None
        self.tracking_failures = 0
        self.max_tracking_failures = 10
    
    def detect_faces_mediapipe(self, image, draw=True):
        """
        Detect faces using MediaPipe (more accurate)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.face_detection.process(image_rgb)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                            int(bboxC.width * iw), int(bboxC.height * ih)
                
                faces.append((x, y, w, h))
                
                if draw:
                    # Draw bounding box
                    cv2.rectangle(image, (x, y), (x + w, y + h), self.colors['face_bbox'], 2)
                    
                    # Draw confidence score
                    confidence = detection.score[0]
                    cv2.putText(image, f'{confidence:.2f}', (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        return faces
    
    def detect_faces_opencv(self, image, draw=True):
        """
        Detect faces using OpenCV (fallback method)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if draw:
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), self.colors['face_bbox'], 2)
        
        return faces
    
    def get_face_count(self, image):
        """
        Get the number of faces detected
        """
        faces = self.detect_faces_mediapipe(image, draw=False)
        return len(faces)
    
    def get_largest_face(self, image):
        """
        Get the largest (closest) face detected
        """
        faces = self.detect_faces_mediapipe(image, draw=False)
        if faces:
            # Return the face with the largest area
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            return largest_face
        return None

class HandAndFaceTracker:
    def __init__(self):
        """
        Combined hand and face tracking system
        """
        self.hand_tracker = HandTracker(
            static_mode=False,
            max_hands=2,
            detection_confidence=0.7,
            tracking_confidence=0.5
        )
        
        self.face_tracker = FaceTracker()
        
        # Colors for combined visualization
        self.colors = {
            'hand': (255, 0, 255),      # Pink
            'face': (0, 255, 0),        # Green
            'text': (255, 255, 255),    # White
            'fps': (0, 255, 255),       # Yellow
            'info': (255, 255, 0)       # Cyan
        }
    
    def process_frame(self, image):
        """
        Process frame for both hand and face detection
        """
        # Detect and draw hands
        image = self.hand_tracker.detect_hands(image, draw=True)
        
        # Detect gesture
        gesture = self.hand_tracker.process_gesture(image)
        if gesture:
            self.hand_tracker.draw_gesture_info(image, gesture)
        
        # Detect and draw faces
        self.face_tracker.detect_faces_mediapipe(image, draw=True)
        
        return image
    
    def get_counts(self, image):
        """
        Get counts of hands and faces detected
        """
        hand_count = self.hand_tracker.get_hand_count()
        face_count = self.face_tracker.get_face_count(image)
        
        return hand_count, face_count
    
    def draw_info(self, image, fps=None, hand_count=None, face_count=None):
        """
        Draw information on the image
        """
        y_offset = 30
        
        # Draw FPS
        if fps is not None:
            cv2.putText(image, f"FPS: {int(fps)}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['fps'], 2)
            y_offset += 30
        
        # Draw hand count
        if hand_count is not None:
            cv2.putText(image, f"Hands: {hand_count}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['hand'], 2)
            y_offset += 30
        
        # Draw face count
        if face_count is not None:
            cv2.putText(image, f"Faces: {face_count}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['face'], 2)
            y_offset += 30
        
        # Draw instructions
        cv2.putText(image, "Press 'q' to quit", (10, image.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)

def main():
    """
    Main function to run hand and face tracking
    """
    # Initialize camera - try multiple indices
    cap = None
    camera_indices = [0, 1, 2]  # Try different camera indices
    
    for camera_index in camera_indices:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            break
    
    # Check if camera opened successfully
    if not cap or not cap.isOpened():
        # Try to open default camera without specifying index
        cap = cv2.VideoCapture()
        if not cap.isOpened():
            return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize combined tracker
    tracker = HandAndFaceTracker()
    
    # FPS calculation variables
    prev_time = 0
    curr_time = 0
    
    try:
        while True:
            # Read frame
            success, frame = cap.read()
            if not success:
                break
            
            # Flip frame horizontally for more intuitive experience
            frame = cv2.flip(frame, 1)
            
            # Process frame for hand and face detection
            frame = tracker.process_frame(frame)
            
            # Get counts
            hand_count, face_count = tracker.get_counts(frame)
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            
            # Draw information
            tracker.draw_info(frame, fps=fps, hand_count=hand_count, face_count=face_count)
            
            # Show frame
            cv2.imshow("Hand & Face Tracking", frame)
            
            # Handle key presses and window close
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # Check if window was closed with X button
            if cv2.getWindowProperty("Hand & Face Tracking", cv2.WND_PROP_VISIBLE) < 1:
                break
        
    except KeyboardInterrupt:
        pass
    
    finally:
        # Cleanup
        if cap:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()