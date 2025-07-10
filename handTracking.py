import cv2
import mediapipe as mp
import time
import numpy as np

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
        
        # Check if fingers are extended (except thumb) - more lenient
        fingers_extended = []
        
        # Index finger - more lenient threshold
        fingers_extended.append(index_tip[2] < index_pip[2] - 10)
        # Middle finger - more lenient threshold
        fingers_extended.append(middle_tip[2] < middle_pip[2] - 10)
        # Ring finger - more lenient threshold
        fingers_extended.append(ring_tip[2] < ring_pip[2] - 10)
        # Pinky - more lenient threshold
        fingers_extended.append(pinky_tip[2] < pinky_pip[2] - 10)
        
        # Check thumb position relative to index finger - more lenient
        thumb_above_index = thumb_tip[2] < index_tip[2] - 20
        thumb_below_index = thumb_tip[2] > index_tip[2] + 20
        
        # Thumbs up: all fingers extended, thumb above index
        if all(fingers_extended) and thumb_above_index:
            return 'up'
        
        # Thumbs down: all fingers extended, thumb below index
        if all(fingers_extended) and thumb_below_index:
            return 'down'
        
        return None
    
    def process_gesture(self, image):
        """
        Process gesture and draw arrow next to hand
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
            self.gesture_cooldown = 30  # Prevent rapid repeated detections
            return gesture
        
        return None
    
    def draw_arrow_next_to_hand(self, image, gesture, landmarks):
        """
        Draw arrow next to the detected hand
        """
        if not gesture or not landmarks:
            return
        
        # Get hand position (use wrist as reference)
        wrist = landmarks[0]
        hand_x, hand_y = wrist[1], wrist[2]
        
        # Calculate arrow position (to the right of the hand)
        arrow_x = hand_x + 100
        arrow_y = hand_y
        
        # Ensure arrow stays within image bounds
        arrow_x = min(arrow_x, image.shape[1] - 50)
        arrow_x = max(arrow_x, 50)
        arrow_y = min(arrow_y, image.shape[0] - 50)
        arrow_y = max(arrow_y, 50)
        
        # Draw arrow based on gesture
        if gesture == 'up':
            # Draw up arrow
            cv2.arrowedLine(image, 
                           (arrow_x, arrow_y + 30), 
                           (arrow_x, arrow_y - 30), 
                           self.colors['thumbs_up'], 5, tipLength=0.3)
            
            # Draw gesture text
            cv2.putText(image, "THUMBS UP", (arrow_x - 40, arrow_y + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['thumbs_up'], 2)
            
        elif gesture == 'down':
            # Draw down arrow
            cv2.arrowedLine(image, 
                           (arrow_x, arrow_y - 30), 
                           (arrow_x, arrow_y + 30), 
                           self.colors['thumbs_down'], 5, tipLength=0.3)
            
            # Draw gesture text
            cv2.putText(image, "THUMBS DOWN", (arrow_x - 50, arrow_y + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['thumbs_down'], 2)
    
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

    def count_extended_fingers(self, landmarks):
        """
        Count the number of extended fingers
        Returns: int (0-5) representing number of extended fingers
        """
        if len(landmarks) < 21:
            return 0
        
        # Get key landmarks for each finger
        # Thumb landmarks
        thumb_tip = landmarks[4]      # Thumb tip
        thumb_ip = landmarks[3]       # Thumb interphalangeal joint
        
        # Index finger landmarks
        index_tip = landmarks[8]      # Index finger tip
        index_pip = landmarks[6]      # Index finger pip
        
        # Middle finger landmarks
        middle_tip = landmarks[12]    # Middle finger tip
        middle_pip = landmarks[10]    # Middle finger pip
        
        # Ring finger landmarks
        ring_tip = landmarks[16]      # Ring finger tip
        ring_pip = landmarks[14]      # Ring finger pip
        
        # Pinky landmarks
        pinky_tip = landmarks[20]     # Pinky tip
        pinky_pip = landmarks[18]     # Pinky pip
        
        # Check if each finger is extended
        fingers_extended = []
        
        # Thumb - check if tip is above the interphalangeal joint
        thumb_extended = thumb_tip[2] < thumb_ip[2] - 10
        fingers_extended.append(thumb_extended)
        
        # Index finger - check if tip is above the pip joint
        index_extended = index_tip[2] < index_pip[2] - 10
        fingers_extended.append(index_extended)
        
        # Middle finger - check if tip is above the pip joint
        middle_extended = middle_tip[2] < middle_pip[2] - 10
        fingers_extended.append(middle_extended)
        
        # Ring finger - check if tip is above the pip joint
        ring_extended = ring_tip[2] < ring_pip[2] - 10
        fingers_extended.append(ring_extended)
        
        # Pinky - check if tip is above the pip joint
        pinky_extended = pinky_tip[2] < pinky_pip[2] - 10
        fingers_extended.append(pinky_extended)
        
        # Count extended fingers
        extended_count = sum(fingers_extended)
        
        return extended_count, fingers_extended

    def draw_finger_count(self, image, finger_count):
        """
        Draw finger count in the top left corner only
        """
        cv2.putText(image, f"FINGERS: {finger_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

class HandTrackerApp:
    def __init__(self):
        """
        Hand tracking application
        """
        self.hand_tracker = HandTracker(
            static_mode=False,
            max_hands=2,
            detection_confidence=0.7,
            tracking_confidence=0.5
        )
        
        # Colors for visualization
        self.colors = {
            'hand': (255, 0, 255),      # Pink
            'text': (255, 255, 255),    # White
            'fps': (0, 255, 255),       # Yellow
            'info': (255, 255, 0)       # Cyan
        }
    
    def process_frame(self, image):
        """
        Process frame for hand detection and finger counting
        """
        # Detect and draw hands
        image = self.hand_tracker.detect_hands(image, draw=True)
        
        # Get landmarks for finger counting and debug info
        landmarks = self.hand_tracker.get_landmarks(image, hand_no=0, draw_points=False)
        if landmarks:
            # Count extended fingers
            finger_count, _ = self.hand_tracker.count_extended_fingers(landmarks)
            
            # Draw finger count in top left
            self.hand_tracker.draw_finger_count(image, finger_count)
            
            # Detect gesture
            gesture = self.hand_tracker.process_gesture(image)
            if gesture:
                self.hand_tracker.draw_arrow_next_to_hand(image, gesture, landmarks)
        
        return image
    
    def get_counts(self, image):
        """
        Get counts of hands and fingers detected
        """
        hand_count = self.hand_tracker.get_hand_count()
        
        # Get finger count for the first detected hand
        finger_count = 0
        landmarks = self.hand_tracker.get_landmarks(image, hand_no=0, draw_points=False)
        if landmarks:
            finger_count, _ = self.hand_tracker.count_extended_fingers(landmarks)
        
        return hand_count, finger_count
    
    def draw_info(self, image, fps=None, hand_count=None, finger_count=None):
        """
        Draw only hand and finger count in the top left
        """
        y_offset = 30
        if hand_count is not None:
            cv2.putText(image, f"Hands: {hand_count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
            y_offset += 30
        if finger_count is not None:
            cv2.putText(image, f"FINGERS: {finger_count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

def main():
    """
    Main function to run hand tracking and finger counting
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
    
    # Initialize hand tracker app
    tracker = HandTrackerApp()
    
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
            
            # Process frame for hand detection and finger counting
            frame = tracker.process_frame(frame)
            
            # Get counts
            hand_count, finger_count = tracker.get_counts(frame)
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            
            # Draw information
            tracker.draw_info(frame, fps=fps, hand_count=hand_count, finger_count=finger_count)
            
            # Show frame
            cv2.imshow("Hand Tracking & Finger Counting", frame)
            
            # Handle key presses and window close
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # Check if window was closed with X button
            if cv2.getWindowProperty("Hand Tracking & Finger Counting", cv2.WND_PROP_VISIBLE) < 1:
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