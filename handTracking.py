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
        
        # Colors for visualization
        self.colors = {
            'landmark': (255, 0, 255),      # Pink
            'connection': (0, 255, 0),      # Green
            'text': (255, 255, 255),        # White
            'fps': (0, 255, 255)            # Yellow
        }
        
        print("Hand Tracker initialized successfully!")
        print("Press 'q' to quit, 'r' to reset")
    
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
    
    def draw_info(self, image, fps=None, hand_count=None):
        """
        Draw information on the image
        """
        # Draw FPS
        if fps is not None:
            cv2.putText(image, f"FPS: {int(fps)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['fps'], 2)
        
        # Draw hand count
        if hand_count is not None:
            cv2.putText(image, f"Hands: {hand_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        
        # Draw instructions
        cv2.putText(image, "Press 'q' to quit, 'r' to reset", (10, image.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)

def main():
    """
    Main function to run hand tracking
    """
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Try camera 0 first
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Initialize hand tracker
    tracker = HandTracker(
        static_mode=False,
        max_hands=2,
        detection_confidence=0.7,
        tracking_confidence=0.5
    )
    
    # FPS calculation variables
    prev_time = 0
    curr_time = 0
    
    print("Hand tracking started! Show your hand to the camera.")
    
    try:
        while True:
            # Read frame
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame")
                break
            
            # Flip frame horizontally for more intuitive experience
            frame = cv2.flip(frame, 1)
            
            # Detect hands
            frame = tracker.detect_hands(frame, draw=True)
            
            # Get hand count
            hand_count = tracker.get_hand_count()
            
            # Get landmarks for first hand if detected
            if hand_count > 0:
                landmarks = tracker.get_landmarks(frame, hand_no=0, draw_points=False)
                
                # Print some useful landmarks
                if landmarks:
                    # Thumb tip (landmark 4)
                    thumb_tip = landmarks[4]
                    print(f"Thumb tip: {thumb_tip[1]}, {thumb_tip[2]}")
                    
                    # Index finger tip (landmark 8)
                    index_tip = landmarks[8]
                    print(f"Index tip: {index_tip[1]}, {index_tip[2]}")
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            
            # Draw information
            tracker.draw_info(frame, fps=fps, hand_count=hand_count)
            
            # Show frame
            cv2.imshow("Hand Tracking", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Reset requested")
                # You can add reset functionality here if needed
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Hand tracking stopped!")

if __name__ == "__main__":
    main()