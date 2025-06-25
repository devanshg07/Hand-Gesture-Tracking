import cv2
import numpy as np
import time

class FaceTracker:
    def __init__(self):
        # Initialize the webcam
        self.cap = cv2.VideoCapture(0)
        
        # Load the pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize tracking variables
        self.face_tracked = False
        self.tracker = None
        self.face_bbox = None
        self.tracking_start_time = None
        
        # Colors for visualization
        self.GREEN = (0, 255, 0)
        self.RED = (0, 0, 255)
        self.BLUE = (255, 0, 0)
        self.YELLOW = (0, 255, 255)
        
        # Tracking parameters
        self.tracking_threshold = 0.3  # Minimum confidence for tracking
        self.detection_interval = 30   # Re-detect face every N frames
        
        print("Face Tracker initialized!")
        print("Press 'q' to quit, 'r' to reset tracking")
    
    def detect_faces(self, frame):
        """Detect faces in the frame using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def initialize_tracker(self, frame, bbox):
        """Initialize the tracker with a detected face"""
        # Use the correct tracker API for newer OpenCV versions
        try:
            # Try CSRT tracker first (more accurate but slower)
            self.tracker = cv2.TrackerCSRT_create()
        except AttributeError:
            try:
                # Fallback to KCF tracker
                self.tracker = cv2.TrackerKCF_create()
            except AttributeError:
                # Final fallback to MOSSE tracker
                self.tracker = cv2.TrackerMOSSE_create()
        
        success = self.tracker.init(frame, bbox)
        if success:
            self.face_tracked = True
            self.face_bbox = bbox
            self.tracking_start_time = time.time()
            print("Face tracking initialized!")
        return success
    
    def update_tracker(self, frame):
        """Update the tracker and return the new bounding box"""
        if self.tracker is None:
            return None, False
        
        success, bbox = self.tracker.update(frame)
        if success:
            self.face_bbox = bbox
        return bbox, success
    
    def draw_face_info(self, frame, bbox, is_tracking=False):
        """Draw face bounding box and information"""
        if bbox is None:
            return
        
        x, y, w, h = [int(v) for v in bbox]
        
        # Choose color based on tracking status
        color = self.GREEN if is_tracking else self.RED
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw tracking status
        status = "TRACKING" if is_tracking else "DETECTED"
        cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw face center point
        center_x, center_y = x + w // 2, y + h // 2
        cv2.circle(frame, (center_x, center_y), 3, self.BLUE, -1)
        
        # Draw tracking time if tracking
        if is_tracking and self.tracking_start_time:
            tracking_time = time.time() - self.tracking_start_time
            cv2.putText(frame, f"Time: {tracking_time:.1f}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.YELLOW, 2)
    
    def reset_tracking(self):
        """Reset the tracking state"""
        self.face_tracked = False
        self.tracker = None
        self.face_bbox = None
        self.tracking_start_time = None
        print("Tracking reset!")
    
    def run(self):
        """Main tracking loop"""
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            
            # Flip the frame horizontally for a more intuitive experience
            frame = cv2.flip(frame, 1)
            
            # Try to update existing tracker
            if self.face_tracked:
                bbox, success = self.update_tracker(frame)
                
                # If tracking fails or we need to re-detect
                if not success or frame_count % self.detection_interval == 0:
                    self.face_tracked = False
                    self.tracker = None
                    print("Tracking lost, re-detecting...")
            
            # Detect faces if not currently tracking
            if not self.face_tracked:
                faces = self.detect_faces(frame)
                
                if len(faces) > 0:
                    # Use the largest face (closest to camera)
                    largest_face = max(faces, key=lambda x: x[2] * x[3])
                    x, y, w, h = largest_face
                    
                    # Initialize tracker with the detected face
                    if self.initialize_tracker(frame, (x, y, w, h)):
                        self.draw_face_info(frame, (x, y, w, h), is_tracking=True)
                    else:
                        self.draw_face_info(frame, (x, y, w, h), is_tracking=False)
                else:
                    # Draw "No face detected" message
                    cv2.putText(frame, "No face detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.RED, 2)
            else:
                # Draw tracked face
                self.draw_face_info(frame, self.face_bbox, is_tracking=True)
            
            # Draw instructions
            cv2.putText(frame, "Press 'q' to quit, 'r' to reset", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show the frame
            cv2.imshow('Face Tracker', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset_tracking()
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Face tracker stopped!")

def main():
    """Main function to run the face tracker"""
    print("Starting Face Tracker...")
    print("Make sure you have a webcam connected!")
    
    tracker = FaceTracker()
    
    try:
        tracker.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if tracker.cap.isOpened():
            tracker.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 