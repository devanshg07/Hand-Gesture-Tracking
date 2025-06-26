import cv2
import numpy as np
import time

class FaceTracker:
    def __init__(self):
        # Initialize the webcam
        self.cap = cv2.VideoCapture(0)
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Load the pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize tracking variables
        self.face_tracked = False
        self.tracker = None
        self.face_bbox = None
        self.tracking_start_time = None
        self.prev_bbox = None
        self.tracking_failures = 0
        self.max_tracking_failures = 5
        self.use_tracking = True  # Flag to enable/disable tracking
        
        # Colors for visualization
        self.GREEN = (0, 255, 0)
        self.RED = (0, 0, 255)
        self.BLUE = (255, 0, 0)
        self.YELLOW = (0, 255, 255)
        
        # Improved tracking parameters
        self.tracking_threshold = 0.2  # Lower threshold for better tracking
        self.detection_interval = 15   # More frequent re-detection
        self.smoothing_factor = 0.7    # Smoothing for bbox updates
        
        print("Enhanced Face Tracker initialized!")
        
        # Check if any trackers are available
        if not (hasattr(cv2, 'TrackerCSRT_create') or hasattr(cv2, 'TrackerKCF_create') or hasattr(cv2, 'TrackerMOSSE_create')):
            print("No trackers available - using detection mode only")
            self.use_tracking = False
        else:
            print("Trackers available - tracking mode enabled")
        
        print("Press 'q' to quit, 'r' to reset tracking")
    
    def detect_faces(self, frame):
        """Enhanced face detection with better parameters"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better detection in varying lighting
        gray = cv2.equalizeHist(gray)
        
        # Use multiple detection parameters for better accuracy
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # Smaller scale factor for more precise detection
            minNeighbors=3,    # Lower threshold for better detection
            minSize=(50, 50),  # Larger minimum size to avoid false positives
            maxSize=(300, 300) # Maximum size constraint
        )
        
        # Filter faces by size and position for better quality
        filtered_faces = []
        for (x, y, w, h) in faces:
            # Check if face is reasonably sized and positioned
            if w > 50 and h > 50 and w < 300 and h < 300:
                # Check if face is not too close to edges
                if x > 20 and y > 20 and x + w < frame.shape[1] - 20 and y + h < frame.shape[0] - 20:
                    filtered_faces.append((x, y, w, h))
        
        return filtered_faces
    
    def initialize_tracker(self, frame, bbox):
        """Initialize the tracker with improved error handling"""
        # Check what trackers are available
        available_trackers = []
        
        # Test for OpenCV 4.x trackers
        if hasattr(cv2, 'TrackerCSRT_create'):
            available_trackers.append('CSRT')
        if hasattr(cv2, 'TrackerKCF_create'):
            available_trackers.append('KCF')
        if hasattr(cv2, 'TrackerMOSSE_create'):
            available_trackers.append('MOSSE')
        
        print(f"Available trackers: {available_trackers}")
        
        # Try available trackers in order of preference
        for tracker_name in available_trackers:
            try:
                if tracker_name == 'CSRT':
                    self.tracker = cv2.TrackerCSRT_create()
                elif tracker_name == 'KCF':
                    self.tracker = cv2.TrackerKCF_create()
                elif tracker_name == 'MOSSE':
                    self.tracker = cv2.TrackerMOSSE_create()
                
                if self.tracker is not None:
                    success = self.tracker.init(frame, bbox)
                    if success:
                        self.face_tracked = True
                        self.face_bbox = bbox
                        self.prev_bbox = bbox
                        self.tracking_start_time = time.time()
                        self.tracking_failures = 0
                        print(f"Face tracking initialized with {tracker_name}!")
                        return True
                    else:
                        print(f"Failed to initialize {tracker_name} tracker")
                else:
                    print(f"Could not create {tracker_name} tracker")
            except Exception as e:
                print(f"Error initializing {tracker_name} tracker: {e}")
                continue
        
        print("No trackers available or all failed to initialize")
        return False
    
    def update_tracker(self, frame):
        """Enhanced tracker update with smoothing and failure handling"""
        if self.tracker is None:
            return None, False
        
        success, bbox = self.tracker.update(frame)
        
        if success and bbox is not None:
            # Apply smoothing to reduce jitter
            if self.prev_bbox is not None:
                smoothed_bbox = []
                for i in range(4):
                    smoothed_val = (self.smoothing_factor * bbox[i] + 
                                  (1 - self.smoothing_factor) * self.prev_bbox[i])
                    smoothed_bbox.append(smoothed_val)
                bbox = tuple(smoothed_bbox)
            
            self.prev_bbox = bbox
            self.face_bbox = bbox
            self.tracking_failures = 0
            return bbox, True
        else:
            self.tracking_failures += 1
            return None, False
    
    def draw_face_info(self, frame, bbox, is_tracking=False):
        """Enhanced visualization with more information"""
        if bbox is None:
            return
        
        x, y, w, h = [int(v) for v in bbox]
        
        # Choose color based on tracking status and confidence
        if is_tracking:
            if self.tracking_failures == 0:
                color = self.GREEN
            elif self.tracking_failures < 3:
                color = self.YELLOW
            else:
                color = self.RED
        else:
            color = self.BLUE
        
        # Draw bounding box with thickness based on confidence
        thickness = 3 if is_tracking else 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        # Draw tracking status
        status = "TRACKING" if is_tracking else "DETECTED"
        if is_tracking and self.tracking_failures > 0:
            status += f" ({self.tracking_failures})"
        
        cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw face center point
        center_x, center_y = x + w // 2, y + h // 2
        cv2.circle(frame, (center_x, center_y), 4, self.BLUE, -1)
        
        # Draw face dimensions
        cv2.putText(frame, f"{w}x{h}", (x, y + h + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
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
        self.prev_bbox = None
        self.tracking_start_time = None
        self.tracking_failures = 0
        print("Tracking reset!")
    
    def run(self):
        """Enhanced main tracking loop"""
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
            if self.face_tracked and self.use_tracking:
                bbox, success = self.update_tracker(frame)
                
                # If tracking fails too many times or we need to re-detect
                if (not success and self.tracking_failures >= self.max_tracking_failures) or \
                   frame_count % self.detection_interval == 0:
                    self.face_tracked = False
                    self.tracker = None
                    print("Tracking lost, re-detecting...")
                elif success:
                    self.draw_face_info(frame, bbox, is_tracking=True)
            
            # Detect faces if not currently tracking or if tracking is disabled
            if not self.face_tracked or not self.use_tracking:
                faces = self.detect_faces(frame)
                
                if len(faces) > 0:
                    # Use the largest face (closest to camera)
                    largest_face = max(faces, key=lambda x: x[2] * x[3])
                    x, y, w, h = largest_face
                    
                    # Initialize tracker with the detected face (if tracking is enabled)
                    if self.use_tracking and self.initialize_tracker(frame, (x, y, w, h)):
                        self.draw_face_info(frame, (x, y, w, h), is_tracking=True)
                    else:
                        # Use detection mode (no tracking)
                        self.face_bbox = (x, y, w, h)
                        self.draw_face_info(frame, (x, y, w, h), is_tracking=False)
                else:
                    # Draw "No face detected" message
                    cv2.putText(frame, "No face detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.RED, 2)
            
            # Draw frame info
            cv2.putText(frame, f"Frame: {frame_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw mode info
            mode = "TRACKING" if self.use_tracking and self.face_tracked else "DETECTION"
            cv2.putText(frame, f"Mode: {mode}", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw instructions
            cv2.putText(frame, "Press 'q' to quit, 'r' to reset", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show the frame
            cv2.imshow('Enhanced Face Tracker', frame)
            
            # Handle key presses and window events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset_tracking()
            
            # Check if window was closed with X button
            if cv2.getWindowProperty('Enhanced Face Tracker', cv2.WND_PROP_VISIBLE) < 1:
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Enhanced face tracker stopped!")

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