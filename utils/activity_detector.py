import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model 
import threading
import time

class ActivityDetector:
    def __init__(self):
        try:
            # Initialize MediaPipe Pose
            self.holistic = mp.solutions.pose
            self.holis = self.holistic.Pose()
            self.drawing = mp.solutions.drawing_utils
            
            # Load model and labels
            self.model = load_model("model.h5")
            self.label = np.load("labels.npy")
            
            self.active_sessions = {}
            self.detection_threads = {}
            
        except Exception as e:
            print(f"Error initializing ActivityDetector: {e}")
            raise

    def inFrame(self, lst):
        """Check if full body is visible in frame"""
        if (lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and 
            lst[15].visibility > 0.6 and lst[16].visibility > 0.6):
            return True 
        return False

    def start_detection(self, session_id):
        """Start activity detection in a separate thread with OpenCV window"""
        try:
            if session_id in self.active_sessions:
                return {
                    'status': 'error',
                    'message': 'Session already running'
                }

            # Initialize session data
            session_data = {
                'session_id': session_id,
                'status': 'starting'
            }
            
            self.active_sessions[session_id] = session_data
            
            # Start detection in a separate thread
            thread = threading.Thread(target=self._run_activity_detection, args=(session_id,))
            thread.daemon = True
            thread.start()
            
            self.detection_threads[session_id] = thread
            
            return {
                'status': 'success',
                'session_id': session_id,
                'message': 'Activity detection started in OpenCV window'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to start detection: {str(e)}'
            }

    def _run_activity_detection(self, session_id):
        """Run the actual activity detection with OpenCV window"""
        cap = None
        try:
            # Start capturing video
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                self.active_sessions[session_id]['status'] = 'error'
                self.active_sessions[session_id]['error'] = 'Could not open camera'
                return

            self.active_sessions[session_id]['status'] = 'running'
            
            print(f"Starting activity detection for session {session_id}")
            print("Press 'Q' to stop detection")
            print("Make sure your full body is visible in the frame")
            
            while self.active_sessions[session_id]['status'] == 'running':
                lst = []

                _, frm = cap.read()

                # Create a window for display
                window = np.zeros((940, 940, 3), dtype="uint8")

                # Flip frame horizontally for mirror effect
                frm = cv2.flip(frm, 1)

                # Process frame with MediaPipe Pose
                res = self.holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

                # Apply slight blur for better visualization
                frm = cv2.blur(frm, (4, 4))
                
                current_activity = "No Activity Detected"
                confidence = 0.0
                activity_color = (0, 0, 255)  # Red for no activity
                
                if res.pose_landmarks and self.inFrame(res.pose_landmarks.landmark):
                    # Extract pose landmarks
                    for i in res.pose_landmarks.landmark:
                        lst.append(i.x - res.pose_landmarks.landmark[0].x)
                        lst.append(i.y - res.pose_landmarks.landmark[0].y)

                    lst = np.array(lst).reshape(1, -1)

                    # Predict activity
                    p = self.model.predict(lst, verbose=0)
                    pred = self.label[np.argmax(p)]
                    confidence = p[0][np.argmax(p)]

                    if confidence > 0.75:
                        current_activity = pred
                        activity_color = (0, 255, 0)  # Green for high confidence
                        # Display activity on frame
                        cv2.putText(frm, f"{pred} ({confidence:.2f})", (50, 80), 
                                   cv2.FONT_ITALIC, 1.3, activity_color, 2)
                    else:
                        current_activity = "Low Confidence"
                        activity_color = (0, 165, 255)  # Orange for low confidence
                        cv2.putText(frm, "Activity: Low Confidence", (50, 80), 
                                   cv2.FONT_ITALIC, 1.0, activity_color, 2)
                        cv2.putText(frm, "Try different pose", (50, 120), 
                                   cv2.FONT_ITALIC, 0.8, activity_color, 2)

                else: 
                    cv2.putText(frm, "Make Sure Full Body Visible", (100, 450), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

                # Draw pose landmarks
                if res.pose_landmarks:
                    self.drawing.draw_landmarks(
                        frm, 
                        res.pose_landmarks, 
                        self.holistic.POSE_CONNECTIONS,
                        connection_drawing_spec=self.drawing.DrawingSpec(
                            color=(255, 255, 255), 
                            thickness=4, 
                            circle_radius=2
                        ),
                        landmark_drawing_spec=self.drawing.DrawingSpec(
                            color=(0, 0, 255), 
                            circle_radius=3, 
                            thickness=3
                        )
                    )

                # Create display window
                window[420:900, 170:810, :] = cv2.resize(frm, (640, 480))
                
                # Add information overlay
                info_y = 50
                cv2.putText(window, "Activity Detection - Press Q to quit", (20, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(window, f"Detected: {current_activity}", (20, info_y + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, activity_color, 2)
                
                cv2.putText(window, f"Confidence: {confidence:.2f}", (20, info_y + 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, activity_color, 2)
                
                cv2.putText(window, "Instructions:", (20, info_y + 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(window, "- Ensure full body is visible", (20, info_y + 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.putText(window, "- Stand in clear view of camera", (20, info_y + 165), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.putText(window, "- Try different yoga poses/activities", (20, info_y + 190), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # Display the window
                cv2.imshow("Activity Detection - Press Q to quit", window)

                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("Stopping activity detection...")
                    break

            # Clean up
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            
            # Update session status
            self.active_sessions[session_id]['status'] = 'completed'
            
            print(f"Activity detection completed for session {session_id}")
            
        except Exception as e:
            print(f"Error in activity detection thread: {e}")
            self.active_sessions[session_id]['status'] = 'error'
            self.active_sessions[session_id]['error'] = str(e)
            if cap:
                cap.release()
            cv2.destroyAllWindows()

    def get_session_status(self, session_id):
        """Get current status of a detection session"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        else:
            return {'status': 'not_found'}