import cv2
import dlib
import numpy as np
import threading
import time

class MovementDetector:
    def __init__(self):
        try:
            # Initialize dlib's face detector and facial landmark predictor
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            
            # 3D model points for head pose estimation
            self.model_points = np.array([
                (0.0, 0.0, 0.0),           # Nose tip
                (0.0, -330.0, -65.0),      # Chin
                (-225.0, 170.0, -135.0),   # Left eye left corner
                (225.0, 170.0, -135.0),    # Right eye right corner
                (-150.0, -150.0, -125.0),  # Left Mouth corner
                (150.0, -150.0, -125.0)    # Right mouth corner
            ])

            # Camera internals (approximate values)
            self.focal_length = 950
            self.center = (320, 240)
            self.camera_matrix = np.array([
                [self.focal_length, 0, self.center[0]],
                [0, self.focal_length, self.center[1]],
                [0, 0, 1]
            ], dtype=np.float32)

            self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            
            # Calibration variables
            self.calibration_samples = []
            self.is_calibrating = True
            self.calibration_count = 0
            self.MAX_CALIBRATION_SAMPLES = 30
            self.avg_pitch = 0
            self.avg_yaw = 0
            
            self.active_sessions = {}
            self.detection_threads = {}
            
        except Exception as e:
            print(f"Error initializing MovementDetector: {e}")
            raise

    def start_detection(self, session_id):
        """Start head movement detection in a separate thread with OpenCV window"""
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
            
            # Reset calibration for new session
            self.calibration_samples = []
            self.is_calibrating = True
            self.calibration_count = 0
            
            # Start detection in a separate thread
            thread = threading.Thread(target=self._run_movement_detection, args=(session_id,))
            thread.daemon = True
            thread.start()
            
            self.detection_threads[session_id] = thread
            
            return {
                'status': 'success',
                'session_id': session_id,
                'message': 'Head movement detection started in OpenCV window'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to start detection: {str(e)}'
            }

    def _run_movement_detection(self, session_id):
        """Run the actual head movement detection with OpenCV window"""
        cap = None
        try:
            # Start capturing video
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not cap.isOpened():
                self.active_sessions[session_id]['status'] = 'error'
                self.active_sessions[session_id]['error'] = 'Could not open camera'
                return

            self.active_sessions[session_id]['status'] = 'running'
            
            print(f"Starting head movement detection for session {session_id}")
            print("Press 'Q' to stop detection")
            print("Press 'R' to recalibrate")
            print("Look straight at the camera for calibration")
            
            while self.active_sessions[session_id]['status'] == 'running':
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.detector(gray)
                
                for face in faces:
                    try:
                        landmarks = self.predictor(gray, face)
                        
                        # Get head pose angles
                        pose_angles, rotation_vector, translation_vector = self.get_head_pose(landmarks)
                        pitch, yaw, roll = pose_angles
                        
                        # Calibration phase
                        if self.is_calibrating:
                            self.calibration_samples.append((pitch, yaw))
                            self.calibration_count += 1
                            
                            if self.calibration_count >= self.MAX_CALIBRATION_SAMPLES:
                                # Calculate average neutral position
                                self.avg_pitch = np.mean([s[0] for s in self.calibration_samples])
                                self.avg_yaw = np.mean([s[1] for s in self.calibration_samples])
                                print(f"Calibration complete: Pitch={self.avg_pitch:.2f}, Yaw={self.avg_yaw:.2f}")
                                self.is_calibrating = False
                            
                            # Show calibration progress
                            cv2.putText(frame, f"Calibrating... {self.calibration_count}/{self.MAX_CALIBRATION_SAMPLES}", 
                                       (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            direction = "CALIBRATING - LOOK STRAIGHT"
                            color = (0, 255, 255)
                        else:
                            # Use calibrated neutral position
                            calibrated_pitch = pitch - self.avg_pitch
                            calibrated_yaw = yaw - self.avg_yaw
                            
                            # Determine head direction with calibrated values
                            direction, color = self.determine_head_direction(calibrated_pitch, calibrated_yaw, roll)
                        
                        # Draw face rectangle
                        x, y = face.left(), face.top()
                        w, h = face.right() - x, face.bottom() - y
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        
                        # Draw pose axes
                        frame = self.draw_pose_axes(frame, rotation_vector, translation_vector)
                        
                        # Display head direction and angles
                        cv2.putText(frame, f"Direction: {direction}", (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        angle_text = f"Pitch: {pitch:.1f}° | Yaw: {yaw:.1f}° | Roll: {roll:.1f}°"
                        if not self.is_calibrating:
                            angle_text += f" | Calibrated: P{calibrated_pitch:.1f}° Y{calibrated_yaw:.1f}°"
                        
                        cv2.putText(frame, angle_text, (x, y + h + 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Draw direction indicator
                        self.draw_direction_indicator(frame, direction, (100, 100))
                        
                        # Draw nose point for reference
                        nose_point = (landmarks.part(30).x, landmarks.part(30).y)
                        cv2.circle(frame, nose_point, 5, (255, 0, 255), -1)
                        
                    except Exception as e:
                        print(f"Error in pose estimation: {e}")
                        continue

                # Display instructions and legend
                cv2.putText(frame, "Head Movement Detection", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                legend_text = [
                    "Green: Straight | Red: Left/Right | Cyan: Up | Yellow: Down | Purple: Diagonal",
                    "Press 'Q' to quit | Press 'R' to recalibrate"
                ]
                
                for i, text in enumerate(legend_text):
                    cv2.putText(frame, text, (10, 60 + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Display the resulting frame in OpenCV window
                cv2.imshow('Head Movement Detection - Press Q to quit', frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("Stopping head movement detection...")
                    break
                elif key == ord('r') or key == ord('R'):
                    # Reset calibration
                    self.calibration_samples = []
                    self.is_calibrating = True
                    self.calibration_count = 0
                    print("Recalibration started...")

            # Clean up
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            
            # Update session status
            self.active_sessions[session_id]['status'] = 'completed'
            
            print(f"Head movement detection completed for session {session_id}")
            
        except Exception as e:
            print(f"Error in movement detection thread: {e}")
            self.active_sessions[session_id]['status'] = 'error'
            self.active_sessions[session_id]['error'] = str(e)
            if cap:
                cap.release()
            cv2.destroyAllWindows()

    def get_head_pose(self, landmarks):
        """Calculate head pose from facial landmarks"""
        # 2D image points from facial landmarks
        image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),    # Chin
            (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
            (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corner
            (landmarks.part(48).x, landmarks.part(48).y),  # Left Mouth corner
            (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
        ], dtype=np.float32)
        
        # Solve PnP to get rotation and translation vectors
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points, self.camera_matrix, self.dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Calculate Euler angles
        pose_angles = self.rotation_matrix_to_euler_angles(rotation_matrix)
        
        return pose_angles, rotation_vector, translation_vector

    def rotation_matrix_to_euler_angles(self, rotation_matrix):
        """Convert rotation matrix to Euler angles (pitch, yaw, roll)"""
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = 0
        
        # Convert to degrees
        pitch = np.degrees(x)
        yaw = np.degrees(y)
        roll = np.degrees(z)
        
        return pitch, yaw, roll

    def determine_head_direction(self, pitch, yaw, roll):
        """Determine head direction based on pitch and yaw angles"""
        
        # Calibrated thresholds for head direction (in degrees)
        PITCH_THRESHOLD = 10    # For up/down movement
        YAW_THRESHOLD = 10      # For left/right movement
        COMBINED_THRESHOLD = 8  # For diagonal movements
        
        direction = "LOOKING STRAIGHT"
        color = (0, 255, 0)  # Green for straight
        
        # Check for combined diagonal movements first
        if  yaw > YAW_THRESHOLD and pitch < -PITCH_THRESHOLD:
            direction = "LOOKING UP-LEFT"
            color = (255, 0, 255)  # Purple
        elif yaw < -YAW_THRESHOLD  and pitch < -PITCH_THRESHOLD:
            direction = "LOOKING UP-RIGHT" 
            color = (255, 0, 255)  # Purple
        elif yaw > YAW_THRESHOLD  and pitch > PITCH_THRESHOLD:
            direction = "LOOKING DOWN-LEFT"
            color = (255, 0, 255)  # Purple
        elif yaw < -YAW_THRESHOLD  and pitch > PITCH_THRESHOLD:
            direction = "LOOKING DOWN-RIGHT"
            color = (255, 0, 255)  # Purple
        # Check primary directions
        elif yaw > YAW_THRESHOLD:
            direction = "LOOKING LEFT"
            color = (0, 0, 255)  # Red
        elif yaw < -YAW_THRESHOLD :
            direction = "LOOKING RIGHT"
            color = (0, 0, 255)  # Red
        elif pitch < -PITCH_THRESHOLD:
            direction = "LOOKING UP"
            color = (255, 255, 0)  # Cyan
        elif pitch > PITCH_THRESHOLD:
            direction = "LOOKING DOWN"
            color = (0, 255, 255)  # Yellow
        else:
            direction = "LOOKING STRAIGHT"
            color = (0, 255, 0)  # Green
        
        return direction, color

    def draw_pose_axes(self, frame, rotation_vector, translation_vector):
        """Draw 3D axes on the face to visualize head pose"""
        # Points for the axes (in 3D space)
        axis_length = 50
        axis_points = np.float32([
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length],
            [0, 0, 0]
        ]).reshape(-1, 3)
        
        # Project 3D points to 2D image plane
        img_points, _ = cv2.projectPoints(
            axis_points, rotation_vector, translation_vector, 
            self.camera_matrix, self.dist_coeffs
        )
        
        img_points = np.int32(img_points).reshape(-1, 2)
        
        # Draw axes
        nose_point = tuple(np.int32([img_points[3]]).reshape(2))
        
        # X-axis (Red) - Right/Left
        cv2.line(frame, nose_point, tuple(img_points[0]), (0, 0, 255), 3)
        # Y-axis (Green) - Up/Down
        cv2.line(frame, nose_point, tuple(img_points[1]), (0, 255, 0), 3)
        # Z-axis (Blue) - Forward/Backward
        cv2.line(frame, nose_point, tuple(img_points[2]), (255, 0, 0), 3)
        
        return frame

    def draw_direction_indicator(self, frame, direction, position):
        """Draw a visual indicator of the head direction"""
        h, w = frame.shape[:2]
        center_x, center_y = position
        
        # Draw center circle
        cv2.circle(frame, (center_x, center_y), 60, (50, 50, 50), -1)
        cv2.circle(frame, (center_x, center_y), 60, (255, 255, 255), 2)
        
        # Draw direction lines
        if "LEFT" in direction:
            cv2.arrowedLine(frame, (center_x, center_y), (center_x - 40, center_y), (0, 0, 255), 3)
        if "RIGHT" in direction:
            cv2.arrowedLine(frame, (center_x, center_y), (center_x + 40, center_y), (0, 0, 255), 3)
        if "UP" in direction:
            cv2.arrowedLine(frame, (center_x, center_y), (center_x, center_y - 40), (255, 255, 0), 3)
        if "DOWN" in direction:
            cv2.arrowedLine(frame, (center_x, center_y), (center_x, center_y + 40), (0, 255, 255), 3)
        if "STRAIGHT" in direction:
            cv2.circle(frame, (center_x, center_y), 15, (0, 255, 0), -1)

    def get_session_status(self, session_id):
        """Get current status of a detection session"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        else:
            return {'status': 'not_found'}