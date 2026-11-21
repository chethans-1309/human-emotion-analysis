import cv2
from deepface import DeepFace
import threading
import time
import sqlite3
from datetime import datetime
import numpy as np
import base64
from io import BytesIO
import os

class EmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.active_sessions = {}
        self.detection_threads = {}

    def process_image(self, image_data, session_id, user_id):
        """Process a single image for emotion detection"""
        try:
            # Convert image data to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {
                    'status': 'error',
                    'message': 'Invalid image data'
                }

            # Convert to RGB for DeepFace
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process image for emotion detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            emotions = {}
            emotion_details = []
            
            for (x, y, w, h) in faces:
                face_roi = rgb_img[y:y+h, x:x+w]
                
                try:
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, silent=True)
                    emotion = result[0]['dominant_emotion']
                    confidence = result[0]['emotion'][emotion]
                    
                    # Count emotions
                    if emotion in emotions:
                        emotions[emotion] += 1
                    else:
                        emotions[emotion] = 1
                    
                    emotion_details.append({
                        'emotion': emotion,
                        'confidence': confidence,
                        'bbox': [int(x), int(y), int(w), int(h)]
                    })
                    
                    # Draw on image
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(img, f"{emotion} ({confidence:.1f}%)", 
                              (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                except Exception as e:
                    print(f"Error in emotion analysis: {e}")
                    # Still draw rectangle for detected face
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(img, "Face Detected", 
                              (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Add info text
            info_text = f"Faces: {len(faces)} | Emotions: {len(emotions)}"
            cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save processed image to file instead of storing in session
            os.makedirs('static/processed', exist_ok=True)
            processed_image_path = f'static/processed/{session_id}_result.jpg'
            cv2.imwrite(processed_image_path, img)
            
            # Save to database
            self._save_image_session_data(session_id, user_id, emotions)
            
            return {
                'status': 'success',
                'session_id': session_id,
                'faces_detected': len(faces),
                'emotions': emotions,
                'emotion_details': emotion_details,
                'processed_image_path': processed_image_path,
                'message': f"Detected {len(faces)} faces with {len(emotions)} emotions"
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Image processing failed: {str(e)}'
            }

    def process_video(self, video_path, session_id, user_id):
        """Process a video file for emotion detection"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {
                    'status': 'error',
                    'message': 'Could not open video file'
                }

            emotions = {}
            total_frames = 0
            processed_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                total_frames += 1
                
                # Process every 10th frame to save time
                if total_frames % 10 != 0:
                    continue
                    
                processed_frames += 1
                
                # Convert to RGB for DeepFace
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in faces:
                    face_roi = rgb_frame[y:y+h, x:x+w]
                    
                    try:
                        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, silent=True)
                        emotion = result[0]['dominant_emotion']
                        
                        if emotion in emotions:
                            emotions[emotion] += 1
                        else:
                            emotions[emotion] = 1
                            
                    except Exception as e:
                        print(f"Error in emotion analysis: {e}")
                        continue
            
            cap.release()
            
            # Save to database
            self._save_video_session_data(session_id, user_id, emotions, processed_frames)
            
            return {
                'status': 'success',
                'session_id': session_id,
                'total_frames': total_frames,
                'processed_frames': processed_frames,
                'emotions': emotions,
                'message': f"Processed {processed_frames} frames from video"
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Video processing failed: {str(e)}'
            }

    def start_live_detection(self, session_id, user_id):
        """Start live emotion detection in a separate thread with OpenCV window"""
        try:
            if session_id in self.active_sessions:
                return {
                    'status': 'error',
                    'message': 'Session already running'
                }

            # Initialize session data
            session_data = {
                'session_id': session_id,
                'user_id': user_id,
                'status': 'starting',
                'emotion_counts': {
                    'happy': 0, 'sad': 0, 'angry': 0, 
                    'surprise': 0, 'fear': 0, 'disgust': 0, 'neutral': 0
                },
                'start_time': datetime.now().isoformat(),
                'last_emotion': None,
                'total_frames': 0
            }
            
            self.active_sessions[session_id] = session_data
            
            # Start detection in a separate thread
            thread = threading.Thread(target=self._run_emotion_detection, args=(session_id,))
            thread.daemon = True
            thread.start()
            
            self.detection_threads[session_id] = thread
            
            return {
                'status': 'success',
                'session_id': session_id,
                'message': 'Emotion detection started in OpenCV window'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to start detection: {str(e)}'
            }

    def _run_emotion_detection(self, session_id):
        """Run the actual emotion detection with OpenCV window"""
        cap = None
        try:
            # Start capturing video (use default camera)
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                self.active_sessions[session_id]['status'] = 'error'
                self.active_sessions[session_id]['error'] = 'Could not open camera'
                return

            self.active_sessions[session_id]['status'] = 'running'
            
            print(f"Starting emotion detection for session {session_id}")
            print("Press 'Q' to stop detection")
            
            while self.active_sessions[session_id]['status'] == 'running':
                # Capture frame-by-frame
                ret, frame = cap.read()
                
                if not ret:
                    print("Failed to grab frame")
                    break

                # Convert frame to grayscale for face detection
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in the frame
                faces = self.face_cascade.detectMultiScale(
                    gray_frame, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )

                emotions_detected = []
                
                for (x, y, w, h) in faces:
                    # Extract the face ROI (Region of Interest)
                    face_roi = frame[y:y + h, x:x + w]
                    
                    try:
                        # Perform emotion analysis on the face ROI
                        # Note: DeepFace expects RGB format but OpenCV uses BGR
                        rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                        result = DeepFace.analyze(rgb_face, actions=['emotion'], enforce_detection=False, silent=True)

                        # Determine the dominant emotion
                        emotion = result[0]['dominant_emotion']
                        confidence = result[0]['emotion'][emotion]
                        
                        emotions_detected.append(emotion)
                        
                        # Update session counts
                        self.active_sessions[session_id]['emotion_counts'][emotion] += 1
                        self.active_sessions[session_id]['last_emotion'] = emotion
                        self.active_sessions[session_id]['total_frames'] += 1

                        # Draw rectangle around face and label with predicted emotion
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, f"{emotion} ({confidence:.1f}%)", 
                                  (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        print(f"Detected emotion: {emotion} with {confidence:.1f}% confidence")
                        
                    except Exception as e:
                        print(f"Error in emotion analysis: {e}")
                        # Still draw rectangle for detected face even if emotion analysis fails
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(frame, "Face Detected", 
                                  (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Display frame info
                info_text = f"Session: {session_id} | Faces: {len(faces)} | Frame: {self.active_sessions[session_id]['total_frames']}"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if emotions_detected:
                    emotion_text = f"Emotions: {', '.join(emotions_detected)}"
                    cv2.putText(frame, emotion_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    emotion_text = "No emotions detected"
                    cv2.putText(frame, emotion_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Display the resulting frame in OpenCV window
                cv2.imshow('Real-time Emotion Detection - Press Q to quit', frame)

                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stopping emotion detection...")
                    break

            # Clean up
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            
            # Save session data
            self._save_session_data(session_id)
            
            # Update session status
            self.active_sessions[session_id]['status'] = 'completed'
            self.active_sessions[session_id]['end_time'] = datetime.now().isoformat()
            
            print(f"Emotion detection completed for session {session_id}")
            
        except Exception as e:
            print(f"Error in emotion detection thread: {e}")
            self.active_sessions[session_id]['status'] = 'error'
            self.active_sessions[session_id]['error'] = str(e)
            if cap:
                cap.release()
            cv2.destroyAllWindows()

    def stop_live_detection(self, session_id):
        """Stop live emotion detection - called when user presses Q"""
        try:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['status'] = 'stopping'
                
                # Wait a bit for the thread to clean up
                time.sleep(2)
                
                # Save session data
                self._save_session_data(session_id)
                
                if session_id in self.active_sessions:
                    session_data = self.active_sessions[session_id].copy()
                    del self.active_sessions[session_id]
                
                if session_id in self.detection_threads:
                    del self.detection_threads[session_id]
                
                return {
                    'status': 'success',
                    'session_id': session_id,
                    'data': session_data
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Session not found'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to stop detection: {str(e)}'
            }

    def get_session_status(self, session_id):
        """Get current status of a detection session"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        else:
            return {'status': 'not_found'}

    def _save_session_data(self, session_id):
        """Save live session data to database"""
        try:
            if session_id not in self.active_sessions:
                return

            session_data = self.active_sessions[session_id]
            
            print(f"Saving session data for {session_id}:")
            print(f"Emotion counts: {session_data['emotion_counts']}")
            print(f"Total frames: {session_data['total_frames']}")
            print(f"Duration: {session_data.get('start_time')} to {session_data.get('end_time', 'ongoing')}")
            
            # Save to database
            conn = sqlite3.connect('instance/database.db')
            cursor = conn.cursor()
            
            user_id = session_data.get('user_id', 1)
            
            for emotion, count in session_data['emotion_counts'].items():
                if count > 0:
                    # Calculate average confidence
                    confidence = min(100, count * 10)
                    
                    # Use consistent timestamp format
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    cursor.execute(
                        'INSERT INTO emotion_predictions (user_id, emotion, confidence, session_id, timestamp) VALUES (?, ?, ?, ?, ?)',
                        (user_id, emotion, confidence, session_id, timestamp)
                    )
            
            conn.commit()
            conn.close()
            
            print("Session data saved successfully")
            
        except Exception as e:
            print(f"Error saving session data: {e}")

    def _save_image_session_data(self, session_id, user_id, emotions):
        """Save image processing results to database"""
        try:
            conn = sqlite3.connect('instance/database.db')
            cursor = conn.cursor()
            
            # Use consistent timestamp format
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            for emotion, count in emotions.items():
                if count > 0:
                    confidence = min(100, count * 20)  # Higher confidence for images
                    
                    cursor.execute(
                        'INSERT INTO emotion_predictions (user_id, emotion, confidence, session_id, timestamp) VALUES (?, ?, ?, ?, ?)',
                        (user_id, emotion, confidence, session_id, timestamp)
                    )
            
            conn.commit()
            conn.close()
            print(f"Image session data saved for {session_id}")
            
        except Exception as e:
            print(f"Error saving image session data: {e}")

    def _save_video_session_data(self, session_id, user_id, emotions, frame_count):
        """Save video processing results to database"""
        try:
            conn = sqlite3.connect('instance/database.db')
            cursor = conn.cursor()
            
            # Use consistent timestamp format
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            for emotion, count in emotions.items():
                if count > 0:
                    confidence = min(100, (count / frame_count) * 100) if frame_count > 0 else 0
                    
                    cursor.execute(
                        'INSERT INTO emotion_predictions (user_id, emotion, confidence, session_id, timestamp) VALUES (?, ?, ?, ?, ?)',
                        (user_id, emotion, confidence, session_id, timestamp)
                    )
            
            conn.commit()
            conn.close()
            print(f"Video session data saved for {session_id}")
            
        except Exception as e:
            print(f"Error saving video session data: {e}")