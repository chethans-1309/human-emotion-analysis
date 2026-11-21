from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import sqlite3
import os
from datetime import datetime
import cv2
import numpy as np
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils.emotion_detector import EmotionDetector
from utils.movement_detector import MovementDetector
from utils.activity_detector import ActivityDetector

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['DATABASE'] = 'instance/database.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Initialize detectors
emotion_detector = EmotionDetector()
movement_detector = MovementDetector()
activity_detector = ActivityDetector()

def get_db_connection():
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with app.app_context():
        conn = get_db_connection()
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS emotion_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                emotion TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS activity_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                activity TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS movement_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                direction TEXT NOT NULL,
                pitch REAL NOT NULL,
                yaw REAL NOT NULL,
                roll REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        conn.commit()
        conn.close()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute(
            'SELECT * FROM users WHERE username = ? AND password = ?',
            (username, password)
        ).fetchone()
        conn.close()
        
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials!', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        try:
            conn = get_db_connection()
            conn.execute(
                'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                (username, email, password)
            )
            conn.commit()
            conn.close()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'error')
    
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    conn = get_db_connection()
    
    # Get emotion statistics
    emotion_stats = conn.execute('''
        SELECT emotion, COUNT(*) as count 
        FROM emotion_predictions 
        WHERE user_id = ? 
        GROUP BY emotion 
        ORDER BY count DESC
    ''', (user_id,)).fetchall()
    
    # Get recent predictions
    recent_emotions = conn.execute('''
        SELECT emotion, confidence, timestamp 
        FROM emotion_predictions 
        WHERE user_id = ? 
        ORDER BY timestamp DESC 
        LIMIT 10
    ''', (user_id,)).fetchall()
    
    conn.close()
    
    return render_template('dashboard.html', 
                         emotion_stats=emotion_stats,
                         recent_emotions=recent_emotions)

@app.route('/emotion', methods=['GET', 'POST'])
def emotion():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        detection_type = request.form['detection_type']
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        user_id = session['user_id']
        
        if detection_type == 'image':
            if 'image_file' not in request.files:
                flash('No file selected!', 'error')
                return redirect(request.url)
            
            file = request.files['image_file']
            if file.filename == '':
                flash('No file selected!', 'error')
                return redirect(request.url)
            
            try:
                # Process image
                image_data = file.read()
                results = emotion_detector.process_image(image_data, session_id, user_id)
                
                if results['status'] == 'success':
                    # Store only essential data in session to avoid header size issues
                    session['emotion_results'] = {
                        'session_id': session_id,
                        'faces_detected': results['faces_detected'],
                        'emotions': results['emotions'],
                        'emotion_details': results.get('emotion_details', []),
                        'processed_image_path': results.get('processed_image_path'),
                        'message': results['message']
                    }
                    session['session_id'] = session_id
                    flash('Image processed successfully!', 'success')
                    return redirect(url_for('emotion_result'))
                else:
                    flash(f"Error processing image: {results['message']}", 'error')
                    return redirect(request.url)
                    
            except Exception as e:
                flash(f"Error processing image: {str(e)}", 'error')
                return redirect(request.url)
        
        elif detection_type == 'video':
            if 'video_file' not in request.files:
                flash('No file selected!', 'error')
                return redirect(request.url)
            
            file = request.files['video_file']
            if file.filename == '':
                flash('No file selected!', 'error')
                return redirect(request.url)
            
            try:
                # Save and process video
                import os
                os.makedirs('static/uploads', exist_ok=True)
                video_path = os.path.join('static/uploads', f"{session_id}_{file.filename}")
                file.save(video_path)
                
                results = emotion_detector.process_video(video_path, session_id, user_id)
                
                if results['status'] == 'success':
                    # Store only essential data in session
                    session['emotion_results'] = {
                        'session_id': session_id,
                        'total_frames': results['total_frames'],
                        'processed_frames': results['processed_frames'],
                        'emotions': results['emotions'],
                        'message': results['message']
                    }
                    session['session_id'] = session_id
                    flash('Video processed successfully!', 'success')
                    return redirect(url_for('emotion_result'))
                else:
                    flash(f"Error processing video: {results['message']}", 'error')
                    return redirect(request.url)
                    
            except Exception as e:
                flash(f"Error processing video: {str(e)}", 'error')
                return redirect(request.url)
        
        elif detection_type == 'live':
            session['session_id'] = session_id
            return redirect(url_for('emotion_live', session_id=session_id))
    
    return render_template('emotion.html')

@app.route('/emotion/result')
def emotion_result():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    results = session.get('emotion_results', {})
    session_id = session.get('session_id', '')
    
    # Generate charts
    current_pie_chart = generate_emotion_pie_chart(results.get('emotions', {}))
    overall_line_chart = generate_overall_line_chart(session['user_id'])
    
    return render_template('emotion_result.html',
                         results=results,
                         session_id=session_id,
                         current_pie_chart=current_pie_chart,
                         overall_line_chart=overall_line_chart)

@app.route('/emotion/live/<session_id>')
def emotion_live(session_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    return render_template('emotion_live.html', session_id=session_id)

@app.route('/emotion/start_live')
def start_emotion_live():
    session_id = request.args.get('session_id')
    return emotion_detector.start_live_detection(session_id)



@app.route('/movement')
def movement():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('movement.html')

@app.route('/movement/start', methods=['POST'])
def start_movement_detection():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Not authenticated'}), 401
    
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = movement_detector.start_detection(session_id)
    return jsonify(result)

@app.route('/movement/check_status', methods=['GET'])
def check_movement_status():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Not authenticated'}), 401
    
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({'status': 'error', 'message': 'Session ID required'}), 400
    
    status = movement_detector.get_session_status(session_id)
    return jsonify(status)
    status = movement_detector.get_session_status(session_id)
    return jsonify(status)

@app.route('/activity')
def activity():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('activity.html')

@app.route('/activity/start', methods=['POST'])
def start_activity_detection():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Not authenticated'}), 401
    
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = activity_detector.start_detection(session_id)
    return jsonify(result)

@app.route('/activity/check_status', methods=['GET'])
def check_activity_status():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Not authenticated'}), 401
    
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({'status': 'error', 'message': 'Session ID required'}), 400
    
    status = activity_detector.get_session_status(session_id)
    return jsonify(status)



@app.route('/emotion/start_live_detection', methods=['POST'])
def start_live_emotion_detection():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Not authenticated'}), 401
    
    session_id = request.json.get('session_id')
    if not session_id:
        return jsonify({'status': 'error', 'message': 'Session ID required'}), 400
    
    user_id = session['user_id']
    result = emotion_detector.start_live_detection(session_id, user_id)
    return jsonify(result)

@app.route('/emotion/check_status')
def check_emotion_status():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Not authenticated'}), 401
    
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({'status': 'error', 'message': 'Session ID required'}), 400
    
    status = emotion_detector.get_session_status(session_id)
    return jsonify(status)



@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

# Utility functions
def generate_emotion_pie_chart(results):
    if not results:
        return None
    
    emotions = list(results.keys())
    counts = list(results.values())
    
    plt.figure(figsize=(8, 6))
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff']
    plt.pie(counts, labels=emotions, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Current Session Emotion Distribution')
    
    # Save plot to bytes
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    graphic = base64.b64encode(image_png).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{graphic}"

def generate_overall_line_chart(user_id):
    conn = get_db_connection()
    
    # Get emotion data over time
    emotion_data = conn.execute('''
        SELECT emotion, timestamp, COUNT(*) as count
        FROM emotion_predictions 
        WHERE user_id = ?
        GROUP BY DATE(timestamp), emotion
        ORDER BY timestamp
    ''', (user_id,)).fetchall()
    
    conn.close()
    
    if not emotion_data:
        return None
    
    plt.figure(figsize=(10, 6))
    
    # Prepare data for plotting
    emotions = {}
    for row in emotion_data:
        emotion = row['emotion']
        
        # Handle different timestamp formats
        timestamp_str = row['timestamp']
        try:
            # Try ISO format first (with T and microseconds)
            if 'T' in timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                # Try standard format without T
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            # If both fail, try parsing just the date part
            try:
                timestamp = datetime.strptime(timestamp_str.split(' ')[0], '%Y-%m-%d')
            except:
                # Last resort: use current time
                timestamp = datetime.now()
                print(f"Warning: Could not parse timestamp: {timestamp_str}")
        
        count = row['count']
        
        if emotion not in emotions:
            emotions[emotion] = {'timestamps': [], 'counts': []}
        
        emotions[emotion]['timestamps'].append(timestamp)
        emotions[emotion]['counts'].append(count)
    
    # Plot each emotion
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff']
    
    # Sort emotions by total count for consistent legend order
    sorted_emotions = sorted(emotions.items(), key=lambda x: sum(x[1]['counts']), reverse=True)
    
    for i, (emotion, data) in enumerate(sorted_emotions):
        if i < len(colors):
            # Sort timestamps and counts together
            sorted_data = sorted(zip(data['timestamps'], data['counts']))
            sorted_timestamps, sorted_counts = zip(*sorted_data) if sorted_data else ([], [])
            
            plt.plot(sorted_timestamps, sorted_counts, 
                    label=emotion.capitalize(), color=colors[i], marker='o', linewidth=2, markersize=4)
    
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.title('Emotion Distribution Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
    
    plt.tight_layout()
    
    # Save plot to bytes
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    graphic = base64.b64encode(image_png).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{graphic}"

if __name__ == '__main__':
    init_db()
    app.run(debug=True)