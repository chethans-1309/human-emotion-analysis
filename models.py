from datetime import datetime

class User:
    def __init__(self, id, username, email, created_at=None):
        self.id = id
        self.username = username
        self.email = email
        self.created_at = created_at or datetime.now()

class EmotionPrediction:
    def __init__(self, id, user_id, emotion, confidence, timestamp=None, session_id=None):
        self.id = id
        self.user_id = user_id
        self.emotion = emotion
        self.confidence = confidence
        self.timestamp = timestamp or datetime.now()
        self.session_id = session_id

class ActivityPrediction:
    def __init__(self, id, user_id, activity, confidence, timestamp=None):
        self.id = id
        self.user_id = user_id
        self.activity = activity
        self.confidence = confidence
        self.timestamp = timestamp or datetime.now()

class MovementPrediction:
    def __init__(self, id, user_id, direction, pitch, yaw, roll, timestamp=None):
        self.id = id
        self.user_id = user_id
        self.direction = direction
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        self.timestamp = timestamp or datetime.now()