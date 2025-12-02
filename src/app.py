# app.py - Flask Backend for Emotion Detection Web App
from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64
from collections import deque
import time
import json

app = Flask(__name__)

# Load model
print("🚀 Loading AI Model...")
model = load_model('model.h5')
print("✅ Model Loaded!")

# Configuration
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global variables for stats
emotion_history = deque(maxlen=50)
prediction_times = deque(maxlen=30)
total_predictions = 0
session_start = time.time()

# Camera
camera = None

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

def detect_emotion(frame):
    global total_predictions
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(48, 48))
    
    result = {
        'faces_detected': len(faces),
        'emotions': [],
        'frame_processed': True
    }
    
    if len(faces) > 0:
        for (x, y, w, h) in faces[:1]:  # Process first face only
            # Extract face ROI
            roi = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (48, 48))
            roi_normalized = np.expand_dims(np.expand_dims(roi_resized, -1), 0) / 255.0
            
            # Predict
            start_time = time.time()
            prediction = model.predict(roi_normalized, verbose=0)[0]
            pred_time = (time.time() - start_time) * 1000
            
            prediction_times.append(pred_time)
            total_predictions += 1
            
            # Get results
            emotion_idx = int(np.argmax(prediction))
            emotion = labels[emotion_idx]
            confidence = float(prediction[emotion_idx] * 100)
            
            emotion_history.append(emotion_idx)
            
            # All probabilities
            probabilities = {labels[i]: float(prediction[i] * 100) for i in range(7)}
            
            # Draw on frame
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label_text = f"{emotion}: {confidence:.1f}%"
            cv2.putText(frame, label_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            result['emotions'].append({
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': probabilities,
                'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                'latency': pred_time
            })
    
    return frame, result

def generate_frames():
    """Generate video frames with emotion detection"""
    cam = get_camera()
    
    while True:
        success, frame = cam.read()
        if not success:
            break
        
        # Detect emotion
        processed_frame, _ = detect_emotion(frame)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/predict', methods=['GET'])
def predict():
    """Get current prediction and stats"""
    cam = get_camera()
    success, frame = cam.read()
    
    if not success:
        return jsonify({'error': 'Camera not available'}), 500
    
    _, result = detect_emotion(frame)
    
    # Add statistics
    result['stats'] = {
        'total_predictions': total_predictions,
        'avg_latency': sum(prediction_times) / len(prediction_times) if prediction_times else 0,
        'uptime': int(time.time() - session_start),
        'emotion_history': [labels[i] for i in list(emotion_history)[-10:]]
    }
    
    return jsonify(result)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    return jsonify({
        'total_predictions': total_predictions,
        'avg_latency': sum(prediction_times) / len(prediction_times) if prediction_times else 0,
        'uptime': int(time.time() - session_start),
        'model_accuracy': 63.35,
        'emotions_count': len(labels),
        'recent_emotions': [labels[i] for i in list(emotion_history)[-20:]]
    })

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

if __name__ == '__main__':
    print("🌐 Starting Flask Web Server...")
    print("📱 Open browser at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
