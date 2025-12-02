# gui_advanced.py - ULTRA ADVANCED Emotion Detection with Maximum Effects! 🚀✨
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFilter, ImageEnhance
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import math
from collections import deque

print("🚀 Loading Advanced AI Model...")
model = load_model('model.h5')
print("✅ Model Loaded Successfully!")

# Configuration
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emoji_map = {i: f'emojis/{labels[i].lower()}.png' for i in range(7)}

# Enhanced neon colors
emotion_colors = {
    'Angry': '#FF1744', 'Disgust': '#00E676', 'Fear': '#D500F9',
    'Happy': '#FFEA00', 'Neutral': '#B0BEC5', 'Sad': '#00B0FF',
    'Surprise': '#FF6D00'
}

gradient_colors = {
    'Angry': ['#FF1744', '#F50057', '#D50000'],
    'Disgust': ['#00E676', '#00C853', '#1DE9B6'],
    'Fear': ['#D500F9', '#AA00FF', '#9C27B0'],
    'Happy': ['#FFEA00', '#FFD600', '#FFC400'],
    'Neutral': ['#B0BEC5', '#90A4AE', '#78909C'],
    'Sad': ['#00B0FF', '#0091EA', '#2196F3'],
    'Surprise': ['#FF6D00', '#FF9100', '#FF9800']
}

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# History tracking (increased buffer)
emotion_history = deque(maxlen=100)
confidence_history = {emotion: deque([0]*100, maxlen=100) for emotion in labels}
fps_history = deque([0]*60, maxlen=60)
latency_history = deque([0]*60, maxlen=60)

# Smoothing
emotion_smoothing = deque(maxlen=3)

# Performance metrics
frame_count = 0
start_time = time.time()
session_start = time.time()  # Track overall session time
fps = 0
total_frames = 0

# Animation state
anim = {'pulse': 0, 'wave': 0, 'glow': 0, 'rotate': 0, 'particles': []}

# ===================== MAIN WINDOW =====================
root = tk.Tk()
root.title("🌟 ULTRA ADVANCED AI EMOTION ANALYZER 🌟")
root.configure(bg='#050505')
root.state('zoomed')

# Styles
style = ttk.Style()
style.theme_use('clam')
style.configure("Neon.Horizontal.TProgressbar",
                troughcolor='#1a1a1a', background='#00d4ff',
                bordercolor='#050505', thickness=25)

# ===================== ANIMATED HEADER =====================
header = tk.Canvas(root, height=90, bg='#050505', highlightthickness=0)
header.pack(fill='x')

def animate_header():
    header.delete('all')
    
    # Animated wave background
    for i in range(8):
        y = 45 + math.sin(anim['wave'] + i * 0.5) * 8
        intensity = int(40 + abs(math.sin(anim['wave'] + i * 0.5)) * 80)
        color = f'#{intensity:02x}{intensity+60:02x}ff'
        header.create_line(0, y, 2000, y, fill=color, width=2)
    
    # Title with glow effect (shadow layers)
    shadow_colors = ['#003366', '#004488', '#0066aa', '#0088cc']
    for offset, shadow_color in enumerate(shadow_colors, 1):
        header.create_text(960 + offset, 25 + offset,
                          text="🌟 ULTRA ADVANCED AI EMOTION ANALYZER 🌟",
                          font=('Arial Black', 36, 'bold'),
                          fill=shadow_color)
    
    # Main title
    header.create_text(960, 25,
                      text="🌟 ULTRA ADVANCED AI EMOTION ANALYZER 🌟",
                      font=('Arial Black', 36, 'bold'),
                      fill='#00d4ff')
    
    # Subtitle with pulse
    pulse_val = int(200 + abs(math.sin(anim["pulse"])) * 55)
    pulse_color = f'#00ff{pulse_val:02x}'
    header.create_text(960, 65,
                      text="⚡ Deep Learning | Real-Time Metrics | High Performance | 63% Accuracy ⚡",
                      font=('Arial', 12, 'bold'),
                      fill=pulse_color)
    
    anim['wave'] += 0.08
    anim['pulse'] += 0.05

animate_header()

# ===================== MAIN CONTAINER =====================
main = tk.Frame(root, bg='#050505')
main.pack(fill='both', expand=True, padx=20, pady=15)

# ========== LEFT: CAMERA & STATS ==========
left = tk.Frame(main, bg='#050505')
left.pack(side='left', fill='both', expand=True, padx=(0, 10))

# Camera container
cam_frame = tk.Frame(left, bg='#00d4ff', bd=2)
cam_frame.pack(pady=5)

cam_title_bar = tk.Frame(cam_frame, bg='#1a1a1a', height=45)
cam_title_bar.pack(fill='x')
tk.Label(cam_title_bar, text="📹 NEURAL VISION SYSTEM",
         font=('Arial', 16, 'bold'), fg='#00d4ff', bg='#1a1a1a').pack(pady=10)

video_canvas = tk.Canvas(cam_frame, width=800, height=600, bg='#000000', highlightthickness=0)
video_canvas.pack(padx=2, pady=2)

# Stats panel
stats_frame = tk.Frame(left, bg='#1a1a1a', bd=2, relief='ridge')
stats_frame.pack(fill='x', pady=10)

tk.Label(stats_frame, text="📊 REAL-TIME PERFORMANCE METRICS",
         font=('Arial', 13, 'bold'), fg='#00d4ff', bg='#1a1a1a').pack(pady=8)

stats_grid = tk.Frame(stats_frame, bg='#1a1a1a')
stats_grid.pack(pady=10)

stat_data = {}
metrics = [
    ('FPS', '0', '#00d4ff', 'Frames/Sec'),
    ('FACES', '0', '#00ff88', 'Detected'),
    ('LATENCY', '0ms', '#FFD700', 'Inference'),
    ('AVG FPS', '0', '#FF1744', 'Average'),
    ('TOTAL', '0', '#D500F9', 'Frames'),
    ('UPTIME', '0s', '#00B0FF', 'Runtime')
]

for i, (key, val, color, subtitle) in enumerate(metrics):
    frame = tk.Frame(stats_grid, bg='#2a2a2a', bd=1, relief='solid')
    frame.grid(row=i//3, column=i%3, padx=8, pady=6, sticky='ew')
    
    tk.Label(frame, text=key, font=('Arial', 9, 'bold'),
             fg='#888888', bg='#2a2a2a').pack(pady=(6, 0))
    val_label = tk.Label(frame, text=val, font=('Arial', 20, 'bold'),
                         fg=color, bg='#2a2a2a')
    val_label.pack(pady=2)
    tk.Label(frame, text=subtitle, font=('Arial', 8),
             fg='#666666', bg='#2a2a2a').pack(pady=(0, 6))
    stat_data[key] = val_label

for i in range(3):
    stats_grid.columnconfigure(i, weight=1)

# FPS Graph
fps_label = tk.Label(left, text="📈 FPS MONITOR", font=('Arial', 11, 'bold'),
                     fg='#ffffff', bg='#1a1a1a')
fps_label.pack(fill='x', pady=(10, 0))
fps_canvas = tk.Canvas(left, width=800, height=80, bg='#0a0a0a',
                       highlightthickness=1, highlightbackground='#333333')
fps_canvas.pack(pady=5)

# ========== MIDDLE: EMOTION DISPLAY ==========
middle = tk.Frame(main, bg='#050505', width=480)
middle.pack(side='left', fill='both', padx=10)
middle.pack_propagate(False)

emotion_card = tk.Frame(middle, bg='#1a1a1a', bd=3, relief='ridge')
emotion_card.pack(fill='both', expand=True, pady=5)

tk.Label(emotion_card, text="🎭 DETECTED EMOTION",
         font=('Arial', 17, 'bold'), fg='#ffffff', bg='#1a1a1a').pack(pady=12)

# Emoji display with effects
emoji_container = tk.Frame(emotion_card, bg='#0a0a0a', bd=2, relief='solid')
emoji_container.pack(pady=10)

emoji_canvas = tk.Canvas(emoji_container, width=300, height=300, bg='#0a0a0a', highlightthickness=0)
emoji_canvas.pack(padx=5, pady=5)

# Emotion name
emotion_name_canvas = tk.Canvas(emotion_card, width=450, height=90, bg='#1a1a1a', highlightthickness=0)
emotion_name_canvas.pack(pady=8)

# Confidence section
conf_frame = tk.Frame(emotion_card, bg='#1a1a1a')
conf_frame.pack(fill='x', padx=30, pady=12)

tk.Label(conf_frame, text="🎯 CONFIDENCE LEVEL",
         font=('Arial', 12, 'bold'), fg='#888888', bg='#1a1a1a').pack()

conf_canvas = tk.Canvas(conf_frame, width=400, height=35, bg='#0a0a0a', highlightthickness=0)
conf_canvas.pack(pady=10)

conf_percent = tk.Label(conf_frame, text="0.0%",
                        font=('Arial', 34, 'bold'), fg='#00d4ff', bg='#1a1a1a')
conf_percent.pack(pady=5)

conf_trend = tk.Label(conf_frame, text="● STABLE",
                      font=('Arial', 10, 'bold'), fg='#FFD700', bg='#1a1a1a')
conf_trend.pack()

# Emotion timeline
tk.Label(emotion_card, text="📈 EMOTION FLOW (100 Frames)",
         font=('Arial', 11, 'bold'), fg='#ffffff', bg='#1a1a1a').pack(pady=(15, 5))

timeline_canvas = tk.Canvas(emotion_card, width=430, height=100, bg='#0a0a0a',
                           highlightthickness=1, highlightbackground='#333333')
timeline_canvas.pack(pady=8)

# ========== RIGHT: PROBABILITY ANALYSIS ==========
right = tk.Frame(main, bg='#050505')
right.pack(side='right', fill='both', expand=True, padx=(10, 0))

prob_card = tk.Frame(right, bg='#1a1a1a', bd=2, relief='ridge')
prob_card.pack(fill='both', expand=True, pady=5)

tk.Label(prob_card, text="🧠 NEURAL NETWORK ANALYSIS",
         font=('Arial', 16, 'bold'), fg='#ffffff', bg='#1a1a1a').pack(pady=12)

prob_container = tk.Frame(prob_card, bg='#1a1a1a')
prob_container.pack(fill='both', expand=True, padx=20, pady=10)

emotion_widgets = {}

for emotion in labels:
    frame = tk.Frame(prob_container, bg='#1a1a1a')
    frame.pack(fill='x', pady=7)
    
    header = tk.Frame(frame, bg='#1a1a1a')
    header.pack(fill='x')
    
    tk.Label(header, text=emotion.upper(), font=('Arial', 12, 'bold'),
             fg=emotion_colors[emotion], bg='#1a1a1a', width=10, anchor='w').pack(side='left')
    
    val_label = tk.Label(header, text="0.0%", font=('Arial', 11, 'bold'),
                         fg='#888888', bg='#1a1a1a', width=7, anchor='e')
    val_label.pack(side='right')
    
    # Gradient progress bar
    bar_canvas = tk.Canvas(frame, width=450, height=22, bg='#0a0a0a', highlightthickness=0)
    bar_canvas.pack(pady=4)
    
    # Sparkline
    spark_canvas = tk.Canvas(frame, width=450, height=35, bg='#0a0a0a', highlightthickness=0)
    spark_canvas.pack(pady=2)
    
    emotion_widgets[emotion] = {
        'value': val_label,
        'bar': bar_canvas,
        'spark': spark_canvas
    }

# ========== STATUS BAR ==========
status_bar = tk.Frame(root, bg='#1a1a1a', height=50, bd=2, relief='ridge')
status_bar.pack(fill='x', side='bottom')

status_indicator = tk.Label(status_bar, text="🟢 SYSTEM ACTIVE",
                            font=('Arial', 11, 'bold'), fg='#00ff88', bg='#1a1a1a')
status_indicator.pack(side='left', padx=25, pady=12)

model_info = tk.Label(status_bar, text="⚡ CNN Model | Kaggle GPU Trained | 100 Epochs",
                     font=('Arial', 10), fg='#888888', bg='#1a1a1a')
model_info.pack(side='left', padx=15, pady=12)

timestamp = tk.Label(status_bar, text="", font=('Arial', 10),
                    fg='#666666', bg='#1a1a1a')
timestamp.pack(side='right', padx=25, pady=12)

# ===================== WEBCAM =====================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

current_emotion = None
current_confidence = 0
last_confidences = deque(maxlen=5)

# ===================== DRAWING FUNCTIONS =====================

def draw_gradient_bar(canvas, percent, colors):
    """Draw gradient progress bar"""
    canvas.delete('all')
    canvas.create_rectangle(0, 0, 450, 22, fill='#0a0a0a', outline='#333333')
    
    if percent > 0:
        width = int(450 * percent / 100)
        steps = len(colors)
        for i in range(steps):
            x1 = int((i / steps) * width)
            x2 = int(((i + 1) / steps) * width)
            canvas.create_rectangle(x1, 0, x2, 22, fill=colors[i], outline='')

def draw_sparkline(canvas, data, color):
    """Draw sparkline chart"""
    canvas.delete('all')
    if len(data) < 2:
        return
    
    max_val = max(data) if max(data) > 0 else 1
    points = []
    
    for i, val in enumerate(data):
        x = (i / (len(data) - 1)) * 450
        y = 35 - (val / max_val) * 30
        points.extend([x, y])
    
    if len(points) >= 4:
        canvas.create_line(points, fill=color, width=2, smooth=True)
        # Fill area under curve
        fill_points = points + [450, 35, 0, 35]
        canvas.create_polygon(fill_points, fill=color, stipple='gray25', outline='')

def draw_timeline(canvas):
    """Draw emotion timeline"""
    canvas.delete('all')
    if len(emotion_history) < 2:
        return
    
    bar_width = 430 / len(emotion_history)
    for i, idx in enumerate(emotion_history):
        if idx is not None:
            color = emotion_colors[labels[idx]]
            x = i * bar_width
            # Gradient height based on confidence
            height = 60
            canvas.create_rectangle(x, 100 - height, x + bar_width - 1, 100,
                                   fill=color, outline='')

def draw_fps_graph(canvas, data):
    """Draw FPS line graph"""
    canvas.delete('all')
    canvas.create_rectangle(0, 0, 800, 80, fill='#0a0a0a', outline='#333333')
    
    if len(data) < 2:
        return
    
    max_fps = max(data) if max(data) > 0 else 60
    points = []
    
    for i, val in enumerate(data):
        x = (i / (len(data) - 1)) * 800
        y = 80 - (val / max_fps) * 70
        points.extend([x, y])
    
    if len(points) >= 4:
        canvas.create_line(points, fill='#00d4ff', width=2, smooth=True)
        # Target line at 30 FPS
        canvas.create_line(0, 80 - (30/max_fps)*70, 800, 80 - (30/max_fps)*70,
                         fill='#FFD700', dash=(4, 4))

def draw_confidence_bar(canvas, percent, color):
    """Draw animated confidence bar"""
    canvas.delete('all')
    canvas.create_rectangle(0, 0, 400, 35, fill='#0a0a0a', outline='#333333', width=2)
    
    if percent > 0:
        width = int(400 * percent / 100)
        # Gradient effect
        for i in range(0, width, 2):
            intensity = int(255 * (i / 400))
            canvas.create_rectangle(i, 0, i + 2, 35, fill=color, outline='')
        
        # Glow effect
        glow_width = int(width * 1.02)
        canvas.create_rectangle(0, 0, glow_width, 35, outline=color, width=2)

# ===================== UPDATE FUNCTION =====================

def update_frame():
    global frame_count, start_time, fps, current_emotion, current_confidence, total_frames, anim, session_start
    
    ok, frame = cap.read()
    if not ok:
        root.after(10, update_frame)
        return
    
    frame_start = time.time()
    frame_count += 1
    total_frames += 1
    
    # Calculate FPS
    if frame_count % 10 == 0:
        elapsed = time.time() - start_time
        fps = 10 / elapsed
        start_time = time.time()
        fps_history.append(fps)
        stat_data['FPS'].configure(text=f"{fps:.1f}")
        
        avg_fps = sum(fps_history) / len(fps_history)
        stat_data['AVG FPS'].configure(text=f"{avg_fps:.1f}")
    
    # Update counters
    stat_data['TOTAL'].configure(text=str(total_frames))
    uptime = int(time.time() - session_start)
    stat_data['UPTIME'].configure(text=f"{uptime}s")
    timestamp.configure(text=time.strftime("%H:%M:%S"))
    
    # Animate header every 5 frames for performance
    if frame_count % 5 == 0:
        try:
            animate_header()
        except:
            pass  # Skip animation if there's an error
    
    # Process frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(48, 48))
    
    stat_data['FACES'].configure(text=str(len(faces)))
    
    pred_idx = None
    pred_probs = None
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        
        # Extract and predict
        roi = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (48, 48))
        roi_norm = np.expand_dims(np.expand_dims(roi_resized, -1), 0) / 255.0
        
        pred_start = time.time()
        pred = model.predict(roi_norm, verbose=0)[0]
        pred_time = (time.time() - pred_start) * 1000
        
        latency_history.append(pred_time)
        stat_data['LATENCY'].configure(text=f"{pred_time:.0f}ms")
        
        # Smoothing
        pred_idx = int(np.argmax(pred))
        emotion_smoothing.append(pred_idx)
        
        if len(emotion_smoothing) >= 3:
            # Use most common emotion from last 3 frames
            from collections import Counter
            pred_idx = Counter(emotion_smoothing).most_common(1)[0][0]
        
        pred_probs = pred
        current_emotion = labels[pred_idx]
        current_confidence = pred[pred_idx] * 100
        
        last_confidences.append(current_confidence)
        
        # Draw enhanced face box
        color = emotion_colors[current_emotion]
        color_bgr = tuple(int(color[i:i+2], 16) for i in (5, 3, 1))
        
        # Animated thickness
        thickness = int(4 + math.sin(frame_count * 0.15) * 2)
        
        # Multiple border layers
        cv2.rectangle(frame, (x-8, y-8), (x+w+8, y+h+8), color_bgr, 1)
        cv2.rectangle(frame, (x-4, y-4), (x+w+4, y+h+4), color_bgr, thickness)
        
        # Label
        label_text = f"{current_emotion}: {current_confidence:.1f}%"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        
        cv2.rectangle(frame, (x-4, y-45), (x+tw+10, y-5), color_bgr, -1)
        cv2.putText(frame, label_text, (x+3, y-18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Corner markers
        corner = 25
        for (cx, cy) in [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]:
            dx = corner if cx == x else -corner
            dy = corner if cy == y else -corner
            cv2.line(frame, (cx, cy), (cx+dx, cy), color_bgr, 4)
            cv2.line(frame, (cx, cy), (cx, cy+dy), color_bgr, 4)
        
        status_indicator.configure(text="🟢 FACE DETECTED", fg='#00ff88')
    else:
        status_indicator.configure(text="🟡 SCANNING...", fg='#FFD700')
    
    # Display video
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img = img.resize((800, 600))
    
    # Add scanlines
    draw = ImageDraw.Draw(img, 'RGBA')
    for i in range(0, 600, 3):
        draw.line([(0, i), (800, i)], fill=(0, 200, 255, 20))
    
    imgtk = ImageTk.PhotoImage(img)
    video_canvas.create_image(0, 0, anchor='nw', image=imgtk)
    video_canvas.image = imgtk
    
    # Update emotion display
    if pred_idx is not None:
        emotion_history.append(pred_idx)
        for i, prob in enumerate(pred_probs):
            confidence_history[labels[i]].append(prob * 100)
        
        # Display emoji with effects
        try:
            emoji_img = Image.open(emoji_map[pred_idx])
            
            # Subtle rotation
            anim['rotate'] = (anim['rotate'] + 1) % 360
            rotate_amount = math.sin(anim['rotate'] * math.pi / 180) * 3
            emoji_img = emoji_img.rotate(rotate_amount, expand=False)
            
            # Resize and enhance
            emoji_img = emoji_img.resize((280, 280))
            enhancer = ImageEnhance.Brightness(emoji_img)
            emoji_img = enhancer.enhance(1.1)
            
            emoji_tk = ImageTk.PhotoImage(emoji_img)
            emoji_canvas.create_image(150, 150, image=emoji_tk)
            emoji_canvas.image = emoji_tk
        except:
            pass
        
        # Update emotion name with glow
        emotion_name_canvas.delete('all')
        
        # Shadow layers for glow effect
        shadow_colors = ['#1a1a1a', '#2a2a2a', '#3a3a3a']
        for offset, shadow_color in enumerate(shadow_colors, 1):
            emotion_name_canvas.create_text(
                225 + offset, 45 + offset,
                text=current_emotion.upper(),
                font=('Arial Black', 38, 'bold'),
                fill=shadow_color
            )
        
        # Main text
        emotion_name_canvas.create_text(
            225, 45,
            text=current_emotion.upper(),
            font=('Arial Black', 38, 'bold'),
            fill=color
        )
        
        # Update confidence
        draw_confidence_bar(conf_canvas, current_confidence, color)
        conf_percent.configure(text=f"{current_confidence:.1f}%")
        
        # Trend indicator
        if len(last_confidences) >= 2:
            diff = last_confidences[-1] - last_confidences[-2]
            if diff > 5:
                conf_trend.configure(text="▲ RISING", fg='#00ff88')
            elif diff < -5:
                conf_trend.configure(text="▼ FALLING", fg='#FF1744')
            else:
                conf_trend.configure(text="● STABLE", fg='#FFD700')
        
        # Update probability bars
        for i, emotion in enumerate(labels):
            prob = pred_probs[i] * 100
            widgets = emotion_widgets[emotion]
            
            widgets['value'].configure(text=f"{prob:.1f}%")
            draw_gradient_bar(widgets['bar'], prob, gradient_colors[emotion])
            draw_sparkline(widgets['spark'], confidence_history[emotion], emotion_colors[emotion])
        
        # Draw timeline
        draw_timeline(timeline_canvas)
    
    # Draw FPS graph
    draw_fps_graph(fps_canvas, fps_history)
    
    # Schedule next update
    root.after(10, update_frame)

def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Start
print("🎬 Launching Advanced GUI...")
update_frame()
root.mainloop()
