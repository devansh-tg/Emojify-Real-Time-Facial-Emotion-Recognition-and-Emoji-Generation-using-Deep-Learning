# 🌐 Emotion Detection Web Application

## 🚀 Complete Website Implementation

A modern, responsive web application for real-time emotion detection using Deep Learning, Flask, and interactive JavaScript.

---

## ✨ Features

### 🎯 **Core Functionality:**
- ✅ Real-time emotion detection from webcam
- ✅ Live video streaming with emotion overlays
- ✅ 7 emotion categories (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
- ✅ Confidence scores and probability distributions
- ✅ Performance metrics (latency, FPS, uptime)
- ✅ Emotion history timeline

### 🎨 **Frontend Features:**
- ✅ Modern, responsive design (works on all devices)
- ✅ Animated gradients and smooth transitions
- ✅ Real-time data updates without page refresh
- ✅ Interactive probability bars
- ✅ Dynamic emotion history visualization
- ✅ Keyboard shortcuts (R: refresh, H: home)

### 🔧 **Backend Features:**
- ✅ Flask REST API
- ✅ Efficient video streaming
- ✅ Real-time face detection
- ✅ Model prediction caching
- ✅ Statistics tracking
- ✅ Error handling

---

## 📁 Project Structure

```
emojify (copy)/
├── app.py                      # Flask backend server
├── model.h5                    # Trained CNN model
├── launch_website.bat          # Easy launcher
├── requirements_web.txt        # Python dependencies
├── templates/
│   ├── index.html             # Main page
│   └── about.html             # About page
├── static/
│   ├── style.css              # Modern CSS styling
│   └── script.js              # Dynamic JavaScript
└── emojis/                    # Emoji images
```

---

## 🚀 Quick Start

### **Method 1: Double-Click Launch (Easiest)**

1. Navigate to project folder
2. Double-click `launch_website.bat`
3. Wait 10-15 seconds
4. Browser opens automatically at `http://localhost:5000`

### **Method 2: Manual Launch**

```bash
# 1. Activate virtual environment
.\.venv\Scripts\activate

# 2. Install Flask (if not installed)
pip install flask werkzeug

# 3. Run the application
python app.py

# 4. Open browser at:
http://localhost:5000
```

---

## 🎯 How to Use

### **For Presentation:**

1. **Start the Server:**
   - Double-click `launch_website.bat`
   - Wait for "Running on http://localhost:5000" message

2. **Open in Browser:**
   - Automatically opens, or manually go to: `http://localhost:5000`

3. **Allow Camera Access:**
   - Browser will ask for camera permission - click "Allow"

4. **Demo the Features:**
   - **Live Detection:** Show your face, try different expressions
   - **Emotion Display:** Point out the large emoji and confidence percentage
   - **Probability Bars:** Show how all 7 emotions are analyzed
   - **Statistics:** Highlight real-time metrics (latency, predictions)
   - **History:** Show the emotion timeline at bottom

5. **Navigate Pages:**
   - Click "About" to show technical details
   - Use keyboard shortcuts (R to refresh, H for home)

---

## 📊 API Endpoints

### **1. Main Page**
```
GET /
Returns: HTML page with live emotion detection
```

### **2. Video Stream**
```
GET /video_feed
Returns: MJPEG video stream with emotion overlay
```

### **3. Prediction API**
```
GET /api/predict
Returns: JSON with current emotion, probabilities, and stats

Example Response:
{
    "faces_detected": 1,
    "emotions": [{
        "emotion": "Happy",
        "confidence": 87.5,
        "probabilities": {
            "Angry": 2.1,
            "Disgust": 1.3,
            "Fear": 3.2,
            "Happy": 87.5,
            "Neutral": 4.1,
            "Sad": 1.2,
            "Surprise": 0.6
        },
        "bbox": {"x": 120, "y": 80, "w": 200, "h": 200},
        "latency": 28.5
    }],
    "stats": {
        "total_predictions": 1234,
        "avg_latency": 25.3,
        "uptime": 145
    }
}
```

### **4. Statistics API**
```
GET /api/stats
Returns: JSON with system statistics
```

---

## 🎨 UI Components

### **1. Header**
- Logo with animated glow effect
- Navigation menu (Home, About, GitHub)
- Sticky positioning

### **2. Main Grid**
- **Left:** Live camera feed (640x480)
- **Right:** Emotion display with emoji

### **3. Statistics Cards**
- Inference time
- Faces detected
- Total predictions
- System uptime

### **4. Probability Analysis**
- 7 animated gradient bars
- Color-coded by emotion
- Real-time percentage updates

### **5. Emotion Timeline**
- Last 50 emotions displayed
- Color-coded bars
- Hover to see emotion name
- Auto-scrolling

---

## 🎯 Key Features Explained

### **Real-Time Updates:**
```javascript
// JavaScript fetches data every 500ms
setInterval(updateData, 500);
```

### **Smooth Animations:**
```css
/* CSS transitions on all elements */
transition: all 0.3s ease;
```

### **Responsive Design:**
```css
/* Works on mobile, tablet, desktop */
@media (max-width: 768px) { ... }
```

### **Dynamic Visualization:**
```javascript
// Bars animate with new data
bar.style.width = `${confidence}%`;
```

---

## 🛠️ Technology Stack

### **Backend:**
- **Flask:** Web framework
- **TensorFlow/Keras:** Deep learning
- **OpenCV:** Computer vision
- **NumPy:** Numerical operations

### **Frontend:**
- **HTML5:** Structure
- **CSS3:** Styling with animations
- **JavaScript (Vanilla):** Dynamic interactions
- **Google Fonts:** Poppins font family

---

## 🎓 For Professors - Presentation Points

### **Technical Highlights:**

1. **Full-Stack Implementation:**
   - Backend API with Flask
   - Real-time video streaming
   - RESTful architecture
   - Responsive frontend

2. **Performance:**
   - 25-35 FPS video streaming
   - 20-30ms inference time
   - Efficient MJPEG streaming
   - Asynchronous updates

3. **User Experience:**
   - No page reloads needed
   - Smooth animations
   - Mobile-friendly
   - Accessible design

4. **Code Quality:**
   - Clean separation of concerns
   - Error handling
   - Commented code
   - Modular structure

### **Demo Script:**

```
1. "This is a complete web application..."
2. "The backend uses Flask to serve the model..."
3. "Frontend updates in real-time without refresh..."
4. "Works on any device - phone, tablet, desktop..."
5. "All 7 emotions analyzed simultaneously..."
6. "Performance metrics updated live..."
```

---

## 🚨 Troubleshooting

### **Server Won't Start:**
```bash
# Check if port 5000 is in use
netstat -ano | findstr :5000

# Kill process if needed
taskkill /F /PID <process_id>

# Or use different port
python app.py --port 5001
```

### **Camera Not Working:**
- Allow camera permissions in browser
- Close other apps using camera
- Check if camera is connected
- Try different browser

### **Slow Performance:**
- Close unnecessary browser tabs
- Reduce video quality in app.py
- Check CPU usage
- Ensure good lighting

### **Module Not Found:**
```bash
# Reinstall dependencies
pip install -r requirements_web.txt

# Or install individually
pip install flask tensorflow opencv-python
```

---

## 📱 Access from Other Devices

### **Same Network:**
```bash
# Find your IP address
ipconfig  # Windows
ifconfig  # Mac/Linux

# Access from other device
http://YOUR_IP:5000

# Example:
http://192.168.1.100:5000
```

---

## 🔒 Security Notes

- **Local Development Only:** Not configured for production
- **Camera Access:** Only works over HTTPS or localhost
- **CORS:** Not enabled by default
- **For Production:** Add authentication, HTTPS, rate limiting

---

## 🎯 Future Enhancements

- [ ] User authentication
- [ ] Save emotion history to database
- [ ] Export statistics as CSV
- [ ] Multiple face detection
- [ ] Age and gender prediction
- [ ] Emotion heatmap visualization
- [ ] Dark/Light theme toggle
- [ ] Mobile app version

---

## 📊 Performance Benchmarks

| Metric | Value |
|--------|-------|
| Inference Time | 20-30ms |
| Video FPS | 25-35 |
| API Response | <50ms |
| Page Load | <2s |
| Memory Usage | ~500MB |

---

## 🎨 Color Scheme

```css
Primary: #00d4ff (Cyan)
Accent: #00ff88 (Green)
Background: #0a0a0a (Dark)
Cards: #1a1a1a (Gray)
Text: #ffffff (White)
```

---

## 👨‍💻 Developer

**Devansh Tyagi**
- GitHub: [@devansh-tg](https://github.com/devansh-tg)
- Repository: [emojify--copy-](https://github.com/devansh-tg/emojify--copy-)

---

## 📜 License

This project is created for educational purposes.

---

## 🙏 Acknowledgments

- TensorFlow Team
- Flask Framework
- OpenCV Community
- Kaggle for free GPU

---

**🌟 Built with ❤️ using Python, Flask, and Modern Web Technologies**
