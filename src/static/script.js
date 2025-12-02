// script.js - Dynamic JavaScript for Real-Time Updates

// Emotion to Emoji mapping
const emotionEmojis = {
    'Angry': '😠',
    'Disgust': '🤢',
    'Fear': '😨',
    'Happy': '😊',
    'Neutral': '😐',
    'Sad': '😢',
    'Surprise': '😮'
};

// Emotion colors
const emotionColors = {
    'Angry': ['#ff1744', '#f50057'],
    'Disgust': ['#00e676', '#00c853'],
    'Fear': ['#d500f9', '#aa00ff'],
    'Happy': ['#ffea00', '#ffd600'],
    'Neutral': ['#b0bec5', '#90a4ae'],
    'Sad': ['#00b0ff', '#0091ea'],
    'Surprise': ['#ff6d00', '#ff9100']
};

// Initialize
let updateInterval;
let historyData = [];

// Start updates when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Initializing Emotion Detection System...');
    initializeProbabilities();
    startUpdates();
});

// Initialize probability bars
function initializeProbabilities() {
    const container = document.getElementById('probabilities');
    const emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'];
    
    emotions.forEach(emotion => {
        const row = document.createElement('div');
        row.className = 'probability-row';
        row.innerHTML = `
            <div class="probability-label" style="color: ${emotionColors[emotion][0]}">${emotion.toUpperCase()}</div>
            <div class="probability-bar-container">
                <div class="probability-bar emotion-${emotion.toLowerCase()}" id="bar-${emotion.toLowerCase()}" style="width: 0%"></div>
            </div>
            <div class="probability-value" id="value-${emotion.toLowerCase()}">0.0%</div>
        `;
        container.appendChild(row);
    });
}

// Start real-time updates
function startUpdates() {
    updateData();
    // Update every 500ms for smooth real-time experience
    updateInterval = setInterval(updateData, 500);
}

// Fetch and update data
async function updateData() {
    try {
        const response = await fetch('/api/predict');
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        const data = await response.json();
        
        // Update emotion display
        if (data.emotions && data.emotions.length > 0) {
            const emotion = data.emotions[0];
            updateEmotionDisplay(emotion);
            updateProbabilities(emotion.probabilities);
            updateStats(data, emotion);
            updateHistory(emotion.emotion);
        } else {
            // No face detected
            document.getElementById('status').textContent = '🟡 SCANNING...';
            document.getElementById('status').style.background = '#ffd700';
        }
        
        // Update face count
        document.getElementById('faces').textContent = data.faces_detected || 0;
        
    } catch (error) {
        console.error('Error fetching data:', error);
        document.getElementById('status').textContent = '🔴 ERROR';
        document.getElementById('status').style.background = '#ff1744';
    }
}

// Update main emotion display
function updateEmotionDisplay(emotion) {
    const emojiElement = document.getElementById('emoji');
    const nameElement = document.getElementById('emotion-name');
    const confidenceFill = document.getElementById('confidence-fill');
    const confidenceText = document.getElementById('confidence-text');
    const statusBadge = document.getElementById('status');
    
    // Update emoji with animation
    emojiElement.style.transform = 'scale(0.8) rotate(-10deg)';
    setTimeout(() => {
        emojiElement.textContent = emotionEmojis[emotion.emotion] || '😐';
        emojiElement.style.transform = 'scale(1) rotate(0deg)';
    }, 150);
    
    // Update emotion name with color
    nameElement.textContent = emotion.emotion.toUpperCase();
    nameElement.style.background = `linear-gradient(135deg, ${emotionColors[emotion.emotion][0]}, ${emotionColors[emotion.emotion][1]})`;
    nameElement.style.webkitBackgroundClip = 'text';
    nameElement.style.webkitTextFillColor = 'transparent';
    
    // Update confidence bar
    const confidence = emotion.confidence.toFixed(1);
    confidenceFill.style.width = `${confidence}%`;
    confidenceFill.style.background = `linear-gradient(90deg, ${emotionColors[emotion.emotion][0]}, ${emotionColors[emotion.emotion][1]})`;
    confidenceText.textContent = `${confidence}%`;
    confidenceText.style.color = emotionColors[emotion.emotion][0];
    
    // Update status
    statusBadge.textContent = '🟢 FACE DETECTED';
    statusBadge.style.background = '#00ff88';
}

// Update probability bars
function updateProbabilities(probabilities) {
    Object.keys(probabilities).forEach(emotion => {
        const value = probabilities[emotion];
        const bar = document.getElementById(`bar-${emotion.toLowerCase()}`);
        const valueDisplay = document.getElementById(`value-${emotion.toLowerCase()}`);
        
        if (bar && valueDisplay) {
            bar.style.width = `${value.toFixed(1)}%`;
            valueDisplay.textContent = `${value.toFixed(1)}%`;
            
            // Highlight if high probability
            if (value > 50) {
                valueDisplay.style.color = emotionColors[emotion][0];
                valueDisplay.style.fontWeight = '700';
            } else {
                valueDisplay.style.color = '#888888';
                valueDisplay.style.fontWeight = '600';
            }
        }
    });
}

// Update statistics
function updateStats(data, emotion) {
    // Latency
    if (emotion && emotion.latency) {
        document.getElementById('latency').textContent = `${Math.round(emotion.latency)}ms`;
    }
    
    // Total predictions
    if (data.stats && data.stats.total_predictions !== undefined) {
        const totalElement = document.getElementById('total-predictions');
        const currentValue = parseInt(totalElement.textContent) || 0;
        const newValue = data.stats.total_predictions;
        
        // Animate count up
        if (newValue > currentValue) {
            animateValue(totalElement, currentValue, newValue, 300);
        }
    }
    
    // Uptime
    if (data.stats && data.stats.uptime !== undefined) {
        const uptime = data.stats.uptime;
        const hours = Math.floor(uptime / 3600);
        const minutes = Math.floor((uptime % 3600) / 60);
        const seconds = uptime % 60;
        
        let uptimeText = '';
        if (hours > 0) {
            uptimeText = `${hours}h ${minutes}m`;
        } else if (minutes > 0) {
            uptimeText = `${minutes}m ${seconds}s`;
        } else {
            uptimeText = `${seconds}s`;
        }
        
        document.getElementById('uptime').textContent = uptimeText;
    }
}

// Update emotion history timeline
function updateHistory(emotion) {
    historyData.push(emotion);
    
    // Keep only last 50 items
    if (historyData.length > 50) {
        historyData.shift();
    }
    
    const timeline = document.getElementById('history-timeline');
    timeline.innerHTML = '';
    
    historyData.forEach((emo, index) => {
        const bar = document.createElement('div');
        bar.className = 'history-bar';
        bar.style.height = `${60 + Math.random() * 20}px`; // Vary height slightly
        bar.style.background = `linear-gradient(180deg, ${emotionColors[emo][0]}, ${emotionColors[emo][1]})`;
        bar.setAttribute('data-emotion', emo);
        
        // Add animation
        bar.style.animation = 'fadeInUp 0.3s ease';
        
        timeline.appendChild(bar);
    });
    
    // Auto-scroll to latest
    timeline.scrollLeft = timeline.scrollWidth;
}

// Animate number counting
function animateValue(element, start, end, duration) {
    const range = end - start;
    const increment = range / (duration / 16); // 60 FPS
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = Math.round(current);
    }, 16);
}

// Handle page visibility change
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        // Page is hidden, stop updates
        clearInterval(updateInterval);
        console.log('⏸️ Updates paused');
    } else {
        // Page is visible, resume updates
        startUpdates();
        console.log('▶️ Updates resumed');
    }
});

// Handle errors
window.addEventListener('error', function(event) {
    console.error('JavaScript Error:', event.error);
});

// Add smooth scroll behavior
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Press 'R' to refresh
    if (e.key === 'r' || e.key === 'R') {
        if (!e.ctrlKey && !e.metaKey) {
            e.preventDefault();
            updateData();
            console.log('🔄 Manual refresh');
        }
    }
    
    // Press 'H' to go home
    if (e.key === 'h' || e.key === 'H') {
        if (!e.ctrlKey && !e.metaKey) {
            e.preventDefault();
            window.location.href = '/';
        }
    }
});

// Performance monitoring
let lastFrameTime = Date.now();
let fps = 0;

function calculateFPS() {
    const now = Date.now();
    const delta = now - lastFrameTime;
    fps = Math.round(1000 / delta);
    lastFrameTime = now;
    requestAnimationFrame(calculateFPS);
}

calculateFPS();

// Log system info
console.log('✅ Emotion Detection System Initialized');
console.log('📊 Real-time updates active');
console.log('⌨️  Keyboard shortcuts: R (refresh), H (home)');
console.log('🎨 Created by Devansh Tyagi');
