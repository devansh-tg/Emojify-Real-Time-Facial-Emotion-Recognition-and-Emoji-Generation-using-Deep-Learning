import cv2
import numpy as np
from tensorflow.keras.models import load_model

# model load
model = load_model("model.h5")

# emotions aur emojis map
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emoji_paths = {
    'angry': 'emojis/angry.png',
    'disgust': 'emojis/disgust.png',
    'fear': 'emojis/fear.png',
    'happy': 'emojis/happy.png',
    'neutral': 'emojis/neutral.png',
    'sad': 'emojis/sad.png',
    'surprise': 'emojis/surprise.png'
}

# emojis load memory me
emojis = {emo: cv2.imread(path, cv2.IMREAD_UNCHANGED) for emo, path in emoji_paths.items()}

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# function: overlay PNG with transparency
def overlay_image(bg, fg, x, y):
    h, w = fg.shape[:2]
    if y+h > bg.shape[0] or x+w > bg.shape[1]:
        return bg
    
    # Check if image has alpha channel (transparency)
    if fg.shape[2] == 4:
        alpha_fg = fg[:, :, 3] / 255.0
        alpha_bg = 1.0 - alpha_fg
        for c in range(3):
            bg[y:y+h, x:x+w, c] = (alpha_fg * fg[:, :, c] + alpha_bg * bg[y:y+h, x:x+w, c])
    else:
        # No alpha channel, just overlay directly
        bg[y:y+h, x:x+w] = fg[:, :, :3]
    return bg

# webcam open
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48))
    
    for (x, y, w, h) in faces:
        # Extract and preprocess face
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi.reshape(1, 48, 48, 1) / 255.0
        
        # Predict emotion
        prediction = model.predict(face_roi, verbose=0)
        emotion = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Show emotion and confidence
        text = f"{emotion} ({confidence:.1f}%)"
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Overlay emoji
        emoji = emojis[emotion]
        if emoji is not None:
            emoji_resized = cv2.resize(emoji, (100, 100))
            # Position emoji above the face
            emoji_x = max(0, x + w//2 - 50)
            emoji_y = max(0, y - 120)
            frame = overlay_image(frame, emoji_resized, emoji_x, emoji_y)

    cv2.imshow("Emoji Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

