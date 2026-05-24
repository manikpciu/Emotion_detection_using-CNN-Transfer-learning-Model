# -*- coding: utf-8 -*-

import cv2
import numpy as np
import tensorflow as tf
import datetime
import os

# ── Folder setup ─────────────────────────────
os.makedirs('results/screenshots', exist_ok=True)

# ── Models config ─────────────────────────────
MODELS = {
    'CNN': {
        'path': 'models/cnn_realface_v2.keras',
        'size': 48,
        'gray': True
    },
    'MobileNetV2': {
        'path': 'models/mobilenet_best.h5',
        'size': 96,
        'gray': False
    },
    'EfficientNetB0': {
        'path': 'models/efficientnet_best.keras',
        'size': 48,
        'gray': False
    }
}

CLASSES = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']

COLORS = {
    'Anger': (0, 0, 255),
    'Disgust': (0, 140, 0),
    'Fear': (128, 0, 128),
    'Happy': (0, 255, 255),
    'Neutral': (200, 200, 200),
    'Sad': (255, 100, 0),
    'Surprised': (0, 165, 255),
}

# ── Load models safely ───────────────────────
print("⏳ Loading models...")

loaded_models = {}

for name, cfg in MODELS.items():
    path = cfg['path']

    # safety check
    if not os.path.exists(path):
        print(f"❌ Model missing: {path}")
        exit()
    loaded_models[name] = tf.keras.models.load_model(path)
    print(f"✅ {name} loaded")

print("🎉 All models ready!\n")

# ── Face detector ────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ── Prediction ───────────────────────────────
def predict(model, face_img, size, gray=False):
    img = cv2.resize(face_img, (size, size))

    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=(0, -1))
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

    preds = model(img, training=False).numpy()
    idx = np.argmax(preds[0])
    confidence = float(preds[0][idx]) * 100

    return CLASSES[idx], confidence, preds[0]

# ── Webcam ────────────────────────────────────
model_names = list(MODELS.keys())
active_idx = 0

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not found!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("🎥 Webcam started!")
print("1=CNN | 2=MobileNetV2 | 3=EfficientNetB0 | S=Screenshot | Q=Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not received!")
        break

    active_name = model_names[active_idx]
    model = loaded_models[active_name]
    size = MODELS[active_name]['size']
    gray_mode = MODELS[active_name]['gray']

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, 1.1, 5, minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        emotion, conf, probs = predict(model, face, size, gray_mode)
        color = COLORS.get(emotion, (255, 255, 255))

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        label = f"{emotion} {conf:.1f}%"
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # ── Bottom bar ──
    cv2.putText(frame,
                f"Model: {active_name}",
                (20, frame.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2)

    cv2.imshow("Emotion Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):
        active_idx = 0
        print("→ CNN")
    elif key == ord('2'):
        active_idx = 1
        print("→ MobileNetV2")
    elif key == ord('3'):
        active_idx = 2
        print("→ EfficientNetB0")

    elif key == ord('s'):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"results/screenshots/{ts}.png"
        cv2.imwrite(path, frame)
        print("📸 Saved:", path)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Done!")