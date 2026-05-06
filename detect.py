import cv2
import torch
from ultralytics import YOLO
import pyttsx3
import time

# 🔊 Voice engine
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# 🔥 Load trained model
model = YOLO("runs/detect/train-5/weights/best.pt")
model.to("cuda")

print("CUDA Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0))

# 🎥 Camera
cap = cv2.VideoCapture(0)

# ⚡ Faster camera settings
cap.set(3, 640)
cap.set(4, 480)

last_alert_time = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera error ❌")
        break

    # 🌞🌙 Works for both DAY + NIGHT
    enhanced = cv2.convertScaleAbs(frame, alpha=1.1, beta=15)

    # 🚀 GPU Prediction
    results = model.predict(
        enhanced,
        imgsz=320,
        conf=0.4,      # balanced for day + night
        device=0,
        verbose=False
    )

    fire_detected = False
    smoke_detected = False

    for box in results[0].boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        conf = float(box.conf[0])

        cls = int(box.cls[0])

        label = model.names[cls]

        # 🔥 FIRE
        if label == "fire":
            fire_detected = True
            color = (0, 0, 255)

        # 💨 SMOKE
        elif label == "smoke":
            smoke_detected = True
            color = (255, 255, 255)

        else:
            continue

        # Draw box
        cv2.rectangle(enhanced, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            enhanced,
            f"{label.upper()} {int(conf * 100)}%",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # 🔊 ALERT SYSTEM
    current_time = time.time()

    # FIRE ALERT
    if fire_detected and current_time - last_alert_time > 5:
        print("🔥 FIRE DETECTED!")
        speak("Warning! Fire detected!")
        last_alert_time = current_time

    # SMOKE ALERT
    elif smoke_detected and current_time - last_alert_time > 5:
        print("💨 SMOKE DETECTED!")
        speak("Warning! Smoke detected!")
        last_alert_time = current_time

    cv2.imshow("🔥 Fire & Smoke Detection", enhanced)

    # ESC to close
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()