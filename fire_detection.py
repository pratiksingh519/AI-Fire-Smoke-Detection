from ultralytics import YOLO

# 🔥 TEMP MODEL (auto download ho jayega)
model = YOLO("yolov8n.pt")

def detect_fire(frame, conf_thres=0.5):
    results = model(frame, verbose=False)[0]

    detected = False
    best_box = None

    for box in results.boxes:
        conf = float(box.conf[0])

        if conf < conf_thres:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if best_box is None or conf > best_box[4]:
            best_box = (x1, y1, x2, y2, conf)
            detected = True

    return detected, best_box