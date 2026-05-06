import cv2
from collections import deque
import winsound
from fire_detection import detect_fire

cap = cv2.VideoCapture(0)

buffer = deque(maxlen=10)
alarm_on = False

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not working ❌")
        break

    detected, box = detect_fire(frame)

    # Draw single box
    if box:
        x1, y1, x2, y2, conf = box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(frame, f"OBJECT {int(conf*100)}%", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # 🔥 temporal logic
    buffer.append(1 if detected else 0)

    if sum(buffer) >= 7:
        if not alarm_on:
            winsound.Beep(1000, 400)
            alarm_on = True

        cv2.putText(frame, "⚠ DETECTION", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    else:
        alarm_on = False

    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()