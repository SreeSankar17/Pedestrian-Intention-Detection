import cv2
from ultralytics import YOLO
import time

# Load YOLOv8 nano - auto downloads on first run (~6MB)
model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = 0

print("Starting detection... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not found!")
        break

    # Run YOLO - only detect persons (class 0)
    results = model(frame, classes=[0], conf=0.5, verbose=False)

    # Draw detections
    person_count = 0
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            person_count += 1

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)

            # Label with confidence
            label = f'Person {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

            # Center point
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 5, (0, 255, 100), -1)

    # FPS counter
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 0.001)
    prev_time = curr_time

    # Info overlay
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(frame, f'Persons: {person_count}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(frame, 'CrossSafe - Module 1: Detection', (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('CrossSafe - Pedestrian Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Session ended. Final FPS: {fps:.1f}")