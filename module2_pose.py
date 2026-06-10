import cv2
import mediapipe as mp
from ultralytics import YOLO
import time

model = YOLO('yolov8n.pt')

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = 0

# Key joints we care about for crossing intention
IMPORTANT_JOINTS = {
    mp_pose.PoseLandmark.LEFT_SHOULDER: 'L.Shoulder',
    mp_pose.PoseLandmark.RIGHT_SHOULDER: 'R.Shoulder',
    mp_pose.PoseLandmark.LEFT_HIP: 'L.Hip',
    mp_pose.PoseLandmark.RIGHT_HIP: 'R.Hip',
    mp_pose.PoseLandmark.LEFT_KNEE: 'L.Knee',
    mp_pose.PoseLandmark.RIGHT_KNEE: 'R.Knee',
    mp_pose.PoseLandmark.LEFT_ANKLE: 'L.Ankle',
    mp_pose.PoseLandmark.RIGHT_ANKLE: 'R.Ankle',
}

def get_body_orientation(landmarks, frame_width):
    """Estimate if person is facing toward/away from camera based on shoulder width"""
    l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    shoulder_width = abs(l_shoulder.x - r_shoulder.x) * frame_width
    if shoulder_width > 120:
        return "Facing Camera", (0, 255, 0)
    elif shoulder_width > 60:
        return "Side View", (0, 255, 255)
    else:
        return "Facing Away", (0, 100, 255)

print("Starting pose estimation... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # YOLO detection
    results = model(frame, classes=[0], conf=0.5, verbose=False)

    # Pose estimation on full frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb)

    # Draw YOLO boxes
    person_count = 0
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            person_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
            cv2.putText(frame, f'Person {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2)

    # Draw pose skeleton
    if pose_results.pose_landmarks:
        # Full skeleton in subtle style
        mp_draw.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_draw.DrawingSpec(color=(0,200,255), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_draw.DrawingSpec(color=(255,200,0), thickness=2)
        )

        landmarks = pose_results.pose_landmarks.landmark

        # Highlight key joints for intention detection
        for joint, name in IMPORTANT_JOINTS.items():
            lm = landmarks[joint]
            if lm.visibility > 0.5:
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (px, py), 8, (255, 0, 255), -1)

        # Body orientation analysis
        orientation, color = get_body_orientation(landmarks, w)
        cv2.putText(frame, f'Orientation: {orientation}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Head direction (nose position relative to center)
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        nose_x = int(nose.x * w)
        if nose_x < w // 2 - 50:
            head_dir = "Looking Left"
        elif nose_x > w // 2 + 50:
            head_dir = "Looking Right"
        else:
            head_dir = "Looking Forward"
        cv2.putText(frame, f'Head: {head_dir}', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 0.001)
    prev_time = curr_time

    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(frame, f'Persons: {person_count}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(frame, 'CrossSafe - Module 2: Pose Estimation', (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('CrossSafe - Pose Estimation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()