import cv2
import time
from collections import deque
import keyboard
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions
)

# -------------------------
# HandLandmarker setup
# -------------------------
options = HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)
hands = HandLandmarker.create_from_options(options)

# -------------------------
# OpenCV setup
# -------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

gesture_history = deque(maxlen=5)
last_trigger_time = 0 #prevent key spamming
trigger_cooldown = 0.3 #only one key press every 0.3 seconds

# -------------------------
# Helper functions
# -------------------------
def index_finger_up(landmarks):
    return landmarks[8][1] < landmarks[6][1]

def all_fingers_up(landmarks, hand_label="Right"): #detects open palm 
    fingers_up = []  #stores true or false for each finger

#thumb logic
    if hand_label == "Right":
        fingers_up.append(landmarks[4][0] > landmarks[3][0])
    else:
        fingers_up.append(landmarks[4][0] < landmarks[3][0])

    for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        fingers_up.append(landmarks[tip][1] < landmarks[pip][1])

    return all(fingers_up)

def most_common_gesture(history):
    if not history:
        return None
    return max(set(history), key=history.count)

# -------------------------
# Main loop
# -------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp = int(time.time() * 1000)
    results = hands.detect_for_video(mp_image, timestamp)

    left_index_up = False
    right_index_up = False
    left_all_fingers_up = False
    right_all_fingers_up = False

    if results.hand_landmarks and results.handedness:
        for hand_landmarks_list, hand_handedness in zip(
            results.hand_landmarks, results.handedness
        ):
            label = hand_handedness[0].category_name
            landmarks = [
                (lm.x * frame.shape[1], lm.y * frame.shape[0])
                for lm in hand_landmarks_list
            ]

            if label == "Left":
                left_index_up = index_finger_up(landmarks)
                left_all_fingers_up = all_fingers_up(landmarks, label)
            elif label == "Right":
                right_index_up = index_finger_up(landmarks)
                right_all_fingers_up = all_fingers_up(landmarks, label)

            for x, y in landmarks:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # -------------------------
    # Gesture determination
    # -------------------------
    current_gesture = None

    if left_all_fingers_up or right_all_fingers_up:
        current_gesture = "jump"
    elif left_index_up and right_index_up:
        current_gesture = "down"
    elif right_index_up and not right_all_fingers_up:
        current_gesture = "left"
    elif left_index_up and not left_all_fingers_up:
        current_gesture = "right"

    gesture_history.append(current_gesture)
    smoothed_gesture = most_common_gesture(gesture_history)

    # -------------------------
    # Trigger keyboard events
    # -------------------------
    if smoothed_gesture and time.time() - last_trigger_time > trigger_cooldown:
        if smoothed_gesture == "left":
            keyboard.press_and_release("left")
        elif smoothed_gesture == "right":
            keyboard.press_and_release("right")
        elif smoothed_gesture == "down":
            keyboard.press_and_release("down")
        elif smoothed_gesture == "jump":
            keyboard.press_and_release("up")

        last_trigger_time = time.time()

    # -------------------------
    # Gesture Position Guide (VERTICAL ALIGN)
    # -------------------------
    y_start = 40
    line_gap = 30

    cv2.putText(frame, "RIGHT HAND : Index Up = RIGHT MOVE",
                (20, y_start),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(frame, "LEFT HAND  : Index Up = LEFT MOVE",
                (20, y_start + line_gap),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.putText(frame, "RIGHT HANDS OPEN = JUMP",
                (20, y_start + 2 * line_gap),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, "BOTH INDEX UP = SLIDE",
                (20, y_start + 3 * line_gap),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if smoothed_gesture:
        cv2.putText(frame, f"Detected: {smoothed_gesture.upper()}",
                    (20, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)

    cv2.imshow("Subway sufers Game", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
