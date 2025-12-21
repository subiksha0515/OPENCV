import cv2
import time
from collections import deque
import keyboard
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions

# -------------------------
# HandLandmarker setup
# -------------------------
options = HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),  # Ensure this file exists
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
last_trigger_time = 0
trigger_cooldown = 0.3

# -------------------------
# Helper functions
# -------------------------
def index_finger_up(landmarks):
    return landmarks[8][1] < landmarks[6][1]

def all_fingers_up(landmarks, hand_label="Right"):
    """
    Detect if all fingers are up. Handles left/right hand.
    """
    fingers_up = []

    # Thumb: depends on hand
    if hand_label == "Right":
        fingers_up.append(landmarks[4][0] > landmarks[3][0])
    else:  # Left hand
        fingers_up.append(landmarks[4][0] < landmarks[3][0])

    # Other fingers: index, middle, ring, pinky
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

    # Convert frame to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp = int(time.time() * 1000)
    results = hands.detect_for_video(mp_image, timestamp)

    left_index_up = False
    right_index_up = False
    left_all_fingers_up = False
    right_all_fingers_up = False

    # Process detected hands
    if results.hand_landmarks and results.handedness:
        for hand_landmarks_list, hand_handedness in zip(results.hand_landmarks, results.handedness):
            label = hand_handedness[0].category_name
            landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in hand_landmarks_list]

            if label == "Left":
                left_index_up = index_finger_up(landmarks)
                left_all_fingers_up = all_fingers_up(landmarks, label)
            elif label == "Right":
                right_index_up = index_finger_up(landmarks)
                right_all_fingers_up = all_fingers_up(landmarks, label)

            # Draw landmarks
            for x, y in landmarks:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # -------------------------
    # Gesture determination
    # Priority: jump > down > left/right
    # -------------------------
    current_gesture = None

    # Jump: either hand has all fingers up
    if left_all_fingers_up or right_all_fingers_up:
        current_gesture = "jump"
    # Down/slide: both index fingers up, but not a jump
    elif left_index_up and right_index_up and not (left_all_fingers_up or right_all_fingers_up):
        current_gesture = "down"
    # Left/right: only one index finger up (not all fingers)
    elif left_index_up and not left_all_fingers_up:
        current_gesture = "left"
    elif right_index_up and not right_all_fingers_up:
        current_gesture = "right"

    # Smooth gesture
    gesture_history.append(current_gesture)
    smoothed_gesture = most_common_gesture(gesture_history)

    # -------------------------
    # Trigger keyboard events
    # -------------------------
    if smoothed_gesture and time.time() - last_trigger_time > trigger_cooldown:
        if smoothed_gesture == "left":
            keyboard.press_and_release("left")
            print("↩️ Turn Left")
        elif smoothed_gesture == "right":
            keyboard.press_and_release("right")
            print("↪️ Turn Right")
        elif smoothed_gesture == "down":
            keyboard.press_and_release("down")
            print("⬇️ Slide Down")
        elif smoothed_gesture == "jump":
            keyboard.press_and_release("up")
            print("⬆️ Jump")
        last_trigger_time = time.time()

    cv2.imshow("Temple Run Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
