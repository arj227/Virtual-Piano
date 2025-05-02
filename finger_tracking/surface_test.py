import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Start with a value to experiment
Z_PRESS_THRESHOLD = 0.05  # Absolute diff; smaller means "closer" to palm

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = hand_landmarks.landmark
            palm_z = lm_list[0].z  # wrist or palm center

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            tip_ids = [4, 8, 12, 16, 20]
            y_pos = 30
            for tip_id in tip_ids:
                finger_z = lm_list[tip_id].z
                diff = abs(finger_z - palm_z)
                text = f"Tip {tip_id} Z diff: {diff:.3f}"

                color = (0, 0, 255) if diff < Z_PRESS_THRESHOLD else (0, 255, 0)
                cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_pos += 25

            # Optionally label pressing
            pressing_fingers = sum(abs(lm_list[i].z - palm_z) < Z_PRESS_THRESHOLD for i in tip_ids)
            if pressing_fingers >= 3:
                cv2.putText(frame, 'PRESSED', (10, y_pos + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Z-Depth Visualizer", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()