import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Threshold to determine if a fingertip is low enough (pressing)
Y_PRESS_THRESHOLD = 0.43  # Normalized (0 = top, 1 = bottom of image)

cap = cv2.VideoCapture("data/Piano1_2.mp4")

tip_ids = {
    4: "Thumb",
    8: "Index",
    12: "Middle",
    16: "Ring",
    20: "Pinky"
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = hand_landmarks.landmark
            pressing_fingers = 0

            for tip_id, name in tip_ids.items():
                tip_y = lm_list[tip_id].y
                is_pressing = tip_y > Y_PRESS_THRESHOLD
                status = "PRESSED" if is_pressing else "UP"

                # Visual
                cx, cy = int(lm_list[tip_id].x * w), int(tip_y * h)
                color = (0, 0, 255) if is_pressing else (0, 255, 0)
                cv2.circle(frame, (cx, cy), 8, color, -1)
                cv2.putText(frame, f"{name}: {tip_y:.2f}", (cx - 40, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if is_pressing:
                    pressing_fingers += 1

            # Display overall status
            overall_status = "PRESSED" if pressing_fingers >= 3 else "NOT PRESSED"
            color = (0, 0, 255) if pressing_fingers >= 3 else (0, 255, 0)
            cv2.putText(frame, overall_status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Y-Press Detection + Logging", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()