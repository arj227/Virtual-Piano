import cv2
import numpy as np
import mediapipe as mp

# for sounds creation
import os
import sys
import subprocess
import threading
import cv2
import numpy as np
import mediapipe as mp
import time

last_played = {}
PLAY_COOLDOWN = 1.0

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cwd        = os.path.dirname(os.path.abspath(__file__))
projectRoot= os.path.dirname(cwd)
pianoSoundsDir = os.path.join(projectRoot, 'data', 'piano_sounds')

KEY_MAP = {
    '1': 'A','2': 'B','3': 'C',
    '4': 'D','5': 'E','6': 'F','7': 'G'
}

def play_note(note: str):
    path = os.path.join(pianoSoundsDir, f"{note}.wav")
    if not os.path.isfile(path):
        print(f"Sample not found: {path}")
        return

    def _play():
        if sys.platform == 'darwin':
            # use the macOS command-line player
            subprocess.call(["afplay", path])
        else:
            # fallback if you ever run on Windows/Linux
            from playsound import playsound
            playsound(path)

    threading.Thread(target=_play, daemon=True).start()



# --- Calibration utility ---
def calibrate_piano(cap):
    """
    Let the user click 4 points in this order:
    1) bottom-left, 2) bottom-right, 3) top-right, 4) top-left.
    Returns a list of 4 (x,y) tuples.
    """
    pts = []
    def _on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    win = "CALIBRATE: click BL,BR,TR,TL"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, _on_mouse)

    while len(pts) < 4:
        ret, frame = cap.read()
        if not ret:
            continue
        for i, (x,y) in enumerate(pts):
            cv2.circle(frame, (x,y), 5, (0,0,255), -1)
            cv2.putText(frame, str(i+1), (x+5,y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        if len(pts) > 1:
            cv2.polylines(frame, [np.array(pts)], False, (0,0,255), 1)
        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyWindow(win)
    if len(pts) != 4:
        raise RuntimeError("Calibration aborted or incomplete.")
    return pts

# ——— Main ———
def main():
    cap = cv2.VideoCapture(0)  # live webcam
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

    # 1) CALIBRATE on startup
    calib_pts = calibrate_piano(cap)
    (bl_x, bl_y), (br_x, br_y), (tr_x, tr_y), (tl_x, tl_y) = calib_pts
    LINE_OFFSET = 10
    key_surface_y = int((bl_y + br_y) / 2) + LINE_OFFSET
    piano_poly = np.array(calib_pts, np.int32).reshape((-1,1,2))

    # 2) SPLIT INTO 7 KEY ZONES
    bl = np.array([bl_x, bl_y], dtype=np.float32)
    br = np.array([br_x, br_y], dtype=np.float32)
    tr = np.array([tr_x, tr_y], dtype=np.float32)
    tl = np.array([tl_x, tl_y], dtype=np.float32)

    key_zones = []
    for i in range(7):
        t0, t1 = i/7.0, (i+1)/7.0
        b0 = bl + (br - bl) * t0
        b1 = bl + (br - bl) * t1
        t0p = tl + (tr - tl) * t0
        t1p = tl + (tr - tl) * t1
        zone = np.array([b0, b1, t1p, t0p], dtype=np.int32)
        key_zones.append(zone.reshape((-1,1,2)))

    # 3) Which fingertip landmarks to track (no thumb)
    tip_ids = {
        8:  "Index",
        12: "Middle",
        16: "Ring",
        20: "Pinky"
    }

    active_zones = set()  # zones currently “held down”

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # draw overall piano boundary & key line
        cv2.polylines(frame, [piano_poly], True, (255,0,0), 2)
        cv2.line(frame, (0, key_surface_y), (w, key_surface_y), (255,0,0), 2)

        # draw key zones & labels
        for idx, kz in enumerate(key_zones, start=1):
            cv2.polylines(frame, [kz], True, (200,200,200), 1)
            M = cv2.moments(kz)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                cv2.putText(frame, str(idx), (cx-5, cy+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        # detect presses this frame
        current_pressed = set()
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                for tip_id, name in tip_ids.items():
                    lm = hand.landmark[tip_id]
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    # determine which zone finger is in
                    zone_hit = None
                    for zi, kz in enumerate(key_zones, start=1):
                        if cv2.pointPolygonTest(kz, (cx,cy), False) >= 0:
                            zone_hit = zi
                            break

                    # pressed if inside zone AND below key surface
                    pressed = (zone_hit is not None) and (cy > key_surface_y)
                    if pressed:
                        current_pressed.add(zone_hit)

                    # draw fingertip dot
                    color = (0,0,255) if pressed else (0,255,0)
                    cv2.circle(frame, (cx,cy), 8, color, -1)

                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # update active_zones: add new, remove released
        for z in current_pressed:
            if z not in active_zones and (now - last_played.get(z, 0) >= PLAY_COOLDOWN):
                active_zones.add(z)
                last_played[z] = now
                note = KEY_MAP.get(str(z))
                if note:
                    play_note(note)

        for z in list(active_zones):
            if z not in current_pressed:
                active_zones.remove(z)

        # display active keys at top
        if active_zones:
            keys_str = ", ".join(f"K{z}" for z in sorted(active_zones))
            cv2.putText(frame, f"Pressed: {keys_str}", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Calibrated 7-Key Detection", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
