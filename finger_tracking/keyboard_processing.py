import cv2
import numpy as np

def detect_keys(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a binary threshold to separate the keys
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Find contours (potential keys)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    keys = []
    for contour in contours:
        # Approximate the contour to a polygon (e.g., rectangle-like shape)
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the contour is a valid rectangle or polygon with 4 points
        if len(approx) == 4:
            # Perspective correction can be done here for tilted views
            # Detect the position and label the key
            x, y, w, h = cv2.boundingRect(approx)
            keys.append((x, y, w, h))  # Store position (or perform further analysis)
    
    return keys

def main():
    # Capture video or image feed
    cap = cv2.VideoCapture("data/Piano1_1.mp4")  # Camera or video file

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect keys
        keys = detect_keys(frame)

        # Draw rectangles around detected keys
        for (x, y, w, h) in keys:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Show the result
        cv2.imshow("Detected Piano Keys", frame)

        # Break on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()