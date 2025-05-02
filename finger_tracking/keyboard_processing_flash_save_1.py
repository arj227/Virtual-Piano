import cv2
import numpy as np

def find_white_keys(frame):
    # Step 1: Preprocess the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # Binary threshold to isolate light parts

    # Step 2: Edge detection
    edges = cv2.Canny(binary, threshold1=100, threshold2=200)  # Detect edges with Canny edge detector

    # Step 3: Find contours (outlines of the keys)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    key_positions = []  # List to store the positions of the white keys
    
    # Step 4: Process each contour
    for contour in contours:
        # Approximate contour to a polygon for smoother shapes
        epsilon = 0.04 * cv2.arcLength(contour, True)  # Approximation accuracy
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Filter out small contours (those that don't represent keys)
        if cv2.contourArea(contour) > 100:
            # Get bounding rectangle for the contour
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)  # Calculate aspect ratio

            # Filter based on aspect ratio (white keys are typically longer than tall)
            if 1.5 < aspect_ratio < 3.0:  # Adjust these values if needed
                key_positions.append((x, y, w, h))  # Store the position as (x, y, width, height)

    # Step 5: Sort the detected keys from left to right by x-coordinate
    key_positions.sort(key=lambda x: x[0])

    return key_positions


def main():
    # Open the video capture (or image)
    cap = cv2.VideoCapture('data/Piano1_1.mp4')  # Replace with your file or use 0 for webcam

    # Process the first frame
    ret, frame = cap.read()

    if ret:
        # Find the white keys in the first frame
        key_positions = find_white_keys(frame)
        print(f"Detected key positions: {key_positions}")

        # Visualize the detected keys (draw rectangles around them)
        for (x, y, w, h) in key_positions:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw red rectangles

        # Show the frame with detected keys
        cv2.imshow('Detected Keys', frame)
        cv2.waitKey(0)  # Wait until a key is pressed
        cv2.destroyAllWindows()

        # Save the positions in a file or dictionary (optional)
        # For example, storing in a dictionary or a list
        key_positions_dict = {i: (x, y, w, h) for i, (x, y, w, h) in enumerate(key_positions)}
        print("Key Positions Dictionary:", key_positions_dict)

    # Release the video capture when done
    cap.release()

if __name__ == "__main__":
    main()