import cv2
import numpy as np

# 1. Load the image (frame from the camera feed)
img = cv2.imread("data/piano_image_1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Apply thresholding or edge detection to isolate the keys
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)  # Invert since white keys will be black

# Alternatively, you can use Canny edge detection if needed:
# edges = cv2.Canny(gray, 50, 150)

# 3. Find contours of the keys
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 4. Sort contours by area (largest first, to avoid detecting smaller irrelevant objects)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# 5. Store positions of keys
key_positions = {'white': [], 'black': []}  # Store the positions of white and black keys

# 6. Process each contour and classify as white or black key
for contour in contours:
    # Get bounding box for each contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Ignore small contours that aren't keys
    if w * h < 500:  # Adjust this threshold based on the image size and key size
        continue
    
    # Assuming white keys are larger than black keys
    aspect_ratio = float(w) / h
    if aspect_ratio > 1.5:  # This is an arbitrary value, tweak it based on your setup
        # Likely a white key
        key_positions['white'].append((x, y, w, h))
    else:
        # Likely a black key
        key_positions['black'].append((x, y, w, h))

# 7. Optionally, apply perspective correction (homography)
# This is important if the camera is tilted, use known key locations to warp the image
# Define the source and destination points for perspective correction

# Once perspective correction is applied, you can get more accurate bounding boxes.
# cv2.findHomography() and cv2.warpPerspective() can be used here.

# 8. Store the detected keys' positions in a dictionary
# Now you have the key positions saved in key_positions
print("White key positions:", key_positions['white'])
print("Black key positions:", key_positions['black'])

# 9. Draw bounding boxes to verify
for (x, y, w, h) in key_positions['white']:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)
for (x, y, w, h) in key_positions['black']:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)

# Show the result
cv2.imshow("Detected Keys", img)
cv2.waitKey(0)
cv2.destroyAllWindows()