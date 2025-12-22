import cv2
from PIL import Image
from util import get_limits

# Yellow in BGR color space
yellow = [0, 255, 255]

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to HSV
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get color limits for yellow
    lower_limit, upper_limit = get_limits(color=yellow)

    # Create mask for yellow color
    mask = cv2.inRange(hsv_image, lower_limit, upper_limit)
    mask = cv2.medianBlur(mask, 3, 3)
    mask_img = Image.fromarray(mask)

    # Get bounding box of detected area
    bbox = mask_img.getbbox()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    print(bbox)

    # Show the frame and mask
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()