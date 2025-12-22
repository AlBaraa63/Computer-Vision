"""Face Anonymizer - Detects and blurs faces in images, videos, or webcam feed."""

import cv2
import mediapipe as mp
import argparse
import os

# Input and output directories
input_dir = os.path.join(os.path.dirname(__file__), 'inputs')
output_dir = os.path.join(os.path.dirname(__file__), 'outputs')

def img_prossess(img, face_detection, blur_amount=30):
    """Detect and blur faces in an image."""
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections:
        for detection in out.detections:
            bbox = detection.location_data.relative_bounding_box
            
            # Convert to pixel coordinates and clip to boundaries
            x1 = max(0, int(bbox.xmin * W))
            y1 = max(0, int(bbox.ymin * H))
            x2 = min(W, int((bbox.xmin + bbox.width) * W))
            y2 = min(H, int((bbox.ymin + bbox.height) * H))
            
            # Blur face region
            if x2 > x1 and y2 > y1:
                img[y1:y2, x1:x2] = cv2.blur(img[y1:y2, x1:x2], (blur_amount, blur_amount))

    return img

# Command-line arguments
parser = argparse.ArgumentParser(description="Face Anonymizer using MediaPipe")
parser.add_argument('--mode', choices=['image', 'video', 'webcam'], default='image', help='Choose the input type')
parser.add_argument('--filePath', help='Path to input image or video file')
parser.add_argument('--blur', type=int, help='Blur intensity (default: 20 for images, 35 for video/webcam)')
args = parser.parse_args()

# Set default file path if none provided
if args.filePath is None:
    args.filePath = os.path.join(input_dir, 'me.jpg')

# Initialize face detection
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    # ========== IMAGE PROCESSING MODE ==========
    if args.mode in ['image']:
        img = cv2.imread(args.filePath)
        blur_amount = args.blur if args.blur else 20  # Default blur for images
        img = img_prossess(img, face_detection, blur_amount)
        
        output_path = os.path.join(output_dir, 'output.png')
        cv2.imwrite(output_path, img)
        print(f"Image saved to: {output_path}")

    # ========== VIDEO PROCESSING MODE ==========
    elif args.mode in ['video']:
        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        output_video_path = os.path.join(output_dir, 'output_video.mp4')
        output_video = cv2.VideoWriter(output_video_path, 
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       25, (frame.shape[1], frame.shape[0]))

        blur_amount = args.blur if args.blur else 35  # Default blur for videos
        while ret:
            frame = img_prossess(frame, face_detection, blur_amount)
            output_video.write(frame)
            ret, frame = cap.read()

        cap.release()
        output_video.release()
        print(f"Video saved to: {output_video_path}")

    # ========== WEBCAM PROCESSING MODE ==========
    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        
        blur_amount = args.blur if args.blur else 35  # Default blur for webcam
        while ret:
            frame = img_prossess(frame, face_detection, blur_amount)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret, frame = cap.read()

        cap.release()