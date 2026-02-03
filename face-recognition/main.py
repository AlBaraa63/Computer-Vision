"""
Real-Time Face Recognition System with Gradio UI
A Python application that performs real-time face detection and recognition.

Author: [AlBaraa63]
Date: October 2025
"""

import cv2
import numpy as np
import face_recognition
import os
import gradio as gr
from PIL import Image

# Configuration
CAMERA_INDEX = 0

encode_list_known = []
class_names = []

def load_known_faces():
    """Load and encode faces from the 'faces' folder."""
    global encode_list_known, class_names
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, 'faces')
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return "Created 'faces' folder. Add images and reload."
    
    images = []
    names = []
    
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(img_rgb)
            if encodings:
                images.append(encodings[0])
                names.append(os.path.splitext(filename)[0])
    
    encode_list_known = images
    class_names = names
    
    return f"Loaded {len(class_names)} faces"

def recognize_face(image):
    """Process a single image and recognize faces."""
    if image is None:
        return None
    
    # Convert PIL to OpenCV format
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Resize for faster processing
    img_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    
    # Find faces
    face_locations = face_recognition.face_locations(img_rgb)
    encodings_current_frame = face_recognition.face_encodings(img_rgb, face_locations)
    
    # Draw rectangles and names
    for encode_face, face_loc in zip(encodings_current_frame, face_locations):
        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        face_dis = face_recognition.face_distance(encode_list_known, encode_face)
        
        if len(face_dis) > 0 and matches[np.argmin(face_dis)]:
            name = class_names[np.argmin(face_dis)].upper()
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)
        
        y1, x2, y2, x1 = [v * 4 for v in face_loc]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
        cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Load faces at startup
status_msg = load_known_faces()
print(status_msg)

# Create interface
demo = gr.Interface(
    fn=recognize_face,
    inputs=gr.Image(sources="webcam", streaming=True),
    outputs="image",
    live=True,
    title="Face Recognition System",
    description=f"Status: {status_msg}"
)

if __name__ == "__main__":
    demo.launch()