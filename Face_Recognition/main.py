"""
Real-Time Face Recognition System
A Python application that performs real-time face detection and recognition using OpenCV and face_recognition library.

Author: [AlBaraa63]
Date: October 2025
"""

import cv2
import numpy as np
import face_recognition
import os

# Configuration
CAMERA_INDEX = 0  # Default camera index (change to 1 or 2 if needed)

def load_images_from_folder(folder_path):
    """
    Load reference images from the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing reference images
        
    Returns:
        tuple: (images, class_names) - List of loaded images and their corresponding names
    """
    images = []
    class_names = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            class_names.append(os.path.splitext(filename)[0])
        else:
            print(f"Warning: Unable to load image {filename}")
    return images, class_names

def find_encodings(images):
    """
    Generate face encodings for the provided images.
    
    Args:
        images (list): List of images to process
        
    Returns:
        list: List of face encodings
    """
    encode_list = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rgb)
        if encodings:
            encode_list.append(encodings[0])
        else:
            print("Warning: No face found in image")
    return encode_list

def main():
    """
    Main function that runs the face recognition system.
    Captures video from webcam, detects faces, and performs recognition.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, 'faces')
    images, class_names = load_images_from_folder(folder_path)
    encode_list_known = find_encodings(images)
    print('Encoding Complete')

    cap = cv2.VideoCapture(CAMERA_INDEX)

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Unable to capture video frame")
            break

        img_small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(img_rgb)
        encodings_current_frame = face_recognition.face_encodings(img_rgb, face_locations)

        for encode_face, face_loc in zip(encodings_current_frame, face_locations):
            matches = face_recognition.compare_faces(encode_list_known, encode_face)
            face_dis = face_recognition.face_distance(encode_list_known, encode_face)
            match_index = np.argmin(face_dis)

            if matches[match_index]:
                name = class_names[match_index].upper()

                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = [v * 4 for v in [y1, x2, y2, x1]]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f"{name}", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            else:
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = [v * 4 for v in [y1, x2, y2, x1]]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red rectangle for unknown faces
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Face Recognition', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
