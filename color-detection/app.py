import streamlit as st
import cv2
from PIL import Image
from util import get_limits

def main():
    st.title("Yellow Object Detection")
    st.write("This application detects yellow objects using your webcam.")
    
    # Create columns for the buttons
    col1, col2 = st.columns(2)
    
    with col1:
        run = st.button('Start Detection')
    
    with col2:
        stop = st.button('Stop Detection')
    
    # Create a placeholder for the video frame
    frame_window = st.image([])
    
    if run:
        cap = cv2.VideoCapture(0)
        yellow = [0, 255, 255]  # Yellow in BGR

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video")
                break
            
            # Convert frame to HSV
            hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Get color limits for yellow
            lower_limit, upper_limit = get_limits(color=yellow)
            
            # Create mask for yellow color
            mask = cv2.inRange(hsv_image, lower_limit, upper_limit)
            mask = cv2.medianBlur(mask, 3)
            
            # Find contours to detect multiple objects
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Filter small noise
                    x, y, w, h = cv2.boundingRect(contour)
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
            
            # Convert BGR to RGB for Streamlit display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update the image placeholder
            frame_window.image(frame)
        
        cap.release()

if __name__ == "__main__":
    main()