# 🎯 Real-Time Face Recognition 

A Python-based real-time face detection and ide## ⚙️ Configuration

### Camera Settings
- Default camera index is set to `0`
- If you have multiple cameras, modify `CAMERA_INDEX` in `main.py`:
  ```python
  CAMERA_INDEX = 1  # Change to 1, 2, etc. as needed
  ```

### Performance Optimization
- The system resizes frames to 25% for faster processing
- Adjust the resize factor in `main.py` if needed:
  ```python
  img_small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)  # Modify fx and fy values
  ```

## 📝 Dependencies

```
deepface          # Advanced face analysis
gradio           # Web interface framework
opencv-python    # Computer vision library
numpy            # Numerical computing
face_recognition # Face recognition library (auto-installed with deepface)
```

## 🔍 Troubleshooting

### Common Issues

**Camera not working:**
- Check if camera is connected and not being used by another application
- Try different camera indices (0, 1, 2) in the configuration

**Face not recognized:**
- Ensure reference images are clear and well-lit
- Add multiple photos of the same person from different angles
- Check that image files are in the correct format

**Performance issues:**
- Reduce frame size by adjusting the resize factor
- Close other applications using the camera
- Ensure good lighting conditions

**Installation errors:**
- Make sure you have Python 3.7+
- Install Visual Studio Build Tools (Windows) for face_recognition library
- Use `pip install --upgrade pip` before installing requirements

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements:
- Add new features
- Improve performance
- Fix bugs
- Enhance documentation

## 📄 License

This project is open source. Feel free to use and modify as needed.

---

**Enjoy your face recognition system! 🎉**cation system that uses computer vision to recognize faces from a webcam feed. The system compares detected faces against a database of known individuals and provides instant visual feedback.

## ✨ Features

- **Real-Time Face Detection**: Processes live video feeds to detect faces instantly
- **Face Recognition**: Identifies known faces by comparing them against pre-loaded reference images
- **Visual Feedback**: 
  - ✅ **Green boxes** with names for recognized faces
  - ❌ **Red boxes** with "Unknown" label for unrecognized faces
- **Optimized Performance**: Uses frame resizing for faster processing
- **Easy Setup**: Simple folder-based image management

## 📁 Project Structure

```
face_recognition/
├── main.py              # Main application script
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── faces/              # Reference images folder
    ├── albaraa.jpg
    ├── Bill.jpg
    └── elon.jpg
```

## 🔧 Technical Implementation

### Core Functions

- **`load_images_from_folder(folder_path)`**: Loads reference images from the faces directory and extracts names from filenames
- **`find_encodings(images)`**: Converts images to RGB format and generates face encodings using face_recognition library
- **`main()`**: Main execution function that handles video capture, face detection, recognition, and display

### Key Technologies

- **OpenCV**: Video capture, image processing, and display
- **face_recognition**: Face detection and encoding generation
- **NumPy**: Numerical operations and array manipulation
- **DeepFace**: Advanced face analysis capabilities
- **Gradio**: Potential web interface support

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- Webcam connected to your system
- Windows/macOS/Linux

### Installation Steps

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd face_recognition
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare reference images**
   - Add photos of people you want to recognize to the `faces/` folder
   - Use clear, well-lit photos showing the face clearly
   - Name files with the person's name (e.g., `john_doe.jpg`)
   - Supported formats: `.jpg`, `.jpeg`, `.png`

4. **Run the application**
   ```bash
   python main.py
   ```

## 🎮 Usage Instructions

1. **Start the application**: Run `python main.py`
2. **Position yourself**: Ensure good lighting and face the camera
3. **View results**: 
   - Known faces will be highlighted in **green** with their names
   - Unknown faces will be highlighted in **red** labeled as "Unknown"
4. **Exit**: Press `q` to quit the application

**Warnings and Notes:**
- Ensure that your webcam is connected and functioning properly before running the script.
- The script captures video from webcam index 2; you may need to adjust this if you have multiple webcams or if the default index does not work.

Feel free to explore the code and experiment with improvements. If you have any questions or encounter issues, don’t hesitate to reach out!