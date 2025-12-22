# 🔒 Face Anonymizer - Privacy Protection Tool

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**An intelligent face detection and anonymization tool that automatically blurs faces in images, videos, and live webcam feeds to protect privacy**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Examples](#-examples) • [Advanced](#-advanced-options)

</div>

---

## 🌟 Features

- 🎭 **Multi-Mode Support** - Process images, videos, or live webcam feed
- 🤖 **AI-Powered Detection** - Uses MediaPipe for accurate face detection
- ⚡ **Real-Time Processing** - Blur faces in live video streams instantly
- 🎚️ **Adjustable Blur** - Customize blur intensity to your needs
- 📦 **Batch Processing** - Anonymize entire videos frame-by-frame
- 🎯 **High Accuracy** - Detects faces at various angles and lighting conditions
- 💾 **Auto-Save** - Automatically saves processed files to organized outputs
- 🔧 **Easy Configuration** - Simple command-line interface
- 🚀 **Fast Performance** - Optimized for speed and efficiency

---

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Webcam (for webcam mode only)
- Basic understanding of command-line usage

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AlBaraa-1/Computer-vision.git
   cd Computer-vision/face_anonymizer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **You're ready to go!** 🎉

---

## 💻 Usage

### Basic Commands

The Face Anonymizer supports three modes: **image**, **video**, and **webcam**.

#### 1. **Image Mode** (Default)

Blur faces in a single image:

```bash
python face_guard.py --mode image --filePath inputs/me.jpg
```

Or simply (uses default image):
```bash
python face_guard.py
```

#### 2. **Video Mode**

Anonymize all faces in a video file:

```bash
python face_guard.py --mode video --filePath inputs/testVideo.mp4
```

#### 3. **Webcam Mode**

Real-time face blurring from your webcam:

```bash
python face_guard.py --mode webcam
```

Press `q` to stop the webcam feed.

---

## 🎛️ Advanced Options

### Customize Blur Intensity

Control how much blur is applied to detected faces:

```bash
# Light blur (value: 10-20)
python face_guard.py --mode image --filePath inputs/me.jpg --blur 15

# Medium blur (value: 20-40) - Default
python face_guard.py --mode image --filePath inputs/me.jpg --blur 30

# Heavy blur (value: 40-60)
python face_guard.py --mode image --filePath inputs/me.jpg --blur 50
```

**Default Blur Values:**
- Images: `20`
- Videos: `35`
- Webcam: `35`

### Complete Command Structure

```bash
python face_guard.py --mode [image|video|webcam] --filePath [path] --blur [intensity]
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--mode` | string | Processing mode: `image`, `video`, or `webcam` | `image` |
| `--filePath` | string | Path to input file (image or video) | `inputs/me.jpg` |
| `--blur` | integer | Blur intensity (10-60 recommended) | 20 (image), 35 (video/webcam) |

---

## 📁 Project Structure

```
face_anonymizer/
│
├── face_guard.py        # Main anonymization script
├── requirements.txt     # Python dependencies
│
├── inputs/             # Place your input files here
│   ├── me.jpg          # Sample image
│   └── testVideo.mp4   # Sample video
│
└── outputs/            # Processed files (auto-generated)
    ├── output.png      # Anonymized image output
    └── output_video.mp4 # Anonymized video output
```

---

## 🎯 Examples

### Example 1: Image Anonymization

**Command:**
```bash
python face_guard.py --mode image --filePath inputs/me.jpg --blur 25
```

**Result:**
- ✅ Faces detected and blurred
- 💾 Output saved to: `outputs/output.png`

---

### Example 2: Video Anonymization

**Command:**
```bash
python face_guard.py --mode video --filePath inputs/testVideo.mp4 --blur 40
```

**Result:**
- ✅ All faces in video blurred frame-by-frame
- 💾 Output saved to: `outputs/output_video.mp4`
- ⏱️ Progress displayed in console

---

### Example 3: Live Webcam Anonymization

**Command:**
```bash
python face_guard.py --mode webcam --blur 30
```

**Result:**
- ✅ Real-time face detection and blurring
- 👁️ Live preview window displayed
- ⌨️ Press `q` to exit

---

## 🔍 How It Works

### Detection Pipeline

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Input     │ ───> │  MediaPipe  │ ───> │  Face       │ ───> │  Blur &     │
│  Source     │      │  Detection  │      │  Bounding   │      │  Save       │
└─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
```

### Step-by-Step Process

1. **Input Loading**
   - Load image, video, or webcam stream
   - Convert to RGB format for MediaPipe

2. **Face Detection**
   - MediaPipe analyzes each frame
   - Detects faces with confidence scoring
   - Extracts bounding box coordinates

3. **Coordinate Conversion**
   - Convert relative coordinates to pixels
   - Clip boundaries to image dimensions
   - Ensure valid region of interest

4. **Face Blurring**
   - Apply Gaussian blur to face regions
   - Configurable blur intensity
   - Preserve non-face areas

5. **Output Generation**
   - Save processed images as PNG
   - Save processed videos as MP4
   - Display real-time webcam feed

---

## 🛠️ Customization

### Adjusting Detection Sensitivity

Modify the face detection confidence in `face_guard.py`:

```python
with mp_face_detection.FaceDetection(
    model_selection=0,           # 0 for close-range, 1 for far-range
    min_detection_confidence=0.5  # Adjust: 0.3 (lenient) to 0.8 (strict)
) as face_detection:
```

**Model Selection:**
- `0` - Best for faces within 2 meters (default)
- `1` - Best for faces beyond 2 meters

**Confidence Threshold:**
- `0.3` - Detects more faces, may have false positives
- `0.5` - Balanced (default)
- `0.8` - Very strict, only high-confidence detections

---

### Custom Blur Effects

You can modify the blur method in `face_guard.py`:

**Option 1: Gaussian Blur (Smoother)**
```python
img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], 
                                      (blur_amount, blur_amount), 0)
```

**Option 2: Median Blur (Preserves edges)**
```python
img[y1:y2, x1:x2] = cv2.medianBlur(img[y1:y2, x1:x2], blur_amount)
```

**Option 3: Pixelation Effect**
```python
# Shrink face region
small = cv2.resize(img[y1:y2, x1:x2], (10, 10), interpolation=cv2.INTER_LINEAR)
# Scale back up
img[y1:y2, x1:x2] = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
```

**Option 4: Black Box**
```python
img[y1:y2, x1:x2] = (0, 0, 0)  # Black rectangle
```

---

### Adding Visual Indicators

Show detection bounding boxes before blurring:

```python
# Draw rectangle around detected face
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Add confidence score
score = detection.score[0]
cv2.putText(img, f'{score:.2f}', (x1, y1-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```

---

### Custom Output Paths

Modify output paths in `face_guard.py`:

```python
# Custom image output
output_path = os.path.join(output_dir, 'anonymized_me.png')

# Custom video output with timestamp
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_video_path = os.path.join(output_dir, f'anonymized_{timestamp}.mp4')
```

---

## 📊 Performance Tips

For optimal performance:

1. **Image Quality** - Higher resolution images take longer to process
2. **Video Frame Rate** - Consider downscaling videos for faster processing
3. **Blur Amount** - Higher blur values increase processing time slightly
4. **Detection Range** - Use appropriate model selection (0 or 1) for your use case
5. **Hardware** - Better CPU/GPU improves real-time webcam performance

### Speed Optimization

For faster video processing, resize frames:

```python
# Resize for faster processing
scale_factor = 0.5
frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
```

---

## 🎯 Use Cases

This face anonymizer can be used for:

- 🎥 **Content Creation** - Protect bystander privacy in videos
- 📸 **Social Media** - Anonymize faces before posting
- 🏢 **Corporate Use** - Protect employee identity in training videos
- 📰 **Journalism** - Protect source identity in reports
- 🏫 **Education** - Anonymize students in educational content
- 🔒 **Data Privacy** - Comply with GDPR and privacy regulations
- 🎭 **Events** - Process event photos to protect attendee privacy
- 🚔 **Security** - Anonymize CCTV footage for public distribution

---

## 🐛 Troubleshooting

### Common Issues

**Issue:** `No module named 'mediapipe'`
- **Solution:** Install dependencies: `pip install -r requirements.txt`

**Issue:** Webcam not opening
- **Solution:** Ensure no other application is using the webcam
- **Solution:** Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in the code

**Issue:** Poor detection accuracy
- **Solution:** Adjust `min_detection_confidence` (try 0.3 for more detections)
- **Solution:** Use `model_selection=1` for distant faces

**Issue:** Video output has no codec
- **Solution:** Install additional codecs or change fourcc code in the script
- **Solution:** Try `cv2.VideoWriter_fourcc(*'XVID')` instead

**Issue:** Faces not fully blurred
- **Solution:** Increase blur amount: `--blur 50`
- **Solution:** Faces might be at extreme angles; adjust detection sensitivity

**Issue:** Processing is too slow
- **Solution:** Reduce input resolution
- **Solution:** Lower blur amount for marginal speed improvement
- **Solution:** Use a more powerful computer for real-time processing

---

## 📦 Requirements

```
mediapipe == 0.10.14
opencv-python == 4.10.0.84
protobuf >= 4.25.3, < 5.0.0
numpy >= 1.24.0, < 2.0.0
```

**Minimum System Requirements:**
- Python 3.7+
- 4GB RAM (8GB recommended for video processing)
- Webcam (for webcam mode)
- Windows / macOS / Linux

---

## 🔐 Privacy & Ethics

### Important Considerations

⚠️ **Ethical Usage Guidelines:**
- Always obtain consent before processing images/videos of people
- Use for legitimate privacy protection purposes only
- Comply with local privacy laws and regulations (GDPR, CCPA, etc.)
- Do not use to hide criminal activity or bypass security measures
- Be transparent about face anonymization in your content

### Data Protection

This tool:
- ✅ Processes data locally (no cloud uploads)
- ✅ Does not store or transmit face data
- ✅ Does not create face recognition databases
- ✅ Operates entirely offline after installation

---

## 🚀 Future Enhancements

Potential improvements:
- [ ] Multiple anonymization methods (pixelation, emojis, masks)
- [ ] GUI interface for easier use
- [ ] Batch processing for multiple files
- [ ] Selective anonymization (anonymize some faces, not others)
- [ ] Face tracking across video frames
- [ ] Custom blur patterns and effects
- [ ] Performance metrics and progress bars
- [ ] GPU acceleration support

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation
- Share use cases and examples

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) - Google's ML solutions for face detection
- [OpenCV](https://opencv.org/) - Computer Vision library
- [NumPy](https://numpy.org/) - Numerical computing library

---

## ⚖️ Legal Disclaimer

This tool is provided for legitimate privacy protection purposes. Users are responsible for:
- Obtaining necessary permissions before processing media containing people
- Complying with applicable privacy laws and regulations
- Using the tool ethically and responsibly

The developers assume no liability for misuse of this software.

---

## 📧 Contact

**Author:** AlBaraa-1  
**Repository:** [Computer-vision](https://github.com/AlBaraa-1/Computer-vision)

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

**Protect Privacy. Use Responsibly.**

Made with ❤️ and Python

</div>
