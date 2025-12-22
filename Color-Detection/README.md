# 🎨 Real-Time Color Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-green.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.26-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**An intelligent real-time color detection system that tracks and highlights specific colors in live video streams**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [How It Works](#-how-it-works) • [Customization](#-customization)

</div>

---

## 🌟 Features

- 🎥 **Real-Time Processing** - Live color detection from webcam feed
- 🎯 **High Accuracy** - Uses HSV color space for robust detection under varying lighting
- 📦 **Bounding Box Detection** - Automatically draws rectangles around detected colors
- 🖼️ **Dual Display** - Shows both original frame and color mask simultaneously
- 🎨 **Customizable Colors** - Easy configuration for any color detection
- ⚡ **Optimized Performance** - Smooth real-time processing with minimal lag
- 🔧 **Modular Design** - Clean, extensible code architecture
- 🎛️ **Adjustable Sensitivity** - Fine-tune detection range and thresholds

---

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Webcam or video capture device
- Basic understanding of BGR color values (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AlBaraa-1/Computer-vision.git
   cd Computer-vision/color_detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

That's it! Your color detection system is now running. 🎉

---

## 💻 Usage

### Basic Usage

Simply run the main script:

```bash
python main.py
```

**Controls:**
- Press `q` to quit the application
- The system automatically detects and tracks yellow objects by default

### What You'll See

The application opens two windows:
1. **Frame Window** - Shows the live webcam feed with green bounding boxes around detected colors
2. **Mask Window** - Shows the color mask (white = detected color, black = everything else)

---

## 📁 Project Structure

```
color_detection/
│
├── main.py              # Main application with webcam capture & detection
├── util.py              # Utility functions for HSV color limit calculation
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

### File Descriptions

| File | Purpose |
|------|---------|
| `main.py` | Core application handling video capture, color detection, and visualization |
| `util.py` | Helper function to calculate HSV color range limits |
| `requirements.txt` | Project dependencies (OpenCV, NumPy, Pillow) |

---

## 🔍 How It Works

### Detection Pipeline

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  Webcam     │ ───> │  BGR to     │ ───> │  Color      │ ───> │  Bounding   │
│  Capture    │      │  HSV        │      │  Masking    │      │  Box Draw   │
└─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
```

### Step-by-Step Process

1. **Video Capture**
   - Captures live video frames from your webcam
   - Processes each frame in real-time

2. **Color Space Conversion**
   - Converts BGR (OpenCV default) to HSV color space
   - HSV is more robust for color detection under different lighting conditions

3. **HSV Color Masking**
   - Calculates upper and lower HSV limits for the target color
   - Creates a binary mask isolating pixels within the color range
   - Applies median blur to reduce noise

4. **Bounding Box Detection**
   - Identifies the bounding box of the detected color region
   - Draws a green rectangle around the detected area

5. **Display**
   - Shows original frame with bounding boxes
   - Shows the binary mask for visualization

### Why HSV Instead of BGR?

HSV (Hue, Saturation, Value) is superior for color detection because:
- **Hue** represents the actual color (0-180°)
- **Saturation** represents color intensity (0-255)
- **Value** represents brightness (0-255)

This separation makes it easier to detect colors regardless of lighting conditions.

---

## 🎨 Customization

### Detecting Different Colors

To detect a different color, modify the `yellow` variable in `main.py`:

```python
# Yellow (default)
yellow = [0, 255, 255]  # BGR format

# Red
red = [0, 0, 255]

# Blue  
blue = [255, 0, 0]

# Green
green = [0, 255, 0]

# Orange
orange = [0, 165, 255]

# Purple
purple = [255, 0, 255]
```

### Common BGR Color Values

| Color | BGR Values |
|-------|------------|
| Yellow | `[0, 255, 255]` |
| Red | `[0, 0, 255]` |
| Blue | `[255, 0, 0]` |
| Green | `[0, 255, 0]` |
| Orange | `[0, 165, 255]` |
| Purple | `[255, 0, 255]` |
| Cyan | `[255, 255, 0]` |
| White | `[255, 255, 255]` |
| Black | `[0, 0, 0]` |

### Adjusting Detection Sensitivity

Modify the `get_limits()` function in `util.py`:

```python
def get_limits(color):
    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    # Adjust these values for sensitivity
    hue_range = 10        # ± Hue tolerance (0-180)
    sat_min = 100         # Minimum saturation (0-255)
    val_min = 100         # Minimum value/brightness (0-255)

    lowerLimit = hsvC[0][0][0] - hue_range, sat_min, val_min
    upperLimit = hsvC[0][0][0] + hue_range, 255, 255

    lowerLimit = np.array(lowerLimit, dtype=np.uint8)
    upperLimit = np.array(upperLimit, dtype=np.uint8)

    return lowerLimit, upperLimit
```

**Sensitivity Tips:**
- **Increase `hue_range`** - Detect wider range of similar colors (less strict)
- **Decrease `hue_range`** - Detect more precise color matches (more strict)
- **Lower `sat_min`** - Detect washed-out/pale colors
- **Lower `val_min`** - Detect colors in darker lighting

### Customizing Bounding Box

Change the rectangle color and thickness in `main.py`:

```python
# Change (0, 255, 0) to any BGR color
# Change 5 to adjust thickness
frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

# Examples:
# Red box:    (0, 0, 255)
# Blue box:   (255, 0, 0)
# Yellow box: (0, 255, 255)
```

### Multi-Color Detection

To detect multiple colors simultaneously, you can extend the code:

```python
# Define multiple colors
colors = {
    'yellow': [0, 255, 255],
    'red': [0, 0, 255],
    'blue': [255, 0, 0]
}

# Process each color
for color_name, color_value in colors.items():
    lower_limit, upper_limit = get_limits(color=color_value)
    mask = cv2.inRange(hsv_image, lower_limit, upper_limit)
    # ... rest of detection logic
```

---

## 🛠️ Advanced Features

### Adding Text Labels

Add color name labels to bounding boxes:

```python
if bbox is not None:
    x1, y1, x2, y2 = bbox
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    # Add text label
    cv2.putText(frame, 'Yellow Detected', (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
```

### Recording Detected Events

Log when colors are detected:

```python
import datetime

if bbox is not None:
    timestamp = datetime.datetime.now()
    print(f"[{timestamp}] Yellow detected at: {bbox}")
```

### Saving Detection Screenshots

Capture frames when color is detected:

```python
if bbox is not None:
    cv2.imwrite(f'detection_{datetime.datetime.now().timestamp()}.jpg', frame)
```

---

## 📊 Performance Tips

For optimal performance:

1. **Lighting** - Ensure good, even lighting for best results
2. **Camera Quality** - Higher resolution cameras provide better detection
3. **Background** - Use contrasting backgrounds to avoid false positives
4. **Distance** - Position objects within 1-3 meters of the camera
5. **Object Size** - Larger objects are detected more reliably

---

## 🐛 Troubleshooting

### Common Issues

**Issue:** Camera not opening
- **Solution:** Ensure no other application is using the webcam
- **Solution:** Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` for external cameras

**Issue:** Poor detection accuracy
- **Solution:** Adjust lighting conditions
- **Solution:** Modify HSV thresholds in `util.py`
- **Solution:** Ensure target color is vibrant and not washed out

**Issue:** Too many false positives
- **Solution:** Decrease `hue_range` for stricter matching
- **Solution:** Increase `sat_min` and `val_min` values

**Issue:** Color not detected at all
- **Solution:** Increase `hue_range` for wider detection
- **Solution:** Lower `sat_min` and `val_min` thresholds
- **Solution:** Verify BGR color values are correct

---

## 📦 Requirements

```
opencv-python == 4.10.0.84
numpy == 1.26.4
pillow == 10.4.0
```

**Minimum System Requirements:**
- Python 3.7+
- 4GB RAM
- Webcam (built-in or external)
- Windows / macOS / Linux

---

## 🎯 Use Cases

This color detection system can be used for:

- 🎮 **Gaming** - Color-based game controls
- 🏭 **Quality Control** - Industrial color inspection
- 🚦 **Traffic Analysis** - Vehicle color identification  
- 🎨 **Art Projects** - Interactive installations
- 🔬 **Research** - Computer vision experiments
- 🎓 **Education** - Learning OpenCV and color spaces
- 🤖 **Robotics** - Object tracking by color
- 📸 **Photography** - Color palette extraction

---

## 🚀 Future Enhancements

Potential improvements:
- [ ] Multiple color detection simultaneously
- [ ] Color palette extraction from images
- [ ] GUI for easy color selection
- [ ] Save detection statistics and analytics
- [ ] Object tracking across frames
- [ ] Motion detection combined with color detection
- [ ] Export detection data to CSV/JSON

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

- [OpenCV](https://opencv.org/) - Computer Vision library
- [NumPy](https://numpy.org/) - Numerical computing library
- [Pillow](https://python-pillow.org/) - Python Imaging Library

---

## 📧 Contact

**Author:** AlBaraa-1  
**Repository:** [Computer-vision](https://github.com/AlBaraa-1/Computer-vision)

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ and Python

</div>
