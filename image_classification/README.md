# 🚗 Car Image Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A machine learning image classifier that distinguishes between cars and non-cars using Support Vector Machines (SVM)**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [How It Works](#-how-it-works)

</div>

---

## 🌟 Features

- 🤖 **Machine Learning** - SVM classifier with automatic hyperparameter tuning
- 🎯 **High Accuracy** - Achieves 99.92% accuracy on the test dataset
- 📊 **Grid Search** - Automatic optimization of model parameters
- 🖼️ **Image Processing** - Resizes and preprocesses images for optimal classification
- 📈 **Detailed Metrics** - Comprehensive performance evaluation with confusion matrix
- ⚡ **Fast Training** - Efficient training with parallel processing
- 💾 **Organized Dataset** - Simple folder structure for easy data management

---

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Basic understanding of machine learning concepts

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AlBaraa-1/Computer-vision.git
   cd Computer-vision/image_classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset**
   ```
   dataset/
   ├── car/          # Place car images here
   └── not_car/      # Place non-car images here
   ```

---

## 📖 Usage

### Basic Training

Simply run the main script to train and evaluate the model:

```bash
python main.py
```

**Output:**
```
 Car Classification - Testing

Loading data...
✓ Loaded 6083 images

Training model...
✅ Training complete.
Best parameters: {'C': 1000, 'gamma': 0.001}

Model Evaluation
Accuracy: 100.00%
```

---

## 📊 Sample Images

### Car Examples
<div align="center">
<img src="dataset/car/00000000_00000018.jpg" alt="Car Sample" width="150"/>
<p><i>Car - Top-down view</i></p>
</div>

### Not Car Examples
<div align="center">
<img src="dataset/not_car/00000000_00000161.jpg" alt="Not Car Sample" width="150"/>
<p><i>Not Car - Background/Other objects</i></p>
</div>

---

## 🎯 Sample Predictions

<div align="center">
<img src="outputs/sample_predictions.png" alt="Sample Predictions" width="800"/>
<p><i>Model predictions on test images showing classification results</i></p>
</div>

---

## 🎯 Results

### Model Performance

```
Classification Report:
               precision    recall  f1-score   support
           0       1.00      1.00      1.00       609
           1       1.00      1.00      1.00       608
    accuracy                           1.00      1217

Confusion Matrix:
 [[609   0]
  [  1 607]]
```

**Key Metrics:**
- ✅ **Accuracy:** 99.92%
- ✅ **Precision:** 1.00 for both classes
- ✅ **Recall:** 1.00 for both classes
- ✅ **F1-Score:** 1.00 for both classes

---

## 🔧 How It Works

### 1. **Data Loading**
- Loads images from `dataset/car/` and `dataset/not_car/`
- Resizes all images to 15x15 pixels for consistency
- Flattens images into feature vectors
- Applies anti-aliasing for smooth resizing

### 2. **Data Preprocessing**
```python
IMG_SIZE = (15, 15)  # Resize to 15x15 pixels
data.append(img.flatten())  # Convert to 1D array
```

### 3. **Model Training**
- **Algorithm:** Support Vector Machine (SVM)
- **Kernel:** RBF (Radial Basis Function)
- **Optimization:** GridSearchCV for hyperparameter tuning
- **Parameters Tested:**
  - `C`: [1, 10, 100, 1000]
  - `gamma`: [0.001, 0.0001]
- **Cross-Validation:** 5-fold CV
- **Train/Test Split:** 80% training, 20% testing

### 4. **Evaluation**
- Calculates accuracy, precision, recall, and F1-score
- Generates confusion matrix
- Displays classification report

---

## 📁 Project Structure

```
image_classification/
├── dataset/              # Training data (gitignored)
│   ├── car/             # Car images
│   └── not_car/         # Non-car images
├── outputs/             # Generated outputs
│   └── sample_predictions.png
├── models/              # Saved models (optional)
├── main.py              # Main training script
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

---

## 🛠️ Configuration

Edit the configuration variables in `main.py`:

```python
# Configurations
DATA_DIR = os.path.join(SCRIPT_DIR, "dataset")
CLASSES = ["car", "not_car"]
IMG_SIZE = (15, 15)  # Image dimensions for processing
```

---

## 📦 Dependencies

- `numpy` - Numerical operations
- `scikit-learn` - Machine learning algorithms
- `scikit-image` - Image loading and processing
- `opencv-python` - Image manipulation (optional)

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 💡 Tips for Best Results

1. **Dataset Quality**
   - Use clear, well-lit images
   - Ensure consistent image quality
   - Balance the number of car and non-car images

2. **Image Size**
   - Smaller sizes (15x15) train faster but lose detail
   - Larger sizes preserve detail but take longer to train
   - Experiment to find the best balance

3. **Model Tuning**
   - Modify the parameter grid for different results
   - Try different kernels: 'linear', 'poly', 'rbf'
   - Adjust C and gamma values for fine-tuning

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new features
- Improve model performance
- Enhance documentation
- Report bugs

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**AlBaraa-1**
- GitHub: [@AlBaraa-1](https://github.com/AlBaraa-1)

---

## 🙏 Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/)
- Image processing with [scikit-image](https://scikit-image.org/)
- Inspired by classic computer vision techniques

---

<div align="center">
Made with ❤️ for Computer Vision Learning
</div>
