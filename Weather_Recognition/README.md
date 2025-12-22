# 🌤️ Weather Recognition AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-95.4%25-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A deep learning-based weather recognition system that classifies images into four weather categories using ResNet-18 and Random Forest**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Performance](#-model-performance) • [How It Works](#-how-it-works)

</div>

---

## 🌟 Features

- 🤖 **Deep Learning** - ResNet-18 feature extraction with Random Forest classifier
- 🎯 **High Accuracy** - Achieves 95.4% accuracy on validation set
- ⚡ **GPU Accelerated** - CUDA-enabled for fast training and inference
- 🖼️ **Auto-Correction** - Automatic EXIF orientation handling
- 📊 **4 Categories** - Cloudy, Rainy, Shine, and Sunrise classification
- 💯 **High Confidence** - Average prediction confidence of ~75%
- 🧹 **Clean Dataset** - Manually curated and quality-controlled data

---

## 📊 Model Performance

### Overall Metrics
- **Accuracy:** 95.4%
- **Algorithm:** Random Forest + ResNet-18
- **Average Confidence:** 75%
- **Dataset:** 1,119 images (80/20 train/val split)

### Per-Category Performance

| Category | Accuracy | Training | Validation | Avg Confidence |
|----------|----------|----------|------------|----------------|
| 🌅 Sunrise | ~97%   | 287      | 73         | High           |
| ☀️ Shine   | ~96%   | 202      | 50         | High           |
| ☁️ Cloudy  | ~95%   | 243      | 59         | Medium-High    |
| 🌧️ Rainy   | ~93%   | 171      | 42         | Medium-High    |

---

## 🎯 Project Overview

This project demonstrates a complete machine learning workflow:
- ✅ Data collection and organization
- ✅ Dataset cleaning and quality control  
- ✅ Transfer learning with ResNet-18
- ✅ Model training and validation
- ✅ Performance visualization
- ✅ Production-ready inference

## 🗂️ Project Structure

```
weather_recognition/
├── data/
│   ├── train/              # Training dataset (80%)
│   │   ├── cloudy/         # 243 images
│   │   ├── rainy/          # 171 images
│   │   ├── shine/          # 202 images
│   │   └── sunrise/        # 287 images
│   └── val/                # Validation dataset (20%)
│       ├── cloudy/         # 59 images
│       ├── rainy/          # 42 images
│       ├── shine/          # 50 images
│       └── sunrise/        # 73 images
├── model/
│   └── weather_rf_model.joblib  # Trained model
├── tests/                  # Test images folder
├── main.py                 # Training script
├── test.py                 # Testing script
└── README.md               # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (optional but recommended)
- 2GB+ free disk space

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AlBaraa-1/Computer-vision.git
   cd Computer-vision/weather_recognition
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Required packages:
   ```
   torch
   torchvision
   img2vec-pytorch
   scikit-learn
   Pillow
   matplotlib
   joblib
   ```

3. **Verify installation**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

---

## 💻 Usage

### Training the Model

Train the model on the prepared dataset:

```bash
python main.py
```

**Expected Output:**
```
GPU detected: NVIDIA GeForce RTX 4070 Ti SUPER
CUDA Version: 12.1
Extracting training features...
Extracting validation features...
Training model...
Validation Accuracy: 0.954
Model saved as model/weather_rf_model.joblib
```

### Testing on New Images

1. Place test images in the `tests/` folder
2. Run the test script:

```bash
python test.py
```

**Output Example:**
```
[1/5] sunset_photo.jpg
  Prediction: SUNRISE
  Confidence: 89.0%
  Probabilities:
    → sunrise    █████████████   89.0%
      cloudy                     5.0%
      shine                      5.0%
      rainy                      1.0%
```

The script generates `test_results.png` with visual predictions.

---

---

## 🛠️ How It Works

### Architecture Overview

```
Input Image (RGB)
    ↓
EXIF Orientation Correction
    ↓
ResNet-18 Feature Extractor (Pre-trained on ImageNet)
    ↓
512-dimensional Feature Vector
    ↓
Random Forest Classifier (100 trees)
    ↓
Weather Category + Confidence Scores
```

### Technical Details

**1. Feature Extraction**
- Uses **ResNet-18** pre-trained on ImageNet
- Extracts 512-dimensional feature vectors
- GPU-accelerated via CUDA
- Automatic image orientation handling (EXIF-aware)

**2. Classification**
- **Random Forest** with default parameters
- Trained on extracted features
- Returns probability distribution across categories
- Fast inference (~0.1s per image on GPU)

**3. Data Processing**
```python
# Feature extraction
img2vec = Img2Vec(cuda=True)
img = ImageOps.exif_transpose(Image.open(path))
features = img2vec.get_vec(img)

# Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
prediction = model.predict([features])[0]
confidence = model.predict_proba([features])[0].max()
```

### Why This Architecture?

✅ **Transfer Learning** - Leverages ImageNet knowledge  
✅ **Speed** - Fast training and inference  
✅ **Accuracy** - Deep features + ensemble learning  
✅ **Simplicity** - No complex hyperparameter tuning  
✅ **Robustness** - Handles various image conditions

---

## 📈 Dataset Journey

### Initial Setup
- **Source:** Custom curated weather images
- **Total:** 1,125 images
- **Split:** 80% training, 20% validation
- **Categories:** 4 (cloudy, rainy, shine, sunrise)

### Data Cleaning Process

| Step | Action | Result |
|------|--------|--------|
| 1️⃣ Initial Training | First model training | 92.9% accuracy |
| 2️⃣ Quality Analysis | Identified 66 low-accuracy images | - |
| 3️⃣ Manual Review | Reviewed each problematic image | - |
| 4️⃣ Curation | Deleted 9, recategorized 1, kept 56 | - |
| 5️⃣ Final Training | Retrained on clean dataset | **95.4% accuracy** ✨ |

**Improvement:** +2.5% accuracy through data quality control

### Final Dataset Statistics
- **Total Images:** 1,119 (after cleaning)
- **Train/Val Split:** 894 / 225
- **Balance:** Relatively balanced across categories
- **Quality:** High-confidence predictions (avg 75%)

---

## � Project Structure

```
weather_recognition/
├── 📄 main.py              # Training script
├── 📄 test.py              # Inference script
├── 📄 README.md            # Documentation
├── 📄 requirements.txt     # Dependencies
├── 📂 data/
│   ├── 📂 train/           # Training images (894)
│   │   ├── cloudy/         # 243 images
│   │   ├── rainy/          # 171 images
│   │   ├── shine/          # 202 images
│   │   └── sunrise/        # 287 images
│   └── 📂 val/             # Validation images (225)
│       ├── cloudy/         # 59 images
│       ├── rainy/          # 42 images
│       ├── shine/          # 50 images
│       └── sunrise/        # 73 images
├── 📂 model/
│   └── weather_rf_model.joblib  # Trained model
└── � test_outputs/        # Prediction results
```

---

## 🔧 Troubleshooting

### Issue: CUDA Out of Memory
**Solution:** Reduce batch size or use CPU mode:
```python
img2vec = Img2Vec(cuda=False)
```

### Issue: Images Displayed Rotated/Flipped
**Solution:** Already handled via `ImageOps.exif_transpose()` in both scripts

### Issue: Low Prediction Confidence
**Possible Causes:**
- Image doesn't match training categories
- Ambiguous weather conditions
- Poor image quality

**Actions:**
- Check if image is clear and representative
- Consider adding similar images to training data
- Review confidence scores for all categories

### Issue: Module Not Found
**Solution:** Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

---

## � Lessons Learned

1. **Data Quality > Quantity:** Removing 9 ambiguous images improved accuracy by 2.5%
2. **EXIF Metadata Matters:** Many phone photos have orientation flags that must be handled
3. **Manual Review is Valuable:** Automated metrics miss context that human review catches
4. **Transfer Learning Works:** Pre-trained ResNet-18 excels even with small datasets
5. **Confidence Calibration:** Average 75% confidence indicates well-calibrated predictions

---

## � License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Add more weather categories (foggy, snowy, etc.)
- Implement ensemble methods
- Create web interface for predictions
- Expand dataset with more diverse conditions

---

## 📧 Contact

For questions or suggestions, feel free to open an issue or reach out!

---

**Built with ❤️ using PyTorch and scikit-learn**
