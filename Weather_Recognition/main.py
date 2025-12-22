import os
import joblib
import torch
from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')

# Auto-detect GPU
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print(f" GPU detected: {torch.cuda.get_device_name(0)}")
    print(f" CUDA Version: {torch.version.cuda}")
else:
    print(" No GPU detected, using CPU")

# Initialize Img2Vec
img2vec = Img2Vec(cuda=USE_CUDA)

def extract_features_from_folder(folder):
    from PIL import ImageOps
    features, labels = [], []
    classes = sorted(os.listdir(folder))

    for cls in classes:
        cls_path = os.path.join(folder, cls)
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            try:
                img = Image.open(img_path)
                # Fix orientation based on EXIF data
                try:
                    img = ImageOps.exif_transpose(img)
                except:
                    pass
                img = img.convert("RGB")
                vec = img2vec.get_vec(img)
                features.append(vec)
                labels.append(cls)
            except Exception as e:
                print(f" Skipped {img_name}: {e}")
    return features, labels

def main():
    print(" Extracting training features...")
    X_train, y_train = extract_features_from_folder(TRAIN_DIR)

    print(" Extracting validation features...")
    X_val, y_val = extract_features_from_folder(VAL_DIR)

    print(" Training model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    score = accuracy_score(y_val, y_pred)
    print(f" Validation Accuracy: {score:.3f}")

    # Save model
    MODEL_DIR = os.path.join(SCRIPT_DIR, 'model')
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'weather_rf_model.joblib')
    joblib.dump(model, model_path)
    print(f" Model saved as {model_path}")

if __name__ == "__main__":
    main()