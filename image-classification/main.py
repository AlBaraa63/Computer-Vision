import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#  CONFIGURATIONS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "dataset")
CLASSES = ["car", "not_car"]
IMG_SIZE = (15, 15)


#  LOAD DATA
def load_data(data_dir=DATA_DIR, classes=CLASSES):
    data, labels = [], []

    for cls in classes:
        folder = os.path.join(data_dir, cls)
        label = 1 if cls == "car" else 0  # numeric labels

        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)

            # read and resize image
            img = imread(img_path)
            img = resize(img, IMG_SIZE, anti_aliasing=True)

            # flatten image and add to data
            data.append(img.flatten())
            labels.append(label)
                
    # convert to numpy arrays 
    data = np.array(data)
    labels = np.array(labels)
    return data, labels


#  TRAIN MODEL
def train_model(x, y):
    # split the dataset
    x_train, x_test, y_train, y_test = train_test_split(
        # shuffle for randomness, st
        x, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
    )
    
    # define model and parameters
    classifier = SVC()
    parameters = [{'gamma': [0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
    grid_search = GridSearchCV(classifier, parameters, cv=5, n_jobs=-1)

    # train
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_

    print("✅ Training complete.")
    print("Best parameters:", grid_search.best_params_)

    return best_model, x_test, y_test


#  EVALUATE MODEL
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    print("\nModel Evaluation")
    print(f"Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return acc


#  MAIN
if __name__ == "__main__":
    print(" Car Classification - Testing\n")
    
    # Load data
    print("Loading data...")
    x, y = load_data()
    print(f"✓ Loaded {len(x)} images\n")

    # Train model
    print("Training model...")
    model, x_test, y_test = train_model(x, y)
    print()
    
    # Evaluate
    evaluate_model(model, x_test, y_test)
