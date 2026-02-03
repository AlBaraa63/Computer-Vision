import os
import joblib
import torch
from img2vec_pytorch import Img2Vec
from PIL import Image
import matplotlib.pyplot as plt

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model', 'weather_rf_model.joblib')
TESTS_DIR = os.path.join(SCRIPT_DIR, 'tests')

# Load model
print("Loading model...")
USE_CUDA = torch.cuda.is_available()
print(f"Using: {'GPU' if USE_CUDA else 'CPU'}")
img2vec = Img2Vec(cuda=USE_CUDA)
model = joblib.load(MODEL_PATH)
print(f"Model loaded successfully!\n")

# Get all test images
if not os.path.exists(TESTS_DIR):
    print(f"✗ Tests folder not found: {TESTS_DIR}")
    print("Please create a 'tests' folder and add your images.")
    exit()

test_images = [f for f in os.listdir(TESTS_DIR) 
               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

if not test_images:
    print(f"✗ No images found in {TESTS_DIR}")
    print("Supported formats: .jpg, .jpeg, .png, .bmp")
    exit()

print("=" * 70)
print(f"TESTING {len(test_images)} IMAGES")
print("=" * 70)

# Test each image
results = []
for idx, img_name in enumerate(sorted(test_images), 1):
    img_path = os.path.join(TESTS_DIR, img_name)
    
    try:
        # Load image and predict
        img = Image.open(img_path)
        
        # Fix orientation based on EXIF data
        try:
            from PIL import ImageOps
            img = ImageOps.exif_transpose(img)
        except:
            pass
        
        img = img.convert("RGB")
        vec = img2vec.get_vec(img)
        prediction = model.predict([vec])[0]
        probabilities = model.predict_proba([vec])[0]
        confidence = probabilities.max()
        
        # Get all class probabilities
        class_probs = dict(zip(model.classes_, probabilities))
        
        # Print result
        print(f"\n[{idx}/{len(test_images)}] {img_name}")
        print(f"  Prediction: {prediction.upper()}")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Probabilities:")
        for cls, prob in sorted(class_probs.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 15)
            marker = "→" if cls == prediction else " "
            print(f"    {marker} {cls:10s} {bar:15s} {prob:.1%}")
        
        results.append({
            'name': img_name,
            'img': img,
            'prediction': prediction,
            'confidence': confidence,
            'probs': class_probs
        })
        
    except Exception as e:
        print(f"\n[{idx}/{len(test_images)}] ✗ Failed: {img_name}")
        print(f"  Error: {e}")

# Create visualization
if results:
    print("\n" + "=" * 70)
    print("Creating visualization...")
    
    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig = plt.figure(figsize=(6 * cols, 5 * rows))
    
    for idx, result in enumerate(results, 1):
        ax = plt.subplot(rows, cols, idx)
        ax.imshow(result['img'])
        ax.axis('off')
        
        # Create title
        title = f"{result['name']}\n"
        title += f"🔹 {result['prediction'].upper()} ({result['confidence']:.0%})"
        
        color = 'lightgreen' if result['confidence'] > 0.7 else 'yellow' if result['confidence'] > 0.5 else 'lightcoral'
        
        ax.set_title(title, fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(SCRIPT_DIR, 'test_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Results saved to: test_results.png")
    
    plt.show()

print("\n" + "=" * 70)
print("TESTING COMPLETE!")
print("=" * 70)
