import tensorflow as tf
import cv2
import numpy as np
import pathlib

model = tf.keras.models.load_model("model_vgg7_final.h5")
class_names = ['bike','bus','car','truck']
print(f"Model loaded! Classes: {class_names}")

# Test folder
VAL_PATH = pathlib.Path("dataset/processed/val")

for cls in class_names:
    # take first image from each class
    img_path = list((VAL_PATH / cls).glob("*.*"))[0]
    print(f"\n--- Testing: {cls} -> {img_path.name} ---")

    img = cv2.imread(str(img_path))
    img_resized = cv2.resize(img, (128,128))
    img_norm = img_resized / 255.0
    pred = model.predict(np.expand_dims(img_norm, 0), verbose=0)

    predicted_idx = np.argmax(pred)
    confidence = np.max(pred) * 100

    print(f"True: {cls}")
    print(f"Predicted: {class_names[predicted_idx]} ({confidence:.2f}% confident)")
    print(f"All probabilities: {pred[0]}")

    if class_names[predicted_idx] == cls:
        print("✅ CORRECT!")
    else:
        print("❌ WRONG - but learning!")

print("\n--- Testing with YOUR own photo ---")
print("Put any bike/bus/car/truck image as 'my_test.jpg' in project folder and run again")
try:
    img = cv2.imread("my_test.jpg")
    if img is not None:
        img_resized = cv2.resize(img, (128,128)) / 255.0
        pred = model.predict(np.expand_dims(img_resized, 0), verbose=0)
        print(f"Your photo predicted: {class_names[np.argmax(pred)]} with {np.max(pred)*100:.1f}% confidence")
except:
    pass

print("\nTESTING DONE - READY FOR FINAL REPORT!")