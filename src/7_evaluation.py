import tensorflow as tf, pathlib, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

model = tf.keras.models.load_model("model_vgg7_final.h5")
val_ds = tf.keras.utils.image_dataset_from_directory("dataset/processed/val", image_size=(128,128), batch_size=32)
class_names = val_ds.class_names
print("Classes:", class_names)

# get true labels
y_true = []
y_pred = []
for x,y in val_ds:
    y_true.extend(y.numpy())
    preds = model.predict(x, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))

print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix - for report graph
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.title("VGG7 - Confusion Matrix (4 Classes)")
plt.savefig("dataset/confusion_matrix.png")
print("Saved: dataset/confusion_matrix.png - PROJECT FULLY DONE!")