import pathlib, cv2, numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

PROCESSED_TRAIN = pathlib.Path("dataset/processed/train")
AUGMENTED = pathlib.Path("dataset/augmented")
AUGMENTED.mkdir(exist_ok=True)

# Augmentation settings
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8,1.2]
)

print("--- AUGMENTATION START ---")
total_created = 0
for cls_dir in PROCESSED_TRAIN.iterdir():
    if not cls_dir.is_dir(): continue
    out_dir = AUGMENTED / cls_dir.name
    out_dir.mkdir(exist_ok=True)
    
    for img_path in cls_dir.glob("*.*"):
        img = cv2.imread(str(img_path))
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)
        
        # create 3 augmented versions per image
        i = 0
        for batch in datagen.flow(img, batch_size=1, save_to_dir=str(out_dir), save_prefix="aug", save_format="jpg"):
            i+=1
            total_created+=1
            if i >= 3: break

print(f"AUGMENTATION DONE - Created {total_created} new images")
print(f"Original train: 320 -> Augmented: +{total_created} = {320+total_created} images")
print(f"Saved to: {AUGMENTED}")