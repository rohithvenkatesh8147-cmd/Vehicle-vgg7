import pathlib, cv2, matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("dataset/labels.csv")

print("CLASS DISTRIBUTION:")
print(df['label_name'].value_counts())

# Bar chart
plt.figure()
df['label_name'].value_counts().plot(kind='bar')
plt.title("Vehicle Dataset - 4 Classes, 400 Images")
plt.ylabel("Images")
plt.tight_layout()
plt.savefig("dataset/eda_class_balance.png")
print("Saved: dataset/eda_class_balance.png")

# Samples
fig, axes = plt.subplots(2,2, figsize=(8,8))
axes = axes.flatten()
for ax, cls in zip(axes, sorted(df['label_name'].unique())):
    path = df[df['label_name']==cls].iloc[0]['filepath']
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.set_title(f"{cls} ({len(df[df['label_name']==cls])} imgs)")
    ax.axis('off')
plt.tight_layout()
plt.savefig("dataset/eda_samples.png")
print("Saved: dataset/eda_samples.png")
print("EDA DONE")