import pathlib, shutil, cv2
from sklearn.model_selection import train_test_split

RAW = pathlib.Path("dataset/raw")
PROCESSED = pathlib.Path("dataset/processed")
if PROCESSED.exists(): shutil.rmtree(PROCESSED)

files=[]
for cls in [p for p in RAW.iterdir() if p.is_dir()]:
    for img in cls.glob("*.*"):
        if img.suffix.lower() in [".jpg",".jpeg",".png"]:
            files.append((img, cls.name.lower()))

train, val = train_test_split(files, test_size=0.2, random_state=42, stratify=[c for _,c in files])

def save(lst, split):
    for src, label in lst:
        d=PROCESSED/split/label
        d.mkdir(parents=True, exist_ok=True)
        im=cv2.imread(str(src))
        if im is None: continue
        im=cv2.resize(im,(128,128))
        cv2.imwrite(str(d/src.name), im)

save(train,"train")
save(val,"val")
print(f"PREPROCESS DONE Train:{len(train)} Val:{len(val)}")