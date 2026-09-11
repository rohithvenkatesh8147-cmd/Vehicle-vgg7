from bing_image_downloader import downloader
from pathlib import Path

ROOT = "dataset/raw"
classes = {
    "auto": "auto rickshaw india",
    "bike": "bike motorcycle india",
    "bus": "bus india",
    "car": "car india",
    "truck": "truck lorry india"
}

Path(ROOT).mkdir(parents=True, exist_ok=True)

for label, query in classes.items():
    print(f"\n--- Downloading {label}: {query} - 10000 images ---")
    downloader.download(
        query, 
        limit=10000, 
        output_dir=ROOT, 
        adult_filter_off=True, 
        force_replace=False,
        timeout=60,
        verbose=True
    )
    # bing creates folder with query name, rename to label
    import shutil, os
    q_folder = Path(ROOT) / query
    target = Path(ROOT) / label
    if q_folder.exists():
        for f in q_folder.glob("*"):
            shutil.move(str(f), str(target / f.name))
        shutil.rmtree(q_folder)
    print(f"{label} done: {len(list(target.glob('*')))} images")

print("\n=== FINAL COUNTS ===")
from pathlib import Path
for c in ["auto","bike","bus","car","truck"]:
    print(f"{c}: {len(list((Path(ROOT)/c).glob('*')))}")