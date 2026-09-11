import pathlib, shutil

src = pathlib.Path('Dataset')
dst = pathlib.Path('dataset/raw')
dst.mkdir(parents=True, exist_ok=True)

mapping = {'Bus':'bus','Car':'car','motorcycle':'bike','Truck':'truck'}

for f in src.iterdir():
    if f.is_dir():
        new_name = mapping.get(f.name, f.name.lower())
        target = dst / new_name
        print(f"Moving {f.name} -> {target}")
        shutil.copytree(f, target, dirs_exist_ok=True)

print("\nMoved! Counts:")
for p in dst.iterdir():
    if p.is_dir():
        print(f"{p.name}: {len(list(p.glob('*')))} images")