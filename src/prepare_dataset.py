
from pathlib import Path
import random
import shutil

SEED = 42
random.seed(SEED)

SOURCE = Path("data/raw/flower_photos")
DEST = Path("data/processed")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

classes = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

if DEST.exists():
    raise FileExistsError(
        f"{DEST} already exists. Remove it before recreating the split."
    )

for class_name in classes:
    source_dir = SOURCE / class_name

    images = sorted(
        p for p in source_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    random.shuffle(images)

    n = len(images)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split_name, files in splits.items():
        target_dir = DEST / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for image in files:
            shutil.copy2(image, target_dir / image.name)

        print(f"{split_name:5} | {class_name:12} | {len(files)} images")

print("\nDataset preparation complete!")
