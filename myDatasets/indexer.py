"""
Dataset indexer.

Scans the dataset directory once and creates metadata.csv.
"""

from pathlib import Path
import csv


CLASS_TO_LABEL = {
    "Back_Lateral_Change": 0,
    "Car_InOut": 1,
    "Collapse_Into_Chair": 2,
    "Gentle_Jump": 3,
    "Jogging_Fast": 4,
    "Jogging_Slow": 5,
    "Lying_Fast_SitAgain": 6,
    "Lying_Slow_SitAgain": 7,
    "Sit_HalfHeight_Fast": 8,
    "Sit_HalfHeight_Slow": 9,
    "Sit_LowHeight_Fast": 10,
    "Sit_LowHeight_Slow": 11,
    "Stairs_Fast": 12,
    "Stairs_Slow": 13,
    "Standing_KneeBend": 14,
    "Standing_NoKneeBend": 15,
    "Stumble_Walking": 16,
    "Walking_Fast": 17,
    "Walking_Slow": 18,
}

VALID_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
}


from pathlib import Path
import pandas as pd


def build_metadata(dataset_root, output_csv):
    dataset_root = Path(dataset_root)

    rows = []

    count = 0

    for img in dataset_root.rglob("*.png"):
        count += 1

        if count % 500 == 0:
            print(f"Visited {count} images")

        parts = img.relative_to(dataset_root).parts
    
        if parts[0] != "Daily Living":
            continue

        activity = parts[1]
        subject = parts[2]

        label = CLASS_TO_LABEL[activity]

        rows.append(
            {
                "image_path": str(img),
                "label": label,
                "activity": activity,
                "subject": subject,
            }
        )

    print("Visited total:", count)
    print("Indexed:", len(rows))

    df = pd.DataFrame(rows)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_csv, index=False)

    print(f"Indexed {len(df)} images.")