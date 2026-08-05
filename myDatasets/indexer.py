"""
Dataset indexer.

Scans the dataset directory once and creates metadata.csv.
"""

from pathlib import Path
import csv


CLASS_TO_LABEL = {
    "Daily Living": 0,
    "Fall": 1,
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

    for img in dataset_root.rglob("*.png"):
        parts = img.relative_to(dataset_root).parts

        # Fall/SA21/image.png
        if parts[0] == "Fall":
            label = 1
            activity = "Fall"
            subject = parts[1]

        # Daily Living/Walking_Slow/SA01/image.png
        elif parts[0] == "Daily Living":
            label = 0
            activity = parts[1]
            subject = parts[2]

        else:
            continue

        rows.append(
            {
                "image_path": str(img),
                "label": label,
                "activity": activity,
                "subject": subject,
            }
        )

    df = pd.DataFrame(rows)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_csv, index=False)

    print(f"Indexed {len(df)} images.")