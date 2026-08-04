"""
Dataset indexer.

Scans the dataset directory once and creates metadata.csv.
"""

from pathlib import Path
import csv


CLASS_TO_LABEL = {
    "ADL": 0,
    "Fall": 1,
}

VALID_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
}


def build_metadata(
    dataset_root: str | Path,
    output_csv: str | Path,
) -> None:

    dataset_root = Path(dataset_root)
    output_csv = Path(output_csv)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for activity_type in CLASS_TO_LABEL:

        type_dir = dataset_root / activity_type

        if not type_dir.exists():
            continue

        for activity_dir in type_dir.iterdir():

            if not activity_dir.is_dir():
                continue

            for subject_dir in activity_dir.iterdir():

                if not subject_dir.is_dir():
                    continue

                for image in subject_dir.iterdir():

                    if image.suffix.lower() not in VALID_EXTENSIONS:
                        continue

                    rows.append(
                        {
                            "image_path": str(image.resolve()),
                            "label": CLASS_TO_LABEL[activity_type],
                            "activity_type": activity_type,
                            "activity": activity_dir.name,
                            "subject": subject_dir.name,
                        }
                    )

    with open(output_csv, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "label",
                "activity_type",
                "activity",
                "subject",
            ],
        )

        writer.writeheader()

        writer.writerows(rows)

    print(f"Indexed {len(rows)} images.")