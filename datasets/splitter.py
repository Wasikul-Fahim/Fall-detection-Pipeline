"""
Subject-wise dataset splitter.
"""

import pandas as pd

from sklearn.model_selection import train_test_split


def split_by_subject(
    metadata_csv: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_state: int = 42,
):

    df = pd.read_csv(metadata_csv)

    subjects = df["subject"].unique()

    train_subjects, temp_subjects = train_test_split(
        subjects,
        train_size=train_ratio,
        random_state=random_state,
        shuffle=True,
    )

    val_subjects, test_subjects = train_test_split(
        temp_subjects,
        train_size=val_ratio / (1 - train_ratio),
        random_state=random_state,
        shuffle=True,
    )

    train_df = df[df.subject.isin(train_subjects)]

    val_df = df[df.subject.isin(val_subjects)]

    test_df = df[df.subject.isin(test_subjects)]

    train_df.to_csv(train_csv, index=False)

    val_df.to_csv(val_csv, index=False)

    test_df.to_csv(test_csv, index=False)

    print("Train images :", len(train_df))
    print("Val images   :", len(val_df))
    print("Test images  :", len(test_df))