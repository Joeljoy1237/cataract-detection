"""
split.py - Train/Val/Test Splitter for Cataract Dataset
"""

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(metadata_csv, output_dir, test_size=0.2, val_size=0.1, seed=42):
    df = pd.read_csv(metadata_csv)

    train_df, temp_df = train_test_split(
        df, test_size=(test_size + val_size), stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (test_size + val_size),
        stratify=temp_df["label"],
        random_state=seed,
    )

    def copy_files(subset_df, subset_name):
        for _, row in subset_df.iterrows():
            subset_folder = os.path.join(output_dir, subset_name, row["label"])
            os.makedirs(subset_folder, exist_ok=True)
            dst = os.path.join(subset_folder, row["processed_filename"])
            if not os.path.exists(dst):
                shutil.copy(row["filepath"], dst)

    print("\nCopying files...")
    copy_files(train_df, "train")
    copy_files(val_df, "val")
    copy_files(test_df, "test")

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print("\n✅ Dataset splitting complete!")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    return train_df, val_df, test_df
