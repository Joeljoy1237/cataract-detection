"""
main.py - Main Pipeline Orchestrator
Coordinates preprocessing, splitting, and augmentation workflow
"""

import os
from datetime import datetime
from preprocessor import CataractPreprocessor
from split import split_dataset
from augmentation import CataractAugmentation


class CataractPipeline:
    """Main pipeline for cataract detection preprocessing and augmentation"""

    def __init__(self):
        """Initialize pipeline components"""
        self.preprocessor = CataractPreprocessor(target_size=(224, 224))
        self.augmentor = CataractAugmentation(augmentation_factor=2)

        print("CATARACT DETECTION PIPELINE")
        print("=" * 50)

    def check_input_data(self):
        """Check if required input data exists"""
        print("\nChecking input data...")

        fundus_excel = "dataset/fundus/metadata.xlsx"
        fundus_images = "dataset/fundus/Training Images"
        slitlamp_folder = "dataset/slit-lamp"

        status = {
            "fundus_excel": os.path.exists(fundus_excel),
            "fundus_images": os.path.exists(fundus_images),
            "slitlamp": os.path.exists(slitlamp_folder),
        }

        if status["fundus_excel"]:
            print(f"✓ Fundus Excel: {fundus_excel}")
        else:
            print(f"✗ Missing: {fundus_excel}")

        if status["fundus_images"]:
            print(f"✓ Fundus Images: {fundus_images}")
        else:
            print(f"✗ Missing: {fundus_images}")

        if status["slitlamp"]:
            print(f"✓ Slit-lamp: {slitlamp_folder}")
        else:
            print(f"✗ Missing: {slitlamp_folder}")

        return status

    def run_preprocessing(self):
        """Execute preprocessing pipeline"""
        print("\n=== PREPROCESSING PHASE ===")

        results = {}

        # Process fundus
        if os.path.exists("dataset/fundus/metadata.xlsx"):
            print("\n1. Processing Fundus Images")
            self.preprocessor.load_fundus_metadata("dataset/fundus/metadata.xlsx")
            fundus_df = self.preprocessor.process_fundus_images(
                "dataset/fundus/Training Images"
            )

            if fundus_df is not None:
                results["fundus"] = fundus_df["label"].value_counts().to_dict()

        # Process slit-lamp
        if os.path.exists("dataset/slit-lamp"):
            print("\n2. Processing Slit-lamp Images")
            slitlamp_df = self.preprocessor.process_slitlamp_images("dataset/slit-lamp")
            if slitlamp_df is not None:
                results["slitlamp"] = slitlamp_df["label"].value_counts().to_dict()

        return results

    def run_splitting(self):
        """Execute dataset splitting into train/val/test"""
        print("\n=== SPLITTING PHASE ===")

        # Split fundus
        fundus_metadata = "processed_data/metadata/fundus_metadata.csv"
        fundus_split_dir = "processed_data/fundus_split"
        if os.path.exists(fundus_metadata):
            split_dataset(fundus_metadata, fundus_split_dir)

        # Split slit-lamp
        slitlamp_metadata = "processed_data/metadata/slitlamp_metadata.csv"
        slitlamp_split_dir = "processed_data/slitlamp_split"
        if os.path.exists(slitlamp_metadata):
            split_dataset(slitlamp_metadata, slitlamp_split_dir)

    def run_augmentation(self):
        """Execute augmentation pipeline (only on training sets)"""
        print("\n=== AUGMENTATION PHASE ===")

        # Augment fundus training data
        fundus_train_folder = "processed_data/fundus_split/train"
        if os.path.exists(fundus_train_folder):
            print("\n1. Augmenting Fundus Training Data")
            self.augmentor.augment_folder(
                fundus_train_folder, "augmented_data/fundus/train", "fundus"
            )

        # Augment slit-lamp training data
        slitlamp_train_folder = "processed_data/slitlamp_split/train"
        if os.path.exists(slitlamp_train_folder):
            print("\n2. Augmenting Slit-lamp Training Data")
            self.augmentor.augment_folder(
                slitlamp_train_folder, "augmented_data/slitlamp/train", "slitlamp"
            )

        self.augmentor.print_summary()

    def run(self):
        """Execute complete pipeline"""
        start_time = datetime.now()

        # Check input data
        self.check_input_data()

        # Run preprocessing
        preprocess_results = self.run_preprocessing()

        # Run splitting
        self.run_splitting()

        # Run augmentation
        self.run_augmentation()

        # Summary
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 50)
        print("✓ PIPELINE COMPLETED")
        print("=" * 50)
        print(f"Execution time: {duration}")

        print("\nProcessed images distribution:")
        for key, value in preprocess_results.items():
            print(f"{key}: {value}")

        print("\nOutput folders:")
        print("  📁 processed_data/ - Preprocessed images (organized by labels)")
        print("  📁 processed_data/*_split - Train/Val/Test splits")
        print("  📁 augmented_data/ - Augmented training data (organized by labels)")

        print("\n🎯 Ready for DenseNet-169 model training!")


def main():
    """Main entry point"""
    print("AI-BASED CATARACT DETECTION")
    print("Preprocessing → Splitting → Augmentation Pipeline")
    print("Authors: BIDHUN B, DEEPAK DAYANANDAN, JOEL JOY, VARGHEESE FRANCIS")
    print("Institution: Carmel College of Engineering and Technology\n")

    try:
        pipeline = CataractPipeline()
        pipeline.run()
        return 0
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())