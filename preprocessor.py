"""
preprocessor.py - Medical Image Preprocessing Module
"""

import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm
import albumentations as A


class CataractPreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.fundus_data = None
        self.fundus_preprocess = self._create_fundus_preprocessor()
        self.slitlamp_preprocess = self._create_slitlamp_preprocessor()
        self._create_directories()

    def _create_directories(self):
        for dir_path in ["processed_data/fundus", "processed_data/slitlamp", "processed_data/metadata"]:
            os.makedirs(dir_path, exist_ok=True)

    def _create_fundus_preprocessor(self):
        return A.Compose(
            [A.Resize(224, 224), A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0), A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
        )

    def _create_slitlamp_preprocessor(self):
        return A.Compose(
            [A.Resize(224, 224), A.ToGray(p=1.0), A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0), A.Normalize(mean=(0.5,), std=(0.5,))]
        )

    def load_fundus_metadata(self, excel_path):
        try:
            df = pd.read_excel(excel_path)
            required_cols = ["ID", "Left-Fundus", "Right-Fundus", "Left-Diagnostic Keywords", "Right-Diagnostic Keywords"]

            column_mapping = {}
            for req_col in required_cols:
                matches = [col for col in df.columns if req_col.lower() in col.lower()]
                if matches:
                    column_mapping[req_col] = matches[0]

            df = df.rename(columns=column_mapping)
            self.fundus_data = df
            print(f"Loaded fundus metadata: {len(df)} records")
            return df
        except Exception as e:
            print(f"Error loading Excel: {e}")
            return None

    def _extract_severity(self, keywords):
        if pd.isna(keywords) or keywords == "":
            return "Normal"
        text = str(keywords).lower()
        if any(word in text for word in ["severe", "advanced", "dense", "mature"]):
            return "Severe"
        elif any(word in text for word in ["moderate", "significant"]):
            return "Moderate"
        elif any(word in text for word in ["mild", "early", "trace", "slight"]):
            return "Mild"
        elif any(word in text for word in ["normal", "clear", "healthy"]):
            return "Normal"
        else:
            return "Mild" if "cataract" in text else "Normal"

    def process_fundus_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return self.fundus_preprocess(image=image)["image"]
        except:
            return None

    def process_slitlamp_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            processed = self.slitlamp_preprocess(image=image)["image"]
            # Convert single-channel → 3-channel for DenseNet
            if len(processed.shape) == 2:
                processed = np.stack([processed] * 3, axis=-1)
            return processed
        except:
            return None

    def process_fundus_images(self, images_folder):
        if self.fundus_data is None:
            print("Error: Load Excel first")
            return None

        processed_data = []
        for _, row in tqdm(self.fundus_data.iterrows(), total=len(self.fundus_data)):
            for eye in ["Left", "Right"]:
                col = f"{eye}-Fundus"
                kw_col = f"{eye}-Diagnostic Keywords"
                if not pd.isna(row[col]):
                    img_path = os.path.join(images_folder, row[col])
                    if os.path.exists(img_path):
                        label = self._extract_severity(row[kw_col])
                        processed_img = self.process_fundus_image(img_path)
                        if processed_img is not None:
                            label_folder = os.path.join("processed_data/fundus", label)
                            os.makedirs(label_folder, exist_ok=True)
                            filename = f"{row['ID']}_{eye}_{os.path.basename(row[col])}"
                            save_path = os.path.join(label_folder, filename)
                            img_to_save = (processed_img * 0.5 + 0.5) * 255
                            img_to_save = img_to_save.astype(np.uint8)
                            cv2.imwrite(save_path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
                            processed_data.append(
                                {"patient_id": row["ID"], "eye": eye, "original_filename": row[col], "processed_filename": filename, "label": label, "filepath": save_path, "original_filepath": img_path}
                            )

        df = pd.DataFrame(processed_data)
        df.to_csv("processed_data/metadata/fundus_metadata.csv", index=False)
        print("Fundus preprocessing complete")
        return df

    def process_slitlamp_images(self, slitlamp_folder):
        categories = ["normal", "mature", "immature"]
        processed_data = []
        for category in categories:
            cat_path = os.path.join(slitlamp_folder, category)
            if not os.path.exists(cat_path):
                continue
            image_files = [f for f in os.listdir(cat_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            for img_file in tqdm(image_files):
                img_path = os.path.join(cat_path, img_file)
                processed_img = self.process_slitlamp_image(img_path)
                if processed_img is not None:
                    label_folder = os.path.join("processed_data/slitlamp", category)
                    os.makedirs(label_folder, exist_ok=True)
                    save_filename = f"processed_{img_file}"
                    save_path = os.path.join(label_folder, save_filename)
                    img_to_save = (processed_img * 0.5 + 0.5) * 255
                    img_to_save = img_to_save.astype(np.uint8)
                    cv2.imwrite(save_path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
                    processed_data.append({"original_filename": img_file, "processed_filename": save_filename, "label": category, "filepath": save_path, "original_filepath": img_path})

        df = pd.DataFrame(processed_data)
        df.to_csv("processed_data/metadata/slitlamp_metadata.csv", index=False)
        print("Slitlamp preprocessing complete")
        return df
