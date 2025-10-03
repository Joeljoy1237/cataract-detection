"""
preprocessor.py - Medical Image Preprocessing Module
Uses professional medical imaging preprocessing techniques
"""

import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm
import albumentations as A

class CataractPreprocessor:
    """
    Medical-grade preprocessing for cataract detection
    Separate pipelines for fundus and slit-lamp images
    """
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.fundus_data = None
        
        # Create preprocessing pipelines
        self.fundus_preprocess = self._create_fundus_preprocessor()
        self.slitlamp_preprocess = self._create_slitlamp_preprocessor()
        
        self._create_directories()
    
    def _create_directories(self):
        """Create output directory structure maintaining input folder structure"""
        dirs = [
            'processed_data/fundus',
            'processed_data/slitlamp',
            'processed_data/metadata'
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def _create_fundus_preprocessor(self):
        """
        Fundus preprocessing pipeline
        - Resize to 224x224
        - CLAHE contrast enhancement
        - Normalization to [-1, 1]
        """
        return A.Compose([
            A.Resize(224, 224),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    
    def _create_slitlamp_preprocessor(self):
        """
        Slit-lamp preprocessing pipeline
        - Resize to 224x224
        - Convert to grayscale (common for slit-lamp)
        - CLAHE for opacity enhancement
        - Single channel normalization
        """
        return A.Compose([
            A.Resize(224, 224),
            A.ToGray(p=1.0),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=(0.5,), std=(0.5,))
        ])
    
    def load_fundus_metadata(self, excel_path):
        """Load fundus dataset metadata from Excel"""
        try:
            df = pd.read_excel(excel_path)
            print(f"Loaded Excel with {len(df)} rows")
            
            # Map column names
            required_cols = ['ID', 'Left-Fundus', 'Right-Fundus', 
                           'Left-Diagnostic Keywords', 'Right-Diagnostic Keywords']
            
            column_mapping = {}
            for req_col in required_cols:
                matches = [col for col in df.columns if req_col.lower() in col.lower()]
                if matches:
                    column_mapping[req_col] = matches[0]
            
            df = df.rename(columns=column_mapping)
            self.fundus_data = df
            
            print(f"Successfully loaded {len(df)} fundus records")
            return df
            
        except Exception as e:
            print(f"Error loading Excel: {e}")
            return None
    
    def _extract_severity(self, keywords):
        """Extract cataract severity from diagnostic keywords"""
        if pd.isna(keywords) or keywords == '':
            return 'Normal'
        
        text = str(keywords).lower()
        
        if any(word in text for word in ['severe', 'advanced', 'dense', 'mature']):
            return 'Severe'
        elif any(word in text for word in ['moderate', 'significant']):
            return 'Moderate'
        elif any(word in text for word in ['mild', 'early', 'trace', 'slight']):
            return 'Mild'
        elif any(word in text for word in ['normal', 'clear', 'healthy']):
            return 'Normal'
        else:
            return 'Mild' if 'cataract' in text else 'Normal'
    
    def process_fundus_image(self, image_path):
        """Process single fundus image with medical preprocessing"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply fundus preprocessing pipeline
            processed = self.fundus_preprocess(image=image)['image']
            
            return processed
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def process_slitlamp_image(self, image_path):
        """Process single slit-lamp image with medical preprocessing"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply slit-lamp preprocessing pipeline
            processed = self.slitlamp_preprocess(image=image)['image']
            
            return processed
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def save_with_folder_structure(self, df, dataset_type='fundus'):
        """Save processed images with label-based folder structure"""
        print(f"Saving {len(df)} {dataset_type} images with folder structure...")
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            # Create label folder
            label_folder = os.path.join(f'processed_data/{dataset_type}', row['label'])
            os.makedirs(label_folder, exist_ok=True)
            
            # Copy processed image to label folder
            if os.path.exists(row['filepath']):
                # Get original filename without path
                original_filename = os.path.basename(row['filepath'])
                new_path = os.path.join(label_folder, original_filename)
                
                # If file doesn't exist at new location, copy it
                if not os.path.exists(new_path):
                    if dataset_type == 'fundus':
                        # For fundus, copy the processed image
                        processed_img = cv2.imread(row['filepath'])
                        cv2.imwrite(new_path, processed_img)
                    else:
                        # For slit-lamp, reprocess if needed
                        original_img_path = row['original_filepath']
                        processed_img = self.process_slitlamp_image(original_img_path)
                        if processed_img is not None:
                            img_to_save = (processed_img * 0.5 + 0.5) * 255
                            img_to_save = img_to_save.astype(np.uint8)
                            cv2.imwrite(new_path, img_to_save)
        
        print(f"Saved {len(df)} images to label-based folders")
        
        return df
    
    def process_fundus_images(self, images_folder):
        """Process all fundus images using Excel metadata and save them with folder structure"""
        if self.fundus_data is None:
            print("Error: Load Excel data first")
            return None
        
        processed_data = []
        print(f"Processing fundus images from: {images_folder}")
        
        for _, row in tqdm(self.fundus_data.iterrows(), total=len(self.fundus_data)):
            # Process left eye
            if not pd.isna(row['Left-Fundus']):
                left_path = os.path.join(images_folder, row['Left-Fundus'])
                if os.path.exists(left_path):
                    label = self._extract_severity(row['Left-Diagnostic Keywords'])
                    processed_img = self.process_fundus_image(left_path)
                    
                    if processed_img is not None:
                        # Save processed image
                        save_filename = f"{row['ID']}_Left_{row['Left-Fundus']}"
                        save_path = os.path.join('processed_data/fundus', save_filename)
                        
                        # Convert normalized image back for saving
                        img_to_save = processed_img.copy()
                        img_to_save = (img_to_save * 0.5 + 0.5) * 255
                        img_to_save = img_to_save.astype(np.uint8)
                        cv2.imwrite(save_path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
                        
                        processed_data.append({
                            'patient_id': row['ID'],
                            'eye': 'Left',
                            'original_filename': row['Left-Fundus'],
                            'processed_filename': save_filename,
                            'label': label,
                            'filepath': save_path,
                            'original_filepath': left_path
                        })
            
            # Process right eye
            if not pd.isna(row['Right-Fundus']):
                right_path = os.path.join(images_folder, row['Right-Fundus'])
                if os.path.exists(right_path):
                    label = self._extract_severity(row['Right-Diagnostic Keywords'])
                    processed_img = self.process_fundus_image(right_path)
                    
                    if processed_img is not None:
                        # Save processed image
                        save_filename = f"{row['ID']}_Right_{row['Right-Fundus']}"
                        save_path = os.path.join('processed_data/fundus', save_filename)
                        
                        # Convert normalized image back for saving
                        img_to_save = processed_img.copy()
                        img_to_save = (img_to_save * 0.5 + 0.5) * 255
                        img_to_save = img_to_save.astype(np.uint8)
                        cv2.imwrite(save_path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
                        
                        processed_data.append({
                            'patient_id': row['ID'],
                            'eye': 'Right',
                            'original_filename': row['Right-Fundus'],
                            'processed_filename': save_filename,
                            'label': label,
                            'filepath': save_path,
                            'original_filepath': right_path
                        })
        
        processed_df = pd.DataFrame(processed_data)
        
        # Save with folder structure
        self.save_with_folder_structure(processed_df, 'fundus')
        
        # Save metadata
        metadata_path = 'processed_data/metadata/fundus_metadata.csv'
        processed_df.to_csv(metadata_path, index=False)
        
        print(f"\nFundus processing completed:")
        print(f"Total: {len(processed_df)}")
        print(f"Distribution:\n{processed_df['label'].value_counts()}")
        print(f"Metadata saved to: {metadata_path}")
        
        return processed_df
    
    def process_slitlamp_images(self, slitlamp_folder):
        """Process slit-lamp images from organized folders and save with folder structure"""
        categories = ['normal', 'mature', 'immature']
        processed_data = []
        
        print(f"Processing slit-lamp images from: {slitlamp_folder}")
        
        for category in categories:
            category_path = os.path.join(slitlamp_folder, category)
            
            if not os.path.exists(category_path):
                print(f"Warning: '{category}' folder not found")
                continue
            
            image_files = [f for f in os.listdir(category_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            print(f"Processing {len(image_files)} images in '{category}'...")
            
            for image_file in tqdm(image_files):
                image_path = os.path.join(category_path, image_file)
                processed_img = self.process_slitlamp_image(image_path)
                
                if processed_img is not None:
                    # Save processed image
                    save_filename = f"processed_{image_file}"
                    save_path = os.path.join('processed_data/slitlamp', save_filename)
                    
                    # Convert normalized image back for saving
                    img_to_save = processed_img.copy()
                    img_to_save = (img_to_save * 0.5 + 0.5) * 255
                    img_to_save = img_to_save.astype(np.uint8)
                    cv2.imwrite(save_path, img_to_save)
                    
                    processed_data.append({
                        'original_filename': image_file,
                        'processed_filename': save_filename,
                        'label': category,
                        'filepath': save_path,
                        'original_filepath': image_path
                    })
        
        processed_df = pd.DataFrame(processed_data)
        
        # Save with folder structure
        self.save_with_folder_structure(processed_df, 'slitlamp')
        
        # Save metadata
        metadata_path = 'processed_data/metadata/slitlamp_metadata.csv'
        processed_df.to_csv(metadata_path, index=False)
        
        print(f"\nSlit-lamp processing completed:")
        print(f"Total: {len(processed_df)}")
        print(f"Distribution:\n{processed_df['label'].value_counts()}")
        print(f"Metadata saved to: {metadata_path}")
        
        return processed_df