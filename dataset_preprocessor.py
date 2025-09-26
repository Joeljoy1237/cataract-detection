import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

class MinimalCataractPreprocessor:
    """
    Minimal cataract image preprocessor using only essential Excel columns:
    - Index (ID)
    - Left Eye image filename
    - Right Eye image filename  
    - Category/diagnostic keywords for severity classification
    
    Applies 4 preprocessing techniques:
    1. Resize to 224x224
    2. Noise removal (Gaussian blur)
    3. Contrast enhancement (CLAHE)
    4. Normalization to [0,1]
    """
    
    def __init__(self, target_size=(224, 224)):
        """
        Initialize preprocessor
        Args:
            target_size: Target image dimensions (height, width)
        """
        self.target_size = target_size
        self.fundus_data = None
        
        # Create output directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary output folders"""
        dirs = [
            'processed_data',
            'processed_data/train', 'processed_data/val', 'processed_data/test'
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def load_excel_data(self, excel_path):
        """
        Load only essential columns from Excel file
        Expected columns: ID, Left-Fundus, Right-Fundus, Left/Right-Diagnostic Keywords
        """
        try:
            # Load Excel file
            df = pd.read_excel(excel_path)
            print(f"Loaded Excel with {len(df)} rows")
            print(f"Available columns: {df.columns.tolist()}")
            
            # Extract only needed columns (flexible column name matching)
            essential_data = []
            
            for _, row in df.iterrows():
                # Get patient ID (index)
                patient_id = row['ID']
                
                # Get left eye data
                left_filename = row['Left-Fundus'] if not pd.isna(row['Left-Fundus']) else None
                left_category = self._extract_severity(row['Left-Diagnostic Keywords']) if not pd.isna(row['Left-Diagnostic Keywords']) else 'Normal'
                
                # Get right eye data  
                right_filename = row['Right-Fundus'] if not pd.isna(row['Right-Fundus']) else None
                right_category = self._extract_severity(row['Right-Diagnostic Keywords']) if not pd.isna(row['Right-Diagnostic Keywords']) else 'Normal'
                
                # Add left eye record if filename exists
                if left_filename:
                    essential_data.append({
                        'index': patient_id,
                        'eye': 'left',
                        'filename': left_filename,
                        'category': left_category
                    })
                
                # Add right eye record if filename exists
                if right_filename:
                    essential_data.append({
                        'index': patient_id,
                        'eye': 'right', 
                        'filename': right_filename,
                        'category': right_category
                    })
            
            # Convert to DataFrame with only essential columns
            self.fundus_data = pd.DataFrame(essential_data)
            
            print(f"Extracted {len(self.fundus_data)} eye images")
            print(f"Category distribution:")
            print(self.fundus_data['category'].value_counts())
            
            return self.fundus_data
            
        except Exception as e:
            print(f"Error loading Excel: {e}")
            return None
    
    def _extract_severity(self, keywords):
        """
        Extract severity category from diagnostic keywords
        Args:
            keywords: Diagnostic text from Excel
        Returns:
            severity: 'Normal', 'Mild', 'Moderate', or 'Severe'
        """
        if pd.isna(keywords) or keywords == '':
            return 'Normal'
        
        # Convert to lowercase for matching
        text = str(keywords).lower()
        
        # Check severity levels (from most severe to least)
        if any(word in text for word in ['severe', 'advanced', 'dense', 'mature']):
            return 'Severe'
        elif any(word in text for word in ['moderate', 'significant']):
            return 'Moderate'  
        elif any(word in text for word in ['mild', 'early', 'trace', 'slight']):
            return 'Mild'
        elif any(word in text for word in ['normal', 'clear', 'healthy']):
            return 'Normal'
        else:
            # Default: if 'cataract' mentioned but no severity, assume mild
            return 'Mild' if 'cataract' in text else 'Normal'
    
    def preprocess_single_image(self, image_path):
        """
        Apply 4 preprocessing techniques to single image
        Args:
            image_path: Path to image file
        Returns:
            processed_image: Preprocessed image array
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert BGR to RGB (OpenCV loads as BGR by default)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # TECHNIQUE 1: RESIZE
            # Resize to standard dimensions for neural network input
            resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
            
            # TECHNIQUE 2: NOISE REMOVAL  
            # Apply Gaussian blur to reduce noise while preserving edges
            denoised = cv2.GaussianBlur(resized, (5, 5), 0)
            
            # TECHNIQUE 3: CONTRAST ENHANCEMENT
            # Use CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # Convert to LAB color space for better contrast processing
            lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])  # Apply only to luminance channel
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # TECHNIQUE 4: NORMALIZATION
            # Scale pixel values from [0-255] to [0-1] for neural network training
            normalized = enhanced.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def process_all_images(self, images_folder):
        """
        Process all images from the dataset
        Args:
            images_folder: Folder containing all image files
        """
        if self.fundus_data is None:
            print("Error: No Excel data loaded. Call load_excel_data() first.")
            return None
        
        processed_data = []
        
        print(f"Processing images from: {images_folder}")
        
        # Process each image record
        for _, row in tqdm(self.fundus_data.iterrows(), total=len(self.fundus_data)):
            # Build full image path
            image_path = os.path.join(images_folder, row['filename'])
            
            # Check if image file exists
            if not os.path.exists(image_path):
                continue
            
            # Apply preprocessing techniques
            processed_image = self.preprocess_single_image(image_path)
            
            if processed_image is not None:
                # Store processed data with only essential information
                processed_data.append({
                    'index': row['index'],        # Patient ID
                    'eye': row['eye'],            # Left or Right
                    'filename': row['filename'],   # Original filename
                    'category': row['category'],   # Severity category
                    'processed_image': processed_image  # Preprocessed image array
                })
        
        # Convert to DataFrame
        processed_df = pd.DataFrame(processed_data)
        
        # Print processing summary
        print(f"\nProcessing Summary:")
        print(f"Total images processed: {len(processed_df)}")
        print(f"Category distribution:")
        print(processed_df['category'].value_counts())
        
        return processed_df
    
    def split_and_save_data(self, processed_df):
        """
        Split processed data into train/validation/test sets and save
        Args:
            processed_df: DataFrame with processed images
        """
        if processed_df is None or len(processed_df) == 0:
            print("No processed data to split")
            return
        
        # Split data: 70% train, 15% validation, 15% test
        # Use stratified split to maintain category distribution
        train_df, temp_df = train_test_split(
            processed_df, 
            test_size=0.3,  # 30% for temp (will be split into val and test)
            stratify=processed_df['category'],
            random_state=42
        )
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,  # Split temp 50/50 for val and test (15% each of total)
            stratify=temp_df['category'], 
            random_state=42
        )
        
        # Save each split
        splits = {
            'train': train_df,
            'val': val_df, 
            'test': test_df
        }
        
        for split_name, split_data in splits.items():
            # Create category folders
            for category in split_data['category'].unique():
                category_dir = os.path.join('processed_data', split_name, category)
                os.makedirs(category_dir, exist_ok=True)
            
            # Save images and create metadata
            metadata_records = []
            
            for _, row in split_data.iterrows():
                # Create filename with patient info
                save_filename = f"{row['index']}_{row['eye']}_{row['filename']}"
                save_path = os.path.join('processed_data', split_name, row['category'], save_filename)
                
                # Convert normalized image back to uint8 format for saving
                image_to_save = (row['processed_image'] * 255).astype(np.uint8)
                
                # Save image (convert RGB back to BGR for OpenCV)
                cv2.imwrite(save_path, cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR))
                
                # Record metadata
                metadata_records.append({
                    'index': row['index'],
                    'eye': row['eye'],
                    'original_filename': row['filename'],
                    'saved_filename': save_filename,
                    'category': row['category'],
                    'file_path': save_path
                })
            
            # Save metadata CSV
            metadata_df = pd.DataFrame(metadata_records)
            metadata_path = f'processed_data/{split_name}_metadata.csv'
            metadata_df.to_csv(metadata_path, index=False)
            
            print(f"{split_name.upper()}: {len(split_data)} images saved")
        
        print(f"\nData split completed:")
        print(f"Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")
        
        # Save summary report
        summary = {
            'total_images': len(processed_df),
            'train_count': len(train_df),
            'val_count': len(val_df),
            'test_count': len(test_df),
            'category_distribution': processed_df['category'].value_counts().to_dict(),
            'preprocessing_techniques': [
                '1. Resize to 224x224',
                '2. Gaussian blur noise removal',
                '3. CLAHE contrast enhancement', 
                '4. Normalization to [0,1]'
            ]
        }
        
        with open('processed_data/summary_report.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("Summary report saved to processed_data/summary_report.json")


def main():
    """
    Main function to run the preprocessing pipeline
    """
    print("MINIMAL CATARACT PREPROCESSING")
    print("Using only: Index + Left/Right Eye + Category")
    print("Techniques: Resize + Noise Removal + Contrast Enhancement + Normalization")
    print("=" * 70)
    
    # Initialize preprocessor
    preprocessor = MinimalCataractPreprocessor(target_size=(224, 224))
    
    # Set file paths (update these to match your structure)
    excel_file = "fundus/metadata.xlsx"
    images_folder = "fundus/Training Images"
    
    # Check if files exist
    if not os.path.exists(excel_file):
        print(f"Excel file not found: {excel_file}")
        return
    
    if not os.path.exists(images_folder):
        print(f"Images folder not found: {images_folder}")
        return
    
    # Step 1: Load essential data from Excel
    print("Step 1: Loading Excel data...")
    preprocessor.load_excel_data(excel_file)
    
    # Step 2: Process all images
    print("\nStep 2: Processing images...")
    processed_data = preprocessor.process_all_images(images_folder)
    
    # Step 3: Split and save data
    print("\nStep 3: Splitting and saving data...")
    preprocessor.split_and_save_data(processed_data)
    
    print("\nPreprocessing completed successfully!")
    print("Check 'processed_data/' folder for results.")


if __name__ == "__main__":
    main()