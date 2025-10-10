"""
preprocessor.py - Medical Image Preprocessing Module
Enhanced version with robust error handling and improved functionality
"""

import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm
import albumentations as A
from typing import Optional, Dict, List, Tuple, Any
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CataractPreprocessor:
    """
    A comprehensive preprocessor for cataract medical images including fundus and slit-lamp images.
    Handles image preprocessing, metadata management, and data organization.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the CataractPreprocessor.
        
        Args:
            target_size (tuple): Target size for resizing images (height, width)
        """
        self.target_size = target_size
        self.fundus_data = None
        self.fundus_preprocess = self._create_fundus_preprocessor()
        self.slitlamp_preprocess = self._create_slitlamp_preprocessor()
        self._create_directories()
        logger.info(f"CataractPreprocessor initialized with target size: {target_size}")

    def _create_directories(self) -> None:
        """Create necessary directories with comprehensive error handling."""
        directories = [
            "processed_data/fundus",
            "processed_data/slitlamp", 
            "processed_data/metadata",
            "processed_data/fundus/Normal",
            "processed_data/fundus/Mild",
            "processed_data/fundus/Moderate",
            "processed_data/fundus/Severe",
            "processed_data/slitlamp/normal",
            "processed_data/slitlamp/mature",
            "processed_data/slitlamp/immature"
        ]
        
        for dir_path in directories:
            try:
                os.makedirs(dir_path, exist_ok=True)
                logger.debug(f"Created/verified directory: {dir_path}")
            except Exception as e:
                logger.error(f"Failed to create directory {dir_path}: {e}")
                raise

    def _create_fundus_preprocessor(self) -> A.Compose:
        """Create albumentations pipeline for fundus image preprocessing."""
        return A.Compose([
            A.Resize(self.target_size[0], self.target_size[1]),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def _create_slitlamp_preprocessor(self) -> A.Compose:
        """Create albumentations pipeline for slit-lamp image preprocessing."""
        return A.Compose([
            A.Resize(self.target_size[0], self.target_size[1]),
            A.ToGray(p=1.0),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=(0.5,), std=(0.5,))
        ])

    def load_fundus_metadata(self, excel_path: str) -> Optional[pd.DataFrame]:
        """
        Load and validate fundus metadata from Excel file.
        
        Args:
            excel_path (str): Path to the Excel metadata file
            
        Returns:
            pd.DataFrame or None: Loaded dataframe or None if failed
        """
        try:
            if not os.path.exists(excel_path):
                logger.error(f"Excel file not found: {excel_path}")
                return None
                
            df = pd.read_excel(excel_path)
            logger.info(f"Raw Excel loaded with {len(df)} rows and {len(df.columns)} columns")
            
            required_cols = [
                "ID", 
                "Left-Fundus", 
                "Right-Fundus", 
                "Left-Diagnostic Keywords", 
                "Right-Diagnostic Keywords"
            ]

            # Robust column mapping with multiple matching strategies
            column_mapping = {}
            missing_cols = []
            available_cols = list(df.columns)
            
            for req_col in required_cols:
                # Try multiple matching strategies
                possible_matches = []
                
                # Exact match (case insensitive)
                exact_matches = [col for col in available_cols if col.lower() == req_col.lower()]
                if exact_matches:
                    possible_matches.extend(exact_matches)
                
                # Partial match with different separators
                partial_matches = [col for col in available_cols 
                                 if req_col.lower().replace('-', ' ') in col.lower().replace('_', ' ').replace('-', ' ')]
                if partial_matches:
                    possible_matches.extend(partial_matches)
                
                # Keyword-based matching
                keywords = req_col.lower().split('-')
                keyword_matches = [col for col in available_cols 
                                 if all(keyword in col.lower() for keyword in keywords)]
                if keyword_matches:
                    possible_matches.extend(keyword_matches)
                
                # Remove duplicates and select best match
                possible_matches = list(set(possible_matches))
                if possible_matches:
                    # Prefer exact matches
                    exact = [m for m in possible_matches if m.lower() == req_col.lower()]
                    if exact:
                        column_mapping[req_col] = exact[0]
                    else:
                        column_mapping[req_col] = possible_matches[0]
                    logger.debug(f"Mapped '{req_col}' -> '{column_mapping[req_col]}'")
                else:
                    missing_cols.append(req_col)
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                logger.info(f"Available columns: {available_cols}")
                return None
            
            # Apply column mapping
            df = df.rename(columns=column_mapping)
            
            # Validate and clean data
            if 'ID' in df.columns:
                if not pd.api.types.is_numeric_dtype(df['ID']):
                    logger.warning("ID column is not numeric, attempting conversion")
                    df['ID'] = pd.to_numeric(df['ID'], errors='coerce')
            
            # Remove rows with missing critical data
            initial_count = len(df)
            df = df.dropna(subset=['ID'])
            if len(df) < initial_count:
                logger.warning(f"Removed {initial_count - len(df)} rows with missing ID")
            
            self.fundus_data = df
            logger.info(f"Successfully loaded fundus metadata: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading Excel file {excel_path}: {e}")
            return None

    def _extract_severity(self, keywords: str) -> str:
        """
        Extract cataract severity from diagnostic keywords with comprehensive logic.
        
        Args:
            keywords (str): Diagnostic keywords text
            
        Returns:
            str: Severity classification (Normal, Mild, Moderate, Severe)
        """
        if pd.isna(keywords) or keywords == "" or str(keywords).lower() == "nan":
            return "Normal"
            
        text = str(keywords).lower().strip()
        
        # Define severity patterns with comprehensive keywords
        severity_patterns = {
            "Severe": [
                "severe", "advanced", "dense", "mature", "hyper-mature", 
                "hypermature", "grade 4", "grade iv", "grade iv", "brunescent"
            ],
            "Moderate": [
                "moderate", "significant", "grade 3", "grade iii", 
                "nuclear grade 3", "cortical grade 3"
            ],
            "Mild": [
                "mild", "early", "trace", "slight", "grade 1", "grade i", 
                "incipient", "minimal", "grade 2", "grade ii"
            ],
            "Normal": [
                "normal", "clear", "healthy", "no dr", "no diabetic retinopathy",
                "no abnormality", "within normal limits"
            ]
        }
        
        # Check for cataract mention
        has_cataract = any(word in text for word in ["cataract", "lens opacity", "lens opacities"])
        
        # Check severity patterns in order of severity (most severe first)
        for severity, patterns in severity_patterns.items():
            if any(pattern in text for pattern in patterns):
                return severity
        
        # Default classification if no patterns match
        if has_cataract:
            logger.debug(f"Defaulting to 'Mild' for cataract without clear severity: {text}")
            return "Mild"
        else:
            return "Normal"

    def process_fundus_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Process fundus image with comprehensive error handling.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            np.ndarray or None: Processed image or None if failed
        """
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image file not found: {image_path}")
                return None
                
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Failed to read image (cv2.imread returned None): {image_path}")
                return None
                
            if image.size == 0:
                logger.warning(f"Empty image (0 size): {image_path}")
                return None
            
            # Check image dimensions
            if image.shape[0] < 50 or image.shape[1] < 50:
                logger.warning(f"Image too small: {image_path} - Shape: {image.shape}")
                return None
                
            # Convert color space
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply preprocessing
            processed = self.fundus_preprocess(image=image)["image"]
            
            # Validate processed image
            if processed is None or processed.size == 0:
                logger.warning(f"Processing resulted in empty image: {image_path}")
                return None
                
            if not isinstance(processed, np.ndarray):
                logger.warning(f"Processing resulted in non-array output: {image_path}")
                return None
                
            logger.debug(f"Successfully processed fundus image: {image_path}")
            return processed
            
        except Exception as e:
            logger.error(f"Error processing fundus image {image_path}: {e}")
            return None

    def process_slitlamp_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Process slit-lamp image with comprehensive error handling.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            np.ndarray or None: Processed image or None if failed
        """
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image file not found: {image_path}")
                return None
                
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Failed to read image: {image_path}")
                return None
                
            if image.size == 0:
                logger.warning(f"Empty image: {image_path}")
                return None
                
            # Check image dimensions
            if image.shape[0] < 50 or image.shape[1] < 50:
                logger.warning(f"Image too small: {image_path} - Shape: {image.shape}")
                return None
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            processed = self.slitlamp_preprocess(image=image)["image"]
            
            # Convert single-channel → 3-channel for DenseNet compatibility
            if len(processed.shape) == 2:
                processed = np.stack([processed] * 3, axis=-1)
            elif len(processed.shape) == 3 and processed.shape[2] == 1:
                processed = np.repeat(processed, 3, axis=2)
                
            # Validate processed image
            if processed is None or processed.size == 0:
                logger.warning(f"Processing resulted in empty image: {image_path}")
                return None
                
            if not isinstance(processed, np.ndarray):
                logger.warning(f"Processing resulted in non-array output: {image_path}")
                return None
                
            logger.debug(f"Successfully processed slit-lamp image: {image_path}")
            return processed
            
        except Exception as e:
            logger.error(f"Error processing slit-lamp image {image_path}: {e}")
            return None

    def process_fundus_images(self, images_folder: str) -> Optional[pd.DataFrame]:
        """
        Process all fundus images based on loaded metadata.
        
        Args:
            images_folder (str): Path to folder containing fundus images
            
        Returns:
            pd.DataFrame or None: Metadata of processed images or None if failed
        """
        if self.fundus_data is None:
            logger.error("Fundus metadata not loaded. Call load_fundus_metadata() first.")
            return None
            
        if not os.path.exists(images_folder):
            logger.error(f"Images folder not found: {images_folder}")
            return None

        processed_data = []
        skipped_images = 0
        processed_count = 0
        
        logger.info(f"Starting fundus image processing from: {images_folder}")
        
        for _, row in tqdm(self.fundus_data.iterrows(), total=len(self.fundus_data), desc="Processing fundus images"):
            for eye in ["Left", "Right"]:
                image_col = f"{eye}-Fundus"
                keywords_col = f"{eye}-Diagnostic Keywords"
                
                if image_col not in row.index or keywords_col not in row.index:
                    logger.warning(f"Missing required columns for patient {row.get('ID', 'Unknown')}")
                    continue
                    
                if not pd.isna(row[image_col]) and row[image_col] not in ["", " "]:
                    img_filename = row[image_col]
                    img_path = os.path.join(images_folder, img_filename)
                    
                    if os.path.exists(img_path):
                        label = self._extract_severity(row[keywords_col])
                        processed_img = self.process_fundus_image(img_path)
                        
                        if processed_img is not None:
                            # Create label-specific directory
                            label_folder = os.path.join("processed_data/fundus", label)
                            os.makedirs(label_folder, exist_ok=True)
                            
                            # Generate unique filename
                            original_name = os.path.splitext(img_filename)[0]
                            extension = os.path.splitext(img_filename)[1] or '.png'
                            filename = f"{row['ID']}_{eye}_{original_name}_processed{extension}"
                            save_path = os.path.join(label_folder, filename)
                            
                            # Convert normalized image back to uint8 for saving
                            img_to_save = (processed_img * 0.5 + 0.5) * 255
                            img_to_save = np.clip(img_to_save, 0, 255).astype(np.uint8)
                            
                            # Save image
                            cv2.imwrite(save_path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
                            
                            processed_data.append({
                                "patient_id": row['ID'],
                                "eye": eye,
                                "original_filename": img_filename,
                                "processed_filename": filename,
                                "label": label,
                                "filepath": save_path,
                                "original_filepath": img_path,
                                "diagnostic_keywords": row[keywords_col]
                            })
                            processed_count += 1
                        else:
                            skipped_images += 1
                            logger.debug(f"Skipped processing image: {img_path}")
                    else:
                        skipped_images += 1
                        logger.debug(f"Image file not found: {img_path}")

        # Create metadata dataframe
        if processed_data:
            df = pd.DataFrame(processed_data)
            metadata_path = "processed_data/metadata/fundus_metadata.csv"
            df.to_csv(metadata_path, index=False)
            
            logger.info(f"Fundus preprocessing complete: {processed_count} images processed, {skipped_images} skipped")
            logger.info(f"Metadata saved to: {metadata_path}")
            
            # Log distribution
            label_distribution = df['label'].value_counts()
            logger.info("Fundus image distribution by severity:")
            for label, count in label_distribution.items():
                logger.info(f"  {label}: {count} images")
                
            return df
        else:
            logger.error("No fundus images were successfully processed")
            return None

    def process_slitlamp_images(self, slitlamp_folder: str) -> Optional[pd.DataFrame]:
        """
        Process all slit-lamp images from categorized folders.
        
        Args:
            slitlamp_folder (str): Path to folder containing slit-lamp image categories
            
        Returns:
            pd.DataFrame or None: Metadata of processed images or None if failed
        """
        if not os.path.exists(slitlamp_folder):
            logger.error(f"Slit-lamp folder not found: {slitlamp_folder}")
            return None

        categories = ["normal", "mature", "immature"]
        processed_data = []
        skipped_images = 0
        processed_count = 0
        
        logger.info(f"Starting slit-lamp image processing from: {slitlamp_folder}")
        
        for category in categories:
            cat_path = os.path.join(slitlamp_folder, category)
            if not os.path.exists(cat_path):
                logger.warning(f"Category folder not found: {cat_path}")
                continue
                
            # Get all image files
            image_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]:
                image_files.extend([f for f in os.listdir(cat_path) if f.lower().endswith(ext[1:])])
            
            logger.info(f"Processing {len(image_files)} images in category: {category}")
            
            for img_file in tqdm(image_files, desc=f"Processing {category}"):
                img_path = os.path.join(cat_path, img_file)
                processed_img = self.process_slitlamp_image(img_path)
                
                if processed_img is not None:
                    # Create category directory
                    label_folder = os.path.join("processed_data/slitlamp", category)
                    os.makedirs(label_folder, exist_ok=True)
                    
                    # Generate unique filename
                    original_name = os.path.splitext(img_file)[0]
                    extension = os.path.splitext(img_file)[1] or '.png'
                    save_filename = f"{category}_{original_name}_processed{extension}"
                    save_path = os.path.join(label_folder, save_filename)
                    
                    # Convert normalized image back to uint8 for saving
                    img_to_save = (processed_img * 0.5 + 0.5) * 255
                    img_to_save = np.clip(img_to_save, 0, 255).astype(np.uint8)
                    
                    # Save image
                    cv2.imwrite(save_path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
                    
                    processed_data.append({
                        "original_filename": img_file,
                        "processed_filename": save_filename,
                        "label": category,
                        "filepath": save_path,
                        "original_filepath": img_path
                    })
                    processed_count += 1
                else:
                    skipped_images += 1
                    logger.debug(f"Skipped processing image: {img_path}")

        # Create metadata dataframe
        if processed_data:
            df = pd.DataFrame(processed_data)
            metadata_path = "processed_data/metadata/slitlamp_metadata.csv"
            df.to_csv(metadata_path, index=False)
            
            logger.info(f"Slit-lamp preprocessing complete: {processed_count} images processed, {skipped_images} skipped")
            logger.info(f"Metadata saved to: {metadata_path}")
            
            # Log distribution
            label_distribution = df['label'].value_counts()
            logger.info("Slit-lamp image distribution by category:")
            for label, count in label_distribution.items():
                logger.info(f"  {label}: {count} images")
                
            return df
        else:
            logger.error("No slit-lamp images were successfully processed")
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about processed data.
        
        Returns:
            dict: Statistics for fundus and slit-lamp images
        """
        stats = {}
        
        # Fundus statistics
        fundus_meta_path = "processed_data/metadata/fundus_metadata.csv"
        if os.path.exists(fundus_meta_path):
            try:
                fundus_df = pd.read_csv(fundus_meta_path)
                stats['fundus'] = {
                    'total_processed': len(fundus_df),
                    'by_severity': fundus_df['label'].value_counts().to_dict(),
                    'by_eye': fundus_df['eye'].value_counts().to_dict() if 'eye' in fundus_df.columns else {},
                    'patients_processed': fundus_df['patient_id'].nunique() if 'patient_id' in fundus_df.columns else 0
                }
            except Exception as e:
                logger.error(f"Error reading fundus metadata: {e}")
                stats['fundus'] = {'error': str(e)}
        
        # Slitlamp statistics
        slitlamp_meta_path = "processed_data/metadata/slitlamp_metadata.csv"
        if os.path.exists(slitlamp_meta_path):
            try:
                slitlamp_df = pd.read_csv(slitlamp_meta_path)
                stats['slitlamp'] = {
                    'total_processed': len(slitlamp_df),
                    'by_category': slitlamp_df['label'].value_counts().to_dict()
                }
            except Exception as e:
                logger.error(f"Error reading slit-lamp metadata: {e}")
                stats['slitlamp'] = {'error': str(e)}
        
        return stats

    def cleanup(self) -> None:
        """Clean up resources and reset processor state."""
        self.fundus_data = None
        logger.info("CataractPreprocessor cleaned up")

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self.cleanup()


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    preprocessor = CataractPreprocessor(target_size=(224, 224))
    
    # Process fundus images
    try:
        # Load metadata
        df = preprocessor.load_fundus_metadata("path/to/your/metadata.xlsx")
        
        if df is not None:
            # Process images
            fundus_meta = preprocessor.process_fundus_images("path/to/fundus/images")
            
            # Process slit-lamp images
            slitlamp_meta = preprocessor.process_slitlamp_images("path/to/slitlamp/images")
            
            # Get statistics
            stats = preprocessor.get_statistics()
            print("Processing Statistics:", stats)
            
    except Exception as e:
        logger.error(f"Error in example usage: {e}")
    
    finally:
        preprocessor.cleanup()