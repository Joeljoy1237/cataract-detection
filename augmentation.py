"""
augmentation.py - Medical Image Augmentation Module
Professional augmentation techniques for medical imaging
"""

import numpy as np
import cv2
import os
from tqdm import tqdm
import albumentations as A

class CataractAugmentation:
    """
    Medical-grade data augmentation for cataract detection
    Separate pipelines for fundus and slit-lamp images
    """
    
    def __init__(self, augmentation_factor=2):
        self.augmentation_factor = augmentation_factor
        self.fundus_augmentor = self._create_fundus_augmentor()
        self.slitlamp_augmentor = self._create_slitlamp_augmentor()
    
    def _create_fundus_augmentor(self):
        """
        Fundus augmentation pipeline
        Includes rotations, flips, lighting variations, color jitter, blur, noise, occlusion
        """
        return A.Compose([
            A.Rotate(limit=20, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.3),
            A.CoarseDropout(
                max_holes=1, 
                max_height=32, 
                max_width=32, 
                fill_value=0,
                p=0.3
            )
        ])
    
    def _create_slitlamp_augmentor(self):
        """
        Slit-lamp augmentation pipeline
        Conservative augmentation: rotations, flips, intensity, motion blur, defocus, noise, occlusion
        """
        return A.Compose([
            A.Rotate(limit=10, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.MotionBlur(blur_limit=3, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.3),
            A.CoarseDropout(
                max_holes=1,
                max_height=20,
                max_width=20,
                fill_value=0,
                p=0.3
            )
        ])
    
    def augment_image(self, image, image_type='fundus'):
        """Apply augmentation based on image type"""
        if image_type == 'fundus':
            return self.fundus_augmentor(image=image)['image']
        else:
            return self.slitlamp_augmentor(image=image)['image']
    
    def augment_folder(self, input_folder, output_folder, image_type='fundus'):
        """Augment all images in folder structure"""
        os.makedirs(output_folder, exist_ok=True)
        
        categories = [d for d in os.listdir(input_folder) 
                     if os.path.isdir(os.path.join(input_folder, d))]
        
        print(f"Augmenting {image_type} images...")
        
        for category in categories:
            input_cat = os.path.join(input_folder, category)
            output_cat = os.path.join(output_folder, category)
            os.makedirs(output_cat, exist_ok=True)
            
            image_files = [f for f in os.listdir(input_cat)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"Processing {category}: {len(image_files)} images")
            
            for img_file in tqdm(image_files):
                input_path = os.path.join(input_cat, img_file)
                image = cv2.imread(input_path)
                
                if image is None:
                    continue
                
                # Handle grayscale (slit-lamp) vs RGB (fundus)
                if len(image.shape) == 2:  # Grayscale
                    pass  # Keep as is
                else:  # RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Copy original
                orig_output = os.path.join(output_cat, f"orig_{img_file}")
                if len(image.shape) == 2:
                    cv2.imwrite(orig_output, image)
                else:
                    cv2.imwrite(orig_output, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                
                # Create augmented versions
                for i in range(self.augmentation_factor):
                    augmented = self.augment_image(image, image_type)
                    
                    aug_filename = f"aug_{i}_{img_file}"
                    aug_output = os.path.join(output_cat, aug_filename)
                    
                    if len(augmented.shape) == 2:
                        cv2.imwrite(aug_output, augmented)
                    else:
                        cv2.imwrite(aug_output, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
            
            print(f"  Created {len(image_files) * (self.augmentation_factor + 1)} images")
    
    def print_summary(self):
        """Print augmentation techniques summary"""
        print("\n=== MEDICAL IMAGE AUGMENTATION SUMMARY ===")
        print(f"Augmentation Factor: {self.augmentation_factor}x")
        print("\nFundus Augmentation Techniques:")
        print("  • Rotation (±20°)")
        print("  • Horizontal Flip")
        print("  • Brightness/Contrast Adjustment")
        print("  • Hue/Saturation/Value (Color Jitter)")
        print("  • Gaussian Blur (Defocus)")
        print("  • Gaussian Noise (Camera Noise)")
        print("  • Cutout/Occlusion (32x32 patches)")
        print("\nSlit-lamp Augmentation Techniques:")
        print("  • Rotation (±10°, conservative)")
        print("  • Horizontal Flip")
        print("  • Brightness/Contrast (Lamp Intensity)")
        print("  • Motion Blur (Eye Movement)")
        print("  • Gaussian Blur (Defocus)")
        print("  • Gaussian Noise (Camera Noise)")
        print("  • Cutout/Occlusion (20x20, Eyelash)")