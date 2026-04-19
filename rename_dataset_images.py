"""
MedEye Dataset Image Renaming Script
Renames all disease images with disease name and sequence numbering
Author: Muhammad Daud
License: MIT Open Source

Usage: python rename_dataset_images.py
"""

import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disease categories
DISEASES = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

# Dataset base path
DATASET_PATH = Path(__file__).parent / 'dataset'


def rename_images_in_disease_folder(disease_name):
    """
    Rename all images in a disease folder with pattern: disease_name_sequence.extension
    
    Args:
        disease_name (str): Name of the disease folder
    
    Returns:
        tuple: (success_count, error_count)
    """
    disease_path = DATASET_PATH / disease_name
    
    if not disease_path.exists():
        logger.warning(f"Disease folder not found: {disease_path}")
        return 0, 0
    
    # Get all image files (jpg, png, jpeg)
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = sorted([
        f for f in disease_path.iterdir() 
        if f.is_file() and f.suffix in image_extensions
    ])
    
    success_count = 0
    error_count = 0
    
    logger.info(f"\nProcessing {disease_name} folder ({len(image_files)} images)...")
    
    # Rename each image with sequence numbering
    for index, old_file in enumerate(image_files, start=1):
        try:
            # Create new filename: disease_name_sequence.extension
            new_name = f"{disease_name}_{index}{old_file.suffix}"
            new_file_path = disease_path / new_name
            
            # Rename the file
            old_file.rename(new_file_path)
            logger.info(f"  ✓ {old_file.name} → {new_name}")
            success_count += 1
            
        except Exception as e:
            logger.error(f"  ✗ Error renaming {old_file.name}: {str(e)}")
            error_count += 1
    
    return success_count, error_count


def main():
    """Main function to rename all disease images"""
    
    print("\n" + "="*70)
    print("MedEye Dataset Image Renaming Tool")
    print("="*70)
    print(f"\nDataset location: {DATASET_PATH}")
    print(f"Total diseases: {len(DISEASES)}")
    print(f"Diseases: {', '.join(DISEASES)}\n")
    
    # Confirm before renaming
    confirm = input("This will rename ALL images in the dataset folders.\nContinue? (yes/no): ").lower().strip()
    
    if confirm != 'yes':
        print("Operation cancelled.")
        return
    
    total_success = 0
    total_errors = 0
    
    # Rename images in each disease folder
    for disease in DISEASES:
        success, errors = rename_images_in_disease_folder(disease)
        total_success += success
        total_errors += errors
    
    # Print summary
    print("\n" + "="*70)
    print("RENAMING SUMMARY")
    print("="*70)
    print(f"Total images renamed: {total_success}")
    print(f"Total errors: {total_errors}")
    print("="*70 + "\n")
    
    if total_errors == 0:
        print("✅ All images renamed successfully!")
    else:
        print(f"⚠️  Completed with {total_errors} error(s)")


if __name__ == "__main__":
    main()
