"""
╔════════════════════════════════════════════════════════════════════════════╗
║                            MEDEYE v2.0                                      ║
║                   Medical Eye Disease Detection System                       ║
║                                                                              ║
║  Developer: Muhammad Daud                                                    ║
║  Data Utilities Module - Data loading, validation, preprocessing            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
import logging

from config import (
    CLASS_NAMES, DATA_DIR, TRAIN_TEST_SPLIT, RANDOM_STATE,
    VALID_IMAGE_EXTENSIONS, IMAGE_NORMALIZATION, SKIP_CORRUPTED_IMAGES,
    LOG_CORRUPTED_IMAGES, EXPECTED_CLASS_DISTRIBUTION, MIN_IMAGE_WIDTH,
    MIN_IMAGE_HEIGHT, MAX_IMAGE_SIZE_MB
)
from logging_setup import get_logger

logger = get_logger(__name__)


class ImageValidator:
    """Validates image files for quality and integrity."""
    
    @staticmethod
    def is_valid_extension(filepath: str) -> bool:
        """Check if file has valid image extension."""
        ext = Path(filepath).suffix.lower().lstrip('.')
        return ext in VALID_IMAGE_EXTENSIONS
    
    @staticmethod
    def is_valid_file_size(filepath: str) -> bool:
        """Check if file size is within limits."""
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        return size_mb <= MAX_IMAGE_SIZE_MB
    
    @staticmethod
    def is_valid_image_dimensions(image: np.ndarray) -> bool:
        """Check if image dimensions are sufficient."""
        height, width = image.shape[:2]
        return width >= MIN_IMAGE_WIDTH and height >= MIN_IMAGE_HEIGHT
    
    @staticmethod
    def is_readable(filepath: str) -> bool:
        """Check if file can be read as image."""
        try:
            img = cv2.imread(filepath)
            return img is not None and len(img.shape) >= 2
        except Exception:
            return False


class ImageLoader:
    """Load and preprocess images from disk."""
    
    def __init__(self):
        self.validator = ImageValidator()
        self.corrupted_images = []
    
    def load_image(
        self,
        filepath: str,
        target_size: Tuple[int, int],
        grayscale: bool = False,
        normalize: bool = True
    ) -> Optional[np.ndarray]:
        """
        Load and preprocess a single image.
        
        Args:
            filepath (str): Path to image file
            target_size (tuple): Target image dimensions (height, width)
            grayscale (bool): Convert to grayscale
            normalize (bool): Normalize pixel values to [0, 1]
        
        Returns:
            np.ndarray: Preprocessed image or None if error
        """
        try:
            # Validation
            if not self.validator.is_valid_extension(filepath):
                logger.warning(f"Invalid extension: {filepath}")
                return None
            
            if not self.validator.is_valid_file_size(filepath):
                logger.warning(f"File too large: {filepath}")
                return None
            
            if not self.validator.is_readable(filepath):
                logger.warning(f"Cannot read image: {filepath}")
                self.corrupted_images.append(filepath)
                return None
            
            # Load image
            if grayscale:
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(filepath, cv2.IMREAD_COLOR)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if image is None:
                logger.warning(f"Failed to load image: {filepath}")
                self.corrupted_images.append(filepath)
                return None
            
            # Dimension validation
            if not self.validator.is_valid_image_dimensions(image):
                logger.warning(f"Image dimensions too small: {filepath}")
                return None
            
            # Resize
            image = cv2.resize(image, target_size)
            
            # Normalize
            if normalize:
                image = image.astype(np.float32) * IMAGE_NORMALIZATION
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {filepath}: {str(e)}")
            self.corrupted_images.append(filepath)
            return None
    
    def load_dataset(
        self,
        target_size: Tuple[int, int],
        grayscale: bool = False,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Load entire dataset from directory structure.
        
        Returns:
            DataFrame with columns: [filepath, label, class_name]
        """
        logger.info("Starting dataset loading...")
        
        filepaths = []
        labels = []
        class_names_list = []
        
        # Dynamically get class names from dataset directory
        if not DATA_DIR.exists():
            raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")
        
        classes = sorted([d for d in os.listdir(DATA_DIR) 
                         if os.path.isdir(DATA_DIR / d)])
        
        logger.info(f"Found classes: {classes}")
        
        for class_idx, class_name in enumerate(classes):
            class_path = DATA_DIR / class_name
            image_count = 0
            
            for filename in os.listdir(class_path):
                filepath = str(class_path / filename)
                
                image = self.load_image(
                    filepath,
                    target_size,
                    grayscale=grayscale,
                    normalize=normalize
                )
                
                if image is not None:
                    filepaths.append(filepath)
                    labels.append(class_idx)
                    class_names_list.append(class_name)
                    image_count += 1
                elif not SKIP_CORRUPTED_IMAGES:
                    logger.error(f"Failed to load: {filepath}")
                    raise RuntimeError(f"Cannot skip corrupted image: {filepath}")
            
            logger.info(f"Loaded {image_count} images from class '{class_name}'")
        
        # Log corrupted images
        if self.corrupted_images and LOG_CORRUPTED_IMAGES:
            logger.warning(f"Found {len(self.corrupted_images)} corrupted images")
            for img_path in self.corrupted_images[:10]:  # Log first 10
                logger.warning(f"  - {img_path}")
        
        df = pd.DataFrame({
            'filepath': filepaths,
            'label': labels,
            'class_name': class_names_list
        })
        
        logger.info(f"Total images loaded: {len(df)}")
        logger.info(f"Dataset distribution:\n{df['class_name'].value_counts()}")
        
        return df


class DataPreprocessor:
    """Preprocess and split data for training."""
    
    @staticmethod
    def get_class_distribution(df: pd.DataFrame) -> Dict[str, int]:
        """Get count of images per class."""
        return df['class_name'].value_counts().to_dict()
    
    @staticmethod
    def validate_class_distribution(df: pd.DataFrame, expected: Dict[str, int]) -> bool:
        """Validate dataset distribution matches expected."""
        actual = DataPreprocessor.get_class_distribution(df)
        
        logger.info("Dataset Distribution Validation:")
        all_valid = True
        for class_name, expected_count in expected.items():
            actual_count = actual.get(class_name, 0)
            tolerance = 0.95  # Allow 5% deviation
            is_valid = actual_count >= expected_count * tolerance
            
            status = "✓" if is_valid else "✗"
            logger.info(f"  {status} {class_name}: {actual_count}/{expected_count}")
            
            if not is_valid:
                all_valid = False
        
        return all_valid
    
    @staticmethod
    def train_test_split_data(
        df: pd.DataFrame,
        test_size: float = TRAIN_TEST_SPLIT,
        random_state: int = RANDOM_STATE,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        
        Args:
            df (pd.DataFrame): Data frame with image data
            test_size (float): Fraction for test set
            random_state (int): Random seed
            stratify (bool): Stratify by class label
        
        Returns:
            Tuple[DataFrame, DataFrame]: Train and test dataframes
        """
        logger.info(f"Splitting data: train={1-test_size:.1%}, test={test_size:.1%}")
        
        stratify_col = df['label'] if stratify else None
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )
        
        logger.info(f"Training samples: {len(train_df)}")
        logger.info(f"Testing samples: {len(test_df)}")
        
        return train_df, test_df


def get_dataset_summary() -> Dict:
    """
    Get summary statistics of dataset without loading all images.
    
    Returns:
        Dictionary with dataset information
    """
    summary = {
        'total_images': sum(EXPECTED_CLASS_DISTRIBUTION.values()),
        'num_classes': len(CLASS_NAMES),
        'classes': CLASS_NAMES,
        'distribution': EXPECTED_CLASS_DISTRIBUTION,
        'data_dir': str(DATA_DIR),
    }
    
    logger.info(f"Dataset Summary: {summary['total_images']} images, "
                f"{summary['num_classes']} classes")
    
    return summary


def create_data_generators(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_size: Tuple[int, int],
    batch_size: int = 32,
    grayscale: bool = False,
    augmentation_config: Dict = None
):
    """
    Create Keras data generators for training.
    
    Args:
        train_df: Training dataframe
        test_df: Testing dataframe
        image_size: Target image size
        batch_size: Batch size
        grayscale: Use grayscale images
        augmentation_config: Data augmentation configuration
    
    Returns:
        Tuple of (train_generator, test_generator)
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    logger.info(f"Creating data generators with batch_size={batch_size}")
    
    # Default augmentation config
    if augmentation_config is None:
        from config import DATA_AUGMENTATION_CONFIG
        augmentation_config = DATA_AUGMENTATION_CONFIG
    
    # Training generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255.,
        **augmentation_config
    )
    
    # Test generator (no augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255.)
    
    color_mode = 'grayscale' if grayscale else 'rgb'
    
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filepath',
        y_col='class_name',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode=color_mode,
        shuffle=True
    )
    
    test_generator = test_datagen.flow_from_dataframe(
        test_df,
        x_col='filepath',
        y_col='class_name',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode=color_mode,
        shuffle=False
    )
    
    logger.info(f"Data generators created successfully")
    
    return train_generator, test_generator


# Quick reference functions
def get_corrupted_images_report(loader: ImageLoader) -> List[str]:
    """Get list of corrupted image paths."""
    return loader.corrupted_images
