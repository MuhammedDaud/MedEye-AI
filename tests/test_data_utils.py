"""
Unit Tests for MedEye Data Utilities
Tests data loading, preprocessing, and validation
"""

import pytest
import sys
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image as PILImage

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_utils import ImageValidator, ImageLoader, DataPreprocessor, get_dataset_summary


class TestImageValidator:
    """Test ImageValidator class."""
    
    def test_valid_extension_jpg(self):
        """Test jpg extension validation."""
        assert ImageValidator.is_valid_extension("image.jpg") is True
    
    def test_valid_extension_jpeg(self):
        """Test jpeg extension validation."""
        assert ImageValidator.is_valid_extension("image.jpeg") is True
    
    def test_valid_extension_png(self):
        """Test png extension validation."""
        assert ImageValidator.is_valid_extension("image.png") is True
    
    def test_invalid_extension(self):
        """Test invalid extension rejection."""
        assert ImageValidator.is_valid_extension("image.txt") is False
        assert ImageValidator.is_valid_extension("image.pdf") is False
    
    def test_valid_file_size(self):
        """Test file size validation."""
        # Create temporary small file
        with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
            # Small file should be valid
            assert ImageValidator.is_valid_file_size(tmp.name) is True
    
    def test_valid_image_dimensions(self):
        """Test image dimension validation."""
        # Create a valid size image (50x50 minimum)
        img = np.ones((100, 100, 3), dtype=np.uint8)
        assert ImageValidator.is_valid_image_dimensions(img) is True
    
    def test_invalid_image_dimensions_too_small(self):
        """Test rejection of too-small images."""
        # Create a small image (less than 50x50)
        img = np.ones((30, 30, 3), dtype=np.uint8)
        assert ImageValidator.is_valid_image_dimensions(img) is False
    
    def test_valid_image_dimensions_grayscale(self):
        """Test dimension validation for grayscale images."""
        # Grayscale image has 2 dimensions
        img = np.ones((100, 100), dtype=np.uint8)
        assert ImageValidator.is_valid_image_dimensions(img) is True


class TestImageLoader:
    """Test ImageLoader class."""
    
    def test_loader_initialization(self):
        """Test ImageLoader initialization."""
        loader = ImageLoader()
        assert loader.corrupted_images == []
        assert loader.validator is not None
    
    def test_corrupted_images_tracking(self):
        """Test tracking of corrupted images."""
        loader = ImageLoader()
        # Try to load non-existent file
        result = loader.load_image(
            "/nonexistent/path/image.jpg",
            target_size=(224, 224)
        )
        assert result is None
        # Corrupted images might be tracked (depends on validation)
    
    def test_load_image_returns_numpy_array(self):
        """Test that load_image returns proper numpy array."""
        loader = ImageLoader()
        # Create a temporary test image
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image
            test_img_path = Path(tmpdir) / "test.jpg"
            img = PILImage.new('RGB', (256, 256), color='red')
            img.save(str(test_img_path))
            
            # Load it
            loaded_img = loader.load_image(
                str(test_img_path),
                target_size=(224, 224),
                normalize=True
            )
            
            assert loaded_img is not None
            assert isinstance(loaded_img, np.ndarray)
            assert loaded_img.shape == (224, 224, 3)


class TestDataPreprocessor:
    """Test DataPreprocessor class."""
    
    def test_get_class_distribution(self):
        """Test class distribution calculation."""
        import pandas as pd
        
        # Create sample dataframe
        df = pd.DataFrame({
            'class_name': ['cataract', 'glaucoma', 'cataract', 'normal', 'glaucoma', 'normal', 'normal']
        })
        
        dist = DataPreprocessor.get_class_distribution(df)
        
        assert dist['cataract'] == 2
        assert dist['glaucoma'] == 2
        assert dist['normal'] == 3
    
    def test_train_test_split(self):
        """Test train-test split."""
        import pandas as pd
        
        # Create sample dataframe
        df = pd.DataFrame({
            'filepath': [f'img_{i}.jpg' for i in range(100)],
            'label': [i % 4 for i in range(100)],
            'class_name': ['class_' + str(i % 4) for i in range(100)]
        })
        
        train_df, test_df = DataPreprocessor.train_test_split_data(
            df,
            test_size=0.2,
            stratify=True
        )
        
        # Check split
        assert len(train_df) == 80
        assert len(test_df) == 20
        assert len(train_df) + len(test_df) == len(df)
    
    def test_train_test_split_stratification(self):
        """Test that train-test split maintains class balance."""
        import pandas as pd
        
        # Create balanced dataframe
        df = pd.DataFrame({
            'filepath': [f'img_{i}.jpg' for i in range(100)],
            'label': [i % 4 for i in range(100)],
            'class_name': ['class_' + str(i % 4) for i in range(100)]
        })
        
        train_df, test_df = DataPreprocessor.train_test_split_data(
            df,
            test_size=0.2,
            stratify=True
        )
        
        # Check class distribution is maintained
        train_dist = train_df['label'].value_counts().sort_index()
        test_dist = test_df['label'].value_counts().sort_index()
        
        # Each class should be represented in both splits
        for label in range(4):
            assert label in train_dist.index
            assert label in test_dist.index


class TestDatasetSummary:
    """Test dataset summary functions."""
    
    def test_get_dataset_summary(self):
        """Test getting dataset summary."""
        summary = get_dataset_summary()
        
        assert 'total_images' in summary
        assert 'num_classes' in summary
        assert 'classes' in summary
        assert 'distribution' in summary
        
        assert summary['num_classes'] == 4
        assert summary['total_images'] == 4217


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
