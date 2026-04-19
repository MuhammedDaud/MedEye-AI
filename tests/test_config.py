"""
Unit Tests for MedEye Configuration
Tests configuration validation and defaults
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from config import (
    CLASS_NAMES, NUM_CLASSES, DATA_DIR, TRAIN_TEST_SPLIT,
    DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, get_class_index, verify_config
)


class TestConfiguration:
    """Test configuration module."""
    
    def test_class_names_defined(self):
        """Test that class names are defined."""
        assert CLASS_NAMES is not None
        assert len(CLASS_NAMES) > 0
    
    def test_num_classes_matches_class_names(self):
        """Test that NUM_CLASSES matches CLASS_NAMES length."""
        assert NUM_CLASSES == len(CLASS_NAMES)
    
    def test_class_names_not_empty(self):
        """Test that class names are not empty strings."""
        for class_name in CLASS_NAMES:
            assert isinstance(class_name, str)
            assert len(class_name) > 0
    
    def test_expected_classes(self):
        """Test expected classes exist."""
        expected_classes = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
        assert sorted(CLASS_NAMES) == sorted(expected_classes)
    
    def test_train_test_split_valid(self):
        """Test train-test split is valid."""
        assert 0 < TRAIN_TEST_SPLIT < 1
    
    def test_batch_size_positive(self):
        """Test batch size is positive."""
        assert DEFAULT_BATCH_SIZE > 0
    
    def test_epochs_positive(self):
        """Test epochs is positive."""
        assert DEFAULT_EPOCHS > 0
    
    def test_get_class_index_valid(self):
        """Test get_class_index returns correct index."""
        for idx, class_name in enumerate(CLASS_NAMES):
            assert get_class_index(class_name) == idx
    
    def test_get_class_index_invalid(self):
        """Test get_class_index raises error for invalid class."""
        with pytest.raises(ValueError):
            get_class_index("invalid_class")
    
    def test_verify_config(self):
        """Test configuration verification."""
        # Should not raise
        assert config.verify_config() is True
    
    def test_data_dir_exists(self):
        """Test that data directory is properly configured."""
        assert isinstance(DATA_DIR, Path)
        # Directory should be defined (may not exist in test environment)
        assert str(DATA_DIR).endswith("dataset")


class TestConfigurationPaths:
    """Test that all paths are properly defined."""
    
    def test_project_root_defined(self):
        """Test project root is defined."""
        assert config.PROJECT_ROOT is not None
        assert isinstance(config.PROJECT_ROOT, Path)
    
    def test_models_dir_defined(self):
        """Test models directory is defined."""
        assert config.MODELS_DIR is not None
        assert isinstance(config.MODELS_DIR, Path)
    
    def test_logs_dir_defined(self):
        """Test logs directory is defined."""
        assert config.LOGS_DIR is not None
        assert isinstance(config.LOGS_DIR, Path)
    
    def test_results_dir_defined(self):
        """Test results directory is defined."""
        assert config.RESULTS_DIR is not None
        assert isinstance(config.RESULTS_DIR, Path)


class TestImageConfiguration:
    """Test image processing configuration."""
    
    def test_transfer_learning_image_size(self):
        """Test transfer learning image size."""
        assert config.TRANSFER_LEARNING_IMG_SIZE == (224, 224)
    
    def test_traditional_ml_image_size(self):
        """Test traditional ML image size."""
        assert config.TRADITIONAL_ML_IMG_SIZE == (128, 128)
    
    def test_valid_image_extensions(self):
        """Test valid image extensions are defined."""
        assert len(config.VALID_IMAGE_EXTENSIONS) > 0
        assert "jpg" in config.VALID_IMAGE_EXTENSIONS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
