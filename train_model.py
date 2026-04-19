"""
╔════════════════════════════════════════════════════════════════════════════╗
║                            MEDEYE v2.0                                      ║
║                   Medical Eye Disease Detection System                       ║
║                                                                              ║
║  Developer: Muhammad Daud                                                    ║
║  Training Script - Model training with full error handling and logging      ║
║                                                                              ║
║  Usage:
║      python train_model.py --model efficientnetb3 --epochs 10
║      python train_model.py --model mobilenet --batch-size 32 --learning-rate 0.0005
║      python train_model.py --baseline --epochs 20
╚════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import sys
from pathlib import Path
import tensorflow as tf

# Import utilities
from config import (
    TRANSFER_LEARNING_IMG_SIZE, CLASS_NAMES, DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE, DATA_AUGMENTATION_CONFIG, get_data_dir,
    verify_config
)
from logging_setup import get_logger, training_logger, error_logger
from data_utils import ImageLoader, DataPreprocessor, create_data_generators
from model_utils import (
    ModelBuilder, ModelCompiler, ModelTrainer, ModelEvaluator,
    ModelSaver
)

logger = get_logger(__name__)


def train_transfer_learning_model(
    model_name: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = 0.001
) -> dict:
    """
    Train a transfer learning model with full error handling.
    
    Args:
        model_name (str): Name of model to train
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
    
    Returns:
        dict: Training results and metrics
    """
    logger.info(f"Starting training: {model_name}")
    logger.info(f"Configuration: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    
    try:
        # Step 1: Verify configuration
        logger.info("Verifying configuration...")
        verify_config()
        
        # Step 2: Load dataset
        logger.info("Loading dataset...")
        image_loader = ImageLoader()
        df = image_loader.load_dataset(
            target_size=TRANSFER_LEARNING_IMG_SIZE,
            grayscale=False,
            normalize=True
        )
        
        # Log corrupted images
        corrupted = image_loader.corrupted_images
        if corrupted:
            logger.warning(f"Skipped {len(corrupted)} corrupted images")
            if len(corrupted) <= 10:
                for img_path in corrupted:
                    logger.warning(f"  - {img_path}")
        
        # Step 3: Validate dataset distribution
        logger.info("Validating dataset distribution...")
        from config import EXPECTED_CLASS_DISTRIBUTION
        DataPreprocessor.validate_class_distribution(df, EXPECTED_CLASS_DISTRIBUTION)
        
        # Step 4: Split data
        logger.info("Splitting dataset...")
        train_df, test_df = DataPreprocessor.train_test_split_data(
            df,
            test_size=0.2,
            stratify=True
        )
        
        # Step 5: Create data generators
        logger.info("Creating data generators...")
        train_gen, test_gen = create_data_generators(
            train_df,
            test_df,
            image_size=TRANSFER_LEARNING_IMG_SIZE,
            batch_size=batch_size,
            grayscale=False,
            augmentation_config=DATA_AUGMENTATION_CONFIG
        )
        
        # Step 6: Build model
        logger.info(f"Building model: {model_name}")
        model = ModelBuilder.build_transfer_learning_model(
            model_name,
            num_classes=len(CLASS_NAMES),
            input_shape=(*TRANSFER_LEARNING_IMG_SIZE, 3),
            freeze_base=True,
            dropout_rate=0.5
        )
        
        # Step 7: Compile model
        logger.info("Compiling model...")
        model = ModelCompiler.compile_model(
            model,
            learning_rate=learning_rate,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        logger.info(f"Model has {model.count_params():,} parameters")
        
        # Step 8: Train model
        logger.info(f"Training model for {epochs} epochs...")
        training_logger.info(f"Training {model_name}")
        history = ModelTrainer.train_model(
            model,
            train_gen,
            test_gen,
            epochs=epochs,
            model_name=model_name,
            verbose=1
        )
        
        # Step 9: Evaluate model
        logger.info("Evaluating model...")
        metrics = ModelEvaluator.evaluate_model(
            model,
            test_gen,
            model_name=model_name
        )
        
        # Step 10: Get predictions
        logger.info("Generating predictions...")
        predictions, true_labels, class_names = ModelEvaluator.get_predictions(
            model,
            test_gen,
            class_names=CLASS_NAMES
        )
        
        # Step 11: Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        
        # Step 12: Save model
        logger.info("Saving model...")
        model_path = ModelSaver.save_model(model, model_name, format="h5")
        
        # Compile results
        results = {
            'model_name': model_name,
            'accuracy': metrics['accuracy'],
            'loss': metrics['loss'],
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'epochs_trained': len(history.history['loss']),
            'model_path': model_path,
            'training_samples': len(train_df),
            'test_samples': len(test_df)
        }
        
        logger.info("Training completed successfully!")
        logger.info(f"Results: {results}")
        
        return results
        
    except FileNotFoundError as e:
        error_logger.error(f"File not found: {str(e)}")
        logger.error(f"Cannot find required file: {str(e)}")
        raise
    except ValueError as e:
        error_logger.error(f"Configuration error: {str(e)}")
        logger.error(f"Configuration error: {str(e)}")
        raise
    except Exception as e:
        error_logger.error(f"Unexpected error during training: {str(e)}", exc_info=True)
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise


def train_baseline_cnn(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> dict:
    """Train baseline CNN model."""
    logger.info("Starting baseline CNN training")
    
    try:
        # Dataset loading
        logger.info("Loading dataset...")
        image_loader = ImageLoader()
        df = image_loader.load_dataset(
            target_size=TRANSFER_LEARNING_IMG_SIZE,
            grayscale=False,
            normalize=True
        )
        
        # Split data
        train_df, test_df = DataPreprocessor.train_test_split_data(df)
        
        # Create generators
        train_gen, test_gen = create_data_generators(
            train_df,
            test_df,
            image_size=TRANSFER_LEARNING_IMG_SIZE,
            batch_size=batch_size
        )
        
        # Build baseline model
        logger.info("Building baseline CNN...")
        model = ModelBuilder.build_baseline_cnn(
            input_shape=(*TRANSFER_LEARNING_IMG_SIZE, 3),
            num_classes=len(CLASS_NAMES)
        )
        
        # Compile
        model = ModelCompiler.compile_model(model)
        
        # Train
        logger.info("Training baseline CNN...")
        history = ModelTrainer.train_model(
            model,
            train_gen,
            test_gen,
            epochs=epochs,
            model_name="baseline_cnn"
        )
        
        # Evaluate
        metrics = ModelEvaluator.evaluate_model(model, test_gen, "baseline_cnn")
        
        # Save
        model_path = ModelSaver.save_model(model, "baseline_cnn")
        
        results = {
            'model_name': 'baseline_cnn',
            'accuracy': metrics['accuracy'],
            'loss': metrics['loss'],
            'model_path': model_path
        }
        
        logger.info(f"Baseline CNN training completed: {results}")
        return results
        
    except Exception as e:
        error_logger.error(f"Error training baseline CNN: {str(e)}")
        raise


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description="Train MedEye models"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='efficientnetb3',
        help='Model to train: efficientnetb3, mobilenet, densenet121, etc.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=DEFAULT_EPOCHS,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate for optimizer'
    )
    parser.add_argument(
        '--baseline',
        action='store_true',
        help='Train baseline CNN instead'
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("MedEye Model Training")
    logger.info("="*60)
    
    try:
        if args.baseline:
            results = train_baseline_cnn(
                epochs=args.epochs,
                batch_size=args.batch_size
            )
        else:
            results = train_transfer_learning_model(
                model_name=args.model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
        
        # Print final results
        logger.info("="*60)
        logger.info("TRAINING RESULTS")
        logger.info("="*60)
        for key, value in results.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
