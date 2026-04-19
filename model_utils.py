"""
╔════════════════════════════════════════════════════════════════════════════╗
║                            MEDEYE v2.0                                      ║
║                   Medical Eye Disease Detection System                       ║
║                                                                              ║
║  Developer: Muhammad Daud                                                    ║
║  Model Utilities Module - Model building, training, evaluation              ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import os
import numpy as np
from typing import Dict, Tuple, Optional, List
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import logging

from config import (
    DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE,
    DEFAULT_LOSS, NUM_CLASSES, MODELS_DIR, RESULTS_DIR
)
from logging_setup import get_logger

logger = get_logger(__name__)


class ModelBuilder:
    """Build transfer learning and custom models."""
    
    @staticmethod
    def build_transfer_learning_model(
        base_model_name: str,
        num_classes: int = NUM_CLASSES,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        freeze_base: bool = True,
        dropout_rate: float = 0.5
    ) -> tf.keras.Model:
        """
        Build transfer learning model with pre-trained base.
        
        Args:
            base_model_name (str): Name of base model (efficientnetb3, mobilenet, etc.)
            num_classes (int): Number of output classes
            input_shape (tuple): Input shape (height, width, channels)
            freeze_base (bool): Freeze base model weights
            dropout_rate (float): Dropout rate for regularization
        
        Returns:
            tf.keras.Model: Compiled model
        """
        logger.info(f"Building transfer learning model: {base_model_name}")
        
        try:
            # Load base model
            if base_model_name.lower() == "efficientnetb3":
                base = tf.keras.applications.EfficientNetB3(
                    input_shape=input_shape,
                    weights='imagenet',
                    include_top=False
                )
            elif base_model_name.lower() == "mobilenet":
                base = tf.keras.applications.MobileNetV2(
                    input_shape=input_shape,
                    weights='imagenet',
                    include_top=False
                )
            elif base_model_name.lower() == "densenet121":
                base = tf.keras.applications.DenseNet121(
                    input_shape=input_shape,
                    weights='imagenet',
                    include_top=False
                )
            elif base_model_name.lower() == "resnet50":
                base = tf.keras.applications.ResNet50(
                    input_shape=input_shape,
                    weights='imagenet',
                    include_top=False
                )
            elif base_model_name.lower() == "vgg16":
                base = tf.keras.applications.VGG16(
                    input_shape=input_shape,
                    weights='imagenet',
                    include_top=False
                )
            elif base_model_name.lower() == "xception":
                base = tf.keras.applications.Xception(
                    input_shape=input_shape,
                    weights='imagenet',
                    include_top=False
                )
            elif base_model_name.lower() == "inceptionv3":
                base = tf.keras.applications.InceptionV3(
                    input_shape=input_shape,
                    weights='imagenet',
                    include_top=False
                )
            else:
                raise ValueError(f"Unknown model: {base_model_name}")
            
            # Freeze base model
            if freeze_base:
                base.trainable = False
                logger.info("Base model weights frozen")
            
            # Build full model
            inputs = tf.keras.Input(shape=input_shape)
            x = base(inputs, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            logger.info(f"Model built successfully with {model.count_params()} parameters")
            
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
    
    @staticmethod
    def build_baseline_cnn(
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = NUM_CLASSES
    ) -> tf.keras.Model:
        """
        Build baseline CNN from scratch.
        
        Args:
            input_shape (tuple): Input shape
            num_classes (int): Number of output classes
        
        Returns:
            tf.keras.Model: Compiled baseline model
        """
        logger.info("Building baseline CNN model")
        
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        logger.info(f"Baseline CNN built with {model.count_params()} parameters")
        return model


class ModelCompiler:
    """Compile models with appropriate optimizers and loss functions."""
    
    @staticmethod
    def compile_model(
        model: tf.keras.Model,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        loss: str = DEFAULT_LOSS,
        metrics: List[str] = None
    ) -> tf.keras.Model:
        """
        Compile a model.
        
        Args:
            model (tf.keras.Model): Model to compile
            learning_rate (float): Learning rate for optimizer
            loss (str): Loss function name
            metrics (list): List of metrics
        
        Returns:
            tf.keras.Model: Compiled model
        """
        if metrics is None:
            metrics = ['accuracy']
        
        logger.info(f"Compiling model with lr={learning_rate}, loss={loss}")
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        return model


class ModelTrainer:
    """Train models with callbacks and monitoring."""
    
    @staticmethod
    def get_callbacks(
        model_name: str,
        patience: int = 5,
        reduce_lr_patience: int = 3
    ) -> List:
        """
        Create training callbacks.
        
        Args:
            model_name (str): Name for checkpoint saving
            patience (int): Early stopping patience
            reduce_lr_patience (int): Learning rate reduction patience
        
        Returns:
            List of callbacks
        """
        checkpoint_path = MODELS_DIR / f"{model_name}_best.h5"
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                str(checkpoint_path),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        logger.info(f"Callbacks created: checkpoint at {checkpoint_path}")
        return callbacks
    
    @staticmethod
    def train_model(
        model: tf.keras.Model,
        train_generator,
        validation_generator,
        epochs: int = DEFAULT_EPOCHS,
        model_name: str = "model",
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """
        Train a model with data generators.
        
        Args:
            model: Keras model
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs: Number of training epochs
            model_name: Name for logging
            verbose: Verbosity level
        
        Returns:
            Training history
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        callbacks = ModelTrainer.get_callbacks(model_name)
        
        try:
            history = model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=epochs,
                callbacks=callbacks,
                verbose=verbose
            )
            
            logger.info("Training completed successfully")
            return history
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise


class ModelEvaluator:
    """Evaluate model performance."""
    
    @staticmethod
    def evaluate_model(
        model: tf.keras.Model,
        test_generator,
        model_name: str = "model"
    ) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            model: Trained Keras model
            test_generator: Test data generator
            model_name: Name for logging
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating model: {model_name}")
        
        try:
            loss, accuracy = model.evaluate(test_generator)
            
            metrics = {
                'loss': float(loss),
                'accuracy': float(accuracy),
                'model_name': model_name
            }
            
            logger.info(f"Test Accuracy: {accuracy:.4f}, Test Loss: {loss:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            raise
    
    @staticmethod
    def get_predictions(
        model: tf.keras.Model,
        test_generator,
        class_names: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions and true labels.
        
        Args:
            model: Trained model
            test_generator: Test data generator
            class_names: List of class names
        
        Returns:
            Tuple of (predictions, true_labels, class_names)
        """
        logger.info("Generating predictions on test set")
        
        predictions = model.predict(test_generator)
        true_labels = test_generator.classes
        
        if class_names is None:
            class_names = list(test_generator.class_indices.keys())
        
        predicted_classes = np.argmax(predictions, axis=1)
        
        logger.info(f"Generated {len(predictions)} predictions")
        
        return predicted_classes, true_labels, class_names


class ModelSaver:
    """Save and load trained models."""
    
    @staticmethod
    def save_model(
        model: tf.keras.Model,
        model_name: str,
        format: str = "h5"
    ) -> str:
        """
        Save trained model.
        
        Args:
            model: Trained model
            model_name: Name for the model
            format: Save format (h5 or keras)
        
        Returns:
            Path to saved model
        """
        filepath = MODELS_DIR / f"{model_name}.{format}"
        
        logger.info(f"Saving model to {filepath}")
        
        try:
            model.save(str(filepath))
            logger.info(f"Model saved successfully: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    @staticmethod
    def load_model(model_path: str) -> tf.keras.Model:
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to model file
        
        Returns:
            Loaded model
        """
        logger.info(f"Loading model from {model_path}")
        
        try:
            model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


def create_model_summary(model: tf.keras.Model) -> str:
    """Get model architecture summary as string."""
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return "\n".join(summary_lines)
