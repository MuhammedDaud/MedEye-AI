# MedEye v2.0: Medical Eye Disease Detection System 👁️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10+](https://img.shields.io/badge/tensorflow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![Status: Production Ready](https://img.shields.io/badge/status-production--ready-green.svg)](#)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Supported Models](#supported-models)
- [Dataset Information](#dataset-information)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## 🎯 Overview

**MedEye** is a comprehensive, production-ready framework for automated detection and classification of eye diseases using deep learning and machine learning. The system analyzes medical retinal images to identify multiple ocular conditions with high accuracy and reliability.

### Key Information

- **Developer:** Muhammad Daud
- **Version:** 2.0
- **License:** MIT Open Source
- **Status:** Production Ready
- **Python Version:** 3.8+
- **Framework:** TensorFlow/Keras 2.10+

### Project Goals

✅ Accurate detection of multiple eye diseases  
✅ High-performance transfer learning models  
✅ Easy-to-use API and comprehensive documentation  
✅ Production-ready error handling and logging  
✅ Extensible architecture for new models  
✅ Comprehensive testing and validation  

---

## ✨ Key Features

### 🧠 Advanced Machine Learning

- **Multiple Transfer Learning Models** with pre-trained weights
- **Traditional ML Support** for comparison and ensemble methods
- **Data Augmentation** for improved model generalization
- **Mixed Precision Training** for faster computation on modern GPUs

### 🔍 Robust Data Processing

- **Comprehensive Image Validation** (resolution, file integrity, format)
- **Automatic Corrupted Image Detection** with detailed logging
- **Stratified Data Splitting** to maintain class balance
- **Flexible Preprocessing** pipeline with grayscale/RGB support

### 📊 Production Features

- **Centralized Configuration** for easy parameter management
- **Structured Logging** with console and file outputs
- **Error Handling** with detailed context and recovery mechanisms
- **Model Serialization** in both HDF5 and SavedModel formats
- **Comprehensive Metrics** including precision, recall, and F1-score

### 🧪 Testing and Validation

- **Unit Tests** for critical components
- **Configuration Validation** on module import
- **Dataset Distribution Validation** before training
- **Metrics Calculation** with sklearn integration

---

## 🧠 Supported Models

### Transfer Learning Models (Recommended)

| Model | Input Size | Parameters | Speed | Accuracy | Use Case |
|-------|-----------|-----------|-------|----------|----------|
| **EfficientNetB3** | 300×300 | 10.7M | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Best Overall |
| **MobileNetV2** | 224×224 | 3.5M | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Mobile/Edge Devices |
| **DenseNet121** | 224×224 | 7.9M | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | High Accuracy |
| **ResNet50** | 224×224 | 25.6M | ⭐⭐⭐ | ⭐⭐⭐⭐ | Standard Baseline |
| **VGG16** | 224×224 | 138M | ⭐⭐ | ⭐⭐⭐⭐ | Feature Extraction |
| **Xception** | 299×299 | 22.9M | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Advanced Analysis |
| **InceptionV3** | 299×299 | 27.2M | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Multi-scale Features |

### Traditional ML Models

- **Support Vector Machine (SVM)** - Linear and non-linear classification
- **Random Forest** - Ensemble-based approach with feature importance
- **Baseline CNN** - Custom CNN built from scratch

### Model Selection Guide

```
Use EfficientNetB3 if:
  → You want best overall performance
  → Balanced speed and accuracy
  → Production deployment required

Use MobileNetV2 if:
  → Mobile or edge device deployment
  → Limited computational resources
  → Real-time inference needed

Use DenseNet121 if:
  → Maximum accuracy is priority
  → Adequate computational resources
  → Dense feature representations needed

Use Traditional ML Models if:
  → Comparison baseline required
  → Interpretability is important
  → Resource constraints
```

---

## 📊 Dataset Information

### Dataset Overview

The project uses a curated collection of **4,217 retinal images** sourced from multiple clinical datasets including IDRiD, Ocular Recognition, and HRF.

### Class Distribution

| Disease | Count | Percentage |
|---------|-------|-----------|
| **Diabetic Retinopathy** | 1,098 | 26.0% |
| **Cataract** | 1,038 | 24.6% |
| **Normal** | 1,074 | 25.5% |
| **Glaucoma** | 1,007 | 23.9% |
| **TOTAL** | **4,217** | **100%** |

### Disease Descriptions

**Cataract**  
Clouding of the natural eye lens, causing vision deterioration. Progressive condition often associated with aging.

**Diabetic Retinopathy**  
Vascular damage to the retina caused by diabetes. Can lead to vision loss if untreated.

**Glaucoma**  
Elevated intraocular pressure damaging the optic nerve. Often called the "silent thief of sight."

**Normal**  
Healthy retinal tissue showing no signs of disease or pathology.

### Dataset Structure

```
dataset/
├── cataract/           (1,038 images)
├── diabetic_retinopathy/ (1,098 images)
├── glaucoma/           (1,007 images)
└── normal/             (1,074 images)
```

---

## 🚀 Installation

### System Requirements

- **OS:** Windows, macOS, or Linux
- **Python:** 3.8 or higher
- **RAM:** Minimum 8GB (16GB+ recommended)
- **GPU:** CUDA 11.0+ compatible GPU (optional but recommended)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/medeye.git
cd medeye
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv medeye_env
source medeye_env/bin/activate  # On Windows: medeye_env\Scripts\activate

# Or using conda
conda create -n medeye python=3.9
conda activate medeye
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow; import cv2; import sklearn; print('✓ All dependencies installed successfully')"
```

### GPU Setup (Optional)

For GPU acceleration with TensorFlow:

```bash
# Install CUDA and cuDNN (follow TensorFlow GPU guide)
pip install tensorflow[and-cuda]  # TensorFlow 2.13+

# Or install GPU-specific packages
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.6
```

---

## ⚡ Quick Start

### 1. Basic Model Training

```bash
# Train EfficientNetB3 (recommended)
python train_model.py --model efficientnetb3 --epochs 20 --batch-size 32

# Train MobileNet with custom parameters
python train_model.py --model mobilenet --epochs 15 --learning-rate 0.0005

# Train baseline CNN
python train_model.py --baseline --epochs 25
```

### 2. Using the API in Python

```python
from config import CLASS_NAMES, TRANSFER_LEARNING_IMG_SIZE
from data_utils import ImageLoader, DataPreprocessor, create_data_generators
from model_utils import ModelBuilder, ModelCompiler, ModelTrainer, ModelEvaluator

# Load dataset
loader = ImageLoader()
df = loader.load_dataset(
    target_size=TRANSFER_LEARNING_IMG_SIZE,
    grayscale=False,
    normalize=True
)

# Split data
train_df, test_df = DataPreprocessor.train_test_split_data(df)

# Create generators
train_gen, test_gen = create_data_generators(
    train_df, test_df,
    image_size=TRANSFER_LEARNING_IMG_SIZE,
    batch_size=32
)

# Build and train model
model = ModelBuilder.build_transfer_learning_model(
    'efficientnetb3',
    num_classes=len(CLASS_NAMES)
)
model = ModelCompiler.compile_model(model)
history = ModelTrainer.train_model(model, train_gen, test_gen, epochs=20)

# Evaluate
metrics = ModelEvaluator.evaluate_model(model, test_gen)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

---

## 📁 Project Structure

```
medeye/
├── README.md                      # This file
├── LICENSE                        # MIT License with full details
├── CODE_OF_CONDUCT.md            # Community guidelines
├── requirements.txt              # Python dependencies
│
├── config.py                     # Centralized configuration
├── logging_setup.py              # Logging configuration
├── data_utils.py                 # Data loading and preprocessing
├── model_utils.py                # Model building and training
├── train_model.py                # Training script with CLI
│
├── dataset/                      # Training dataset
│   ├── cataract/
│   ├── diabetic_retinopathy/
│   ├── glaucoma/
│   └── normal/
│
├── models/                       # Trained model directory (auto-created)
├── logs/                         # Training logs directory (auto-created)
├── results/                      # Results directory (auto-created)
│
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_data_utils.py
│   └── test_model_utils.py
│
└── [Model Folders]/              # Individual model implementations
    ├── Baseline CNN Model/
    ├── DenseNet121 Model/
    ├── EfficientNET Model/
    ├── InceptionV3 - Improved Baseline/
    ├── MobileNet Model/
    ├── Random Forest Model/
    ├── ResNet50 Model/
    ├── SVM Model/
    ├── VGG16 Model/
    └── Xception Model/
```

---

## 🎓 Model Training

### Training Configuration

Edit `config.py` to modify training parameters:

```python
# Training parameters
DEFAULT_BATCH_SIZE = 64          # Increase for better GPU utilization
DEFAULT_EPOCHS = 10              # Set training epochs
DEFAULT_LEARNING_RATE = 0.001    # Adjust learning rate

# Data augmentation
DATA_AUGMENTATION_CONFIG = {
    "rotation_range": 20,
    "width_shift_range": 0.15,
    "height_shift_range": 0.15,
    "shear_range": 0.2,
    "zoom_range": 0.2,
    "horizontal_flip": False,
    "vertical_flip": True,
    "fill_mode": "nearest",
}
```

### Training Scripts

#### Command Line Interface

```bash
# Full help
python train_model.py --help

# Examples
python train_model.py --model efficientnetb3 --epochs 30 --batch-size 16 --learning-rate 0.0001

python train_model.py --model densenet121 --epochs 25

python train_model.py --baseline --epochs 20
```

#### Programmatic API

```python
from train_model import train_transfer_learning_model, train_baseline_cnn

# Train transfer learning model
results = train_transfer_learning_model(
    model_name='efficientnetb3',
    epochs=20,
    batch_size=32,
    learning_rate=0.001
)

# Train baseline
results = train_baseline_cnn(epochs=25, batch_size=32)

print(f"Accuracy: {results['accuracy']}")
print(f"F1-Score: {results['f1_score']}")
print(f"Model saved at: {results['model_path']}")
```

### Early Stopping and Learning Rate Reduction

The training automatically implements:

- **Early Stopping:** Prevents overfitting by stopping when validation loss plateaus
- **Learning Rate Reduction:** Reduces learning rate when progress stalls
- **Model Checkpointing:** Saves best model based on validation accuracy

---

## 📊 Model Evaluation

### Evaluation Metrics

Comprehensive evaluation includes:

- **Accuracy:** Overall correctness of predictions
- **Precision:** True positive rate among positive predictions
- **Recall:** True positive rate among actual positives
- **F1-Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Detailed classification breakdown
- **Classification Report:** Per-class performance metrics

### Evaluation Example

```python
from model_utils import ModelEvaluator

# Evaluate model
metrics = ModelEvaluator.evaluate_model(model, test_generator)

print(f"Accuracy:  {metrics['accuracy']:.4f}")
print(f"Loss:      {metrics['loss']:.4f}")

# Get predictions
predictions, true_labels, classes = ModelEvaluator.get_predictions(
    model, 
    test_generator,
    class_names=CLASS_NAMES
)

# Calculate sklearn metrics
from sklearn.metrics import classification_report
print(classification_report(true_labels, predictions, target_names=classes))
```

---

## 🔧 API Reference

### Configuration Module (`config.py`)

```python
# Import configuration
from config import (
    CLASS_NAMES,           # ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
    NUM_CLASSES,          # 4
    DATA_DIR,             # Path to dataset
    TRANSFER_LEARNING_IMG_SIZE,  # (224, 224)
    DEFAULT_BATCH_SIZE,   # 64
    DEFAULT_EPOCHS,       # 10
    get_data_dir(),       # Get dataset directory
    verify_config(),      # Validate configuration
)
```

### Data Utilities (`data_utils.py`)

#### ImageValidator

```python
from data_utils import ImageValidator

validator = ImageValidator()
validator.is_valid_extension('image.jpg')      # True/False
validator.is_valid_file_size('image.jpg')      # True/False
validator.is_readable('image.jpg')             # True/False
```

#### ImageLoader

```python
from data_utils import ImageLoader

loader = ImageLoader()
df = loader.load_dataset(
    target_size=(224, 224),
    grayscale=False,
    normalize=True
)
# Returns DataFrame with columns: filepath, label, class_name
```

#### DataPreprocessor

```python
from data_utils import DataPreprocessor

# Split data
train_df, test_df = DataPreprocessor.train_test_split_data(
    df, 
    test_size=0.2,
    stratify=True
)

# Validate distribution
DataPreprocessor.validate_class_distribution(df, expected_distribution)
```

### Model Utilities (`model_utils.py`)

#### ModelBuilder

```python
from model_utils import ModelBuilder

# Transfer learning model
model = ModelBuilder.build_transfer_learning_model(
    'efficientnetb3',
    num_classes=4,
    input_shape=(224, 224, 3),
    freeze_base=True,
    dropout_rate=0.5
)

# Baseline CNN
model = ModelBuilder.build_baseline_cnn(
    input_shape=(224, 224, 3),
    num_classes=4
)
```

#### ModelCompiler

```python
from model_utils import ModelCompiler

model = ModelCompiler.compile_model(
    model,
    learning_rate=0.001,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

#### ModelTrainer

```python
from model_utils import ModelTrainer

history = ModelTrainer.train_model(
    model,
    train_generator,
    test_generator,
    epochs=20,
    model_name='efficientnetb3',
    verbose=1
)
```

#### ModelEvaluator

```python
from model_utils import ModelEvaluator

metrics = ModelEvaluator.evaluate_model(
    model,
    test_generator,
    model_name='efficientnetb3'
)

predictions, labels, classes = ModelEvaluator.get_predictions(
    model,
    test_generator,
    class_names=CLASS_NAMES
)
```

---

## ⚙️ Configuration Guide

### Dataset Configuration

```python
# config.py
DATA_DIR = Path("dataset")  # Dataset root directory
CLASS_NAMES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
NUM_CLASSES = 4

# Image size (transfer learning uses 224x224, traditional ML uses 128x128)
TRANSFER_LEARNING_IMG_SIZE = (224, 224)
TRADITIONAL_ML_IMG_SIZE = (128, 128)
```

### Training Configuration

```python
# Default training parameters
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_OPTIMIZER = "adam"
DEFAULT_LOSS = "categorical_crossentropy"

# Data splitting
TRAIN_TEST_SPLIT = 0.2        # 80-20 split
VALIDATION_SPLIT = 0.1        # 10% validation
RANDOM_STATE = 42             # Reproducibility

# Data augmentation
DATA_AUGMENTATION_CONFIG = {
    "rotation_range": 20,
    "width_shift_range": 0.15,
    "height_shift_range": 0.15,
    "shear_range": 0.2,
    "zoom_range": 0.2,
    "horizontal_flip": False,
    "vertical_flip": True,
    "fill_mode": "nearest",
}
```

### Logging Configuration

```python
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
LOG_FILE = Path("logs/medeye.log")
```

### Error Handling

```python
SKIP_CORRUPTED_IMAGES = True  # Skip instead of failing
LOG_CORRUPTED_IMAGES = True   # Log problematic images
MAX_RETRY_ATTEMPTS = 3        # Retry failed operations

# Image validation
MIN_IMAGE_WIDTH = 50
MIN_IMAGE_HEIGHT = 50
MAX_IMAGE_SIZE_MB = 50
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. **"Dataset directory not found"**

```python
# Solution: Ensure dataset folder structure is correct
# Check: dataset/cataract/, dataset/glaucoma/, etc. exist with images
import os
for disease in ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']:
    path = f'dataset/{disease}'
    if os.path.exists(path):
        print(f"✓ {disease}: {len(os.listdir(path))} images")
    else:
        print(f"✗ {disease}: NOT FOUND")
```

#### 2. **"Out of memory" during training**

```python
# Solution: Reduce batch size and epochs
python train_model.py --model efficientnetb3 --batch-size 16 --epochs 5
```

#### 3. **"No GPU found"**

```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If empty, install GPU support:
pip install tensorflow[and-cuda]
```

#### 4. **"Corrupted images detected"**

```python
# Check which images are problematic
from data_utils import ImageLoader
loader = ImageLoader()
df = loader.load_dataset((224, 224), grayscale=False, normalize=True)
print("Corrupted images:", loader.corrupted_images)
# Then remove/replace these images
```

#### 5. **"Model not converging"**

```python
# Solutions:
# 1. Increase learning rate (too low, 0.01)
# 2. Check data augmentation settings
# 3. Verify class balance
# 4. Increase training epochs
# 5. Check for data leakage
```

### Getting Help

1. **Check Logs:** Review `logs/medeye.log` for detailed error messages
2. **Test Configuration:** Run `python -c "from config import verify_config; verify_config()"`
3. **Validate Data:** Use ImageValidator to check image quality
4. **Check Dependencies:** Run `pip list` to verify all packages installed

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints for better IDE support
- Write unit tests for new features
- Update documentation accordingly

### Reporting Issues

- Use GitHub Issues for bug reports
- Include detailed description and steps to reproduce
- Attach logs and error messages
- Specify Python version and dependencies

### Code of Conduct

Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community guidelines.

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

### License Summary

- ✅ **Use** for personal, academic, or commercial projects
- ✅ **Modify** the code as needed
- ✅ **Distribute** with proper attribution
- ❌ **No Warranty** - use at your own risk
- ❌ **No Liability** - authors not responsible for outcomes

### Attribution

When using MedEye, please cite:

```bibtex
@software{medeye2026,
  title={MedEye: Medical Eye Disease Detection System},
  author={Daud, Muhammad},
  year={2026},
  url={https://github.com/yourusername/medeye},
  license={MIT}
}
```

---

## 📚 Citation and References

### Cite This Project

**APA Format:**
```
Daud, M. (2026). MedEye: Medical Eye disease detection system (Version 2.0) 
[Computer software]. https://github.com/yourusername/medeye
```

**Chicago Format:**
```
Daud, Muhammad. "MedEye: Medical Eye Disease Detection System." 
Version 2.0. GitHub, 2026. https://github.com/yourusername/medeye.
```

### Key References

- **TensorFlow Documentation:** https://www.tensorflow.org/
- **Transfer Learning:** Tan, M., & Le, Q. (2019). EfficientNet paper
- **Medical Imaging:** Standards and best practices from MICCAI
- **Retinal Disease Classification:** IDRiD Dataset publication

---

## 📞 Contact and Support

### Getting Help

- **Documentation:** See this README and inline code comments
- **Issues:** GitHub Issues for bug reports and feature requests
- **Discussions:** GitHub Discussions for questions and ideas

### Project Information

- **Repository:** https://github.com/yourusername/medeye
- **Version:** 2.0
- **Last Updated:** 2026
- **Status:** ✅ Production Ready

### Connect

- **Author:** Muhammad Daud
- **Email:** [contact-email@example.com]
- **GitHub:** [@yourusername](https://github.com/yourusername)

---

## 📈 Project Roadmap

### Version 2.1 (Planned)

- [ ] Real-time inference optimization
- [ ] Web API with FastAPI
- [ ] Model quantization for edge devices
- [ ] Extended dataset support
- [ ] CI/CD pipeline integration

### Version 3.0 (Future)

- [ ] Ensemble prediction methods
- [ ] Explainability features (Grad-CAM visualization)
- [ ] Multi-disease hierarchical classification
- [ ] Privacy-preserving federated learning
- [ ] Cloud deployment templates (AWS, GCP, Azure)

---

## 🎉 Acknowledgments

- **Dataset Sources:** IDRiD, Ocular Recognition, HRF
- **Pre-trained Models:** TensorFlow/Keras Model Zoo
- **Community:** Contributors and users who improve the project
- **Libraries:** TensorFlow, scikit-learn, OpenCV, and other dependencies

---

**Made with ❤️ by Muhammad Daud**

*Last Updated: 2026 | Licensed under MIT | Status: Production Ready*
   - Description: Utilizes the Xception architecture for improved performance.
   - Directory: `/RetinaXpert/Xception Model`

### Machine Learning Models
1. **SVM Model**
   - Description: Support Vector Machine model for disease classification.
   - Directory: `/RetinaXpert/SVM Model`

2. **Random Forest Model**
   - Description: Implements a Random Forest classifier for accurate predictions.
   - Directory: `/RetinaXpert/Random Forest Model`

3. **Decision Tree**
   - Description: Utilizes a Decision Tree algorithm for disease categorization.
   - Directory: `/RetinaXpert/Decision Tree Model`


## Introduction 🌟

Eye diseases can significantly impact vision, and early detection is crucial for effective treatment. RetinaXpert addresses this challenge by leveraging state-of-the-art models to analyze retinal images and identify conditions such as glaucoma, cataracts, diabetic retinopathy, and normal cases.


## Elaborate Study PDF 📑

In the `/RetinaXpert/Elaborate_Study.pdf` directory, you will find a comprehensive PDF document providing an in-depth comparative analysis of the model performance. This study includes detailed insights into the strengths, weaknesses, and nuances of each model, along with visualizations of key metrics and performance on specific subsets of the dataset.


## Aim and Objective 🎯

The primary goals of RetinaXpert include:

- **Disease Recognition:** Accurately identify and categorize eye diseases.
- **High Accuracy:** Achieve reliable and precise predictions.
- **Early Detection:** Detect eye diseases at an early stage for timely intervention.
- **Efficiency:** Develop models that balance accuracy and computational efficiency.
- **Practical Applicability:** Ensure the models are applicable in real-world clinical scenarios.
- **Generalizability:** Create models that generalize well to diverse datasets and patient populations.
- **Adherence to Ethical Standards:** Prioritize ethical considerations in the development and deployment of the models.
- **Interpretability:** Provide insights into model decisions for better understanding by healthcare professionals.
- **Continuous Learning:** Enable the model to adapt and improve over time with new data and insights.
- **Validation:** Rigorously validate the models to ensure their reliability and safety in a healthcare context.


## Getting Started 🚀

To use the RetinaXpert models, refer to the specific model directories for detailed instructions on how to load, train, and evaluate each model. Additionally, ensure that you have the necessary dependencies installed as specified in the project's requirements.


## Feedback and Contributions 🤝

RetinaXpert welcomes feedback and contributions from the community. If you encounter issues, have suggestions, or want to contribute improvements, please follow the guidelines outlined in [CONTRIBUTING.md](CONTRIBUTING.md)

Thank you for choosing RetinaXpert! We hope our models contribute to the advancement of eye disease diagnosis and treatment.
