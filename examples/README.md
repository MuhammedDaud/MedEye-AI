# MedEye Examples & Tutorials

This directory contains example notebooks and tutorials for using the MedEye framework.

## 📚 Available Examples

### 1. **01_quick_start.ipynb**
**Introduction to MedEye** - Get started in 5 minutes
- Load the retinal image dataset
- Train a model with default configuration
- Evaluate and visualize results
- Make predictions on new images

**Best for:** First-time users, quick prototyping

### 2. **02_model_training_guide.ipynb**
**Complete Training Guide** - In-depth training walkthrough
- Dataset exploration and visualization
- Data validation and augmentation
- Training different model architectures
- Comparing model performance
- Saving and loading models

**Best for:** Understanding the training pipeline, hyperparameter tuning

### 3. **03_model_comparison.ipynb**
**Model Comparison Analysis** - Compare all 10 models
- Train and evaluate each model
- Create comparison charts
- Analyze speed vs accuracy trade-offs
- Visualize confusion matrices
- Generate comprehensive reports

**Best for:** Model selection, performance analysis

### 4. **04_inference_and_deployment.ipynb**
**Inference & Deployment Guide** - Use trained models
- Load pre-trained models
- Make predictions on new images
- Batch inference on multiple images
- Export models for deployment
- Integration with your applications

**Best for:** Production use, deploying models

### 5. **05_custom_training.ipynb**
**Custom Training Configuration** - Advanced training
- Fine-tune model hyperparameters
- Implement custom data augmentation
- Use custom loss functions
- Multi-GPU training
- Monitor training with TensorBoard

**Best for:** Advanced users, optimization

### 6. **06_data_exploration.ipynb**
**Dataset Exploration** - Analyze your data
- Dataset statistics and distribution
- Image visualization
- Disease characteristics
- Data quality checks
- Visualization techniques

**Best for:** Understanding your data, data preprocessing

### 7. **07_troubleshooting_guide.ipynb**
**Common Issues & Solutions** - Debug your code
- Dataset loading issues
- GPU memory problems
- Model training issues
- Performance problems
- Data validation errors

**Best for:** Troubleshooting, debugging

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter or JupyterLab
- MedEye dependencies (see requirements.txt)
- Dataset folder with images

### Setup

1. **Install Jupyter** (if not already installed):
```bash
pip install jupyter jupyterlab
```

2. **Install MedEye dependencies**:
```bash
pip install -r ../requirements.txt
```

3. **Start Jupyter**:
```bash
jupyter lab
```

4. **Open a notebook**:
- Navigate to the examples folder
- Click on any `.ipynb` file to open

## 📝 Notebook Structure

Each notebook follows a consistent structure:

1. **Introduction** - What you'll learn
2. **Setup** - Import libraries and configure
3. **Main Content** - Step-by-step examples
4. **Exercises** - Practice problems (optional)
5. **Summary** - Key takeaways
6. **Further Reading** - Advanced topics

## 💡 Tips for Using Notebooks

### Installation & Running
```bash
# Run notebook from command line
jupyter nbconvert --to notebook --execute 01_quick_start.ipynb

# Run specific cells
jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name=python 01_quick_start.ipynb
```

### Code Cells
- **Code cells** - Executable Python code
- **Markdown cells** - Documentation and explanations
- Use `Shift+Enter` to run a cell
- Use `Ctrl+Shift+P` for command palette

### Common Notebook Commands
```python
# Display plots inline
%matplotlib inline

# Reload changed modules
%reload_ext autoreload
%autoreload 2

# Measure execution time
%timeit function_call()

# Debug with ipdb
%debug
```

## 🔗 Integration with Main Code

The notebooks import from the main project modules:

```python
from config import CLASS_NAMES, TRANSFER_LEARNING_IMG_SIZE
from data_utils import ImageLoader, DataPreprocessor, create_data_generators
from model_utils import ModelBuilder, ModelCompiler, ModelTrainer, ModelEvaluator
from train_model import train_transfer_learning_model
```

## 📊 Output Examples

Running these notebooks will generate:
- Training plots and loss curves
- Confusion matrices and classification reports
- Model comparison charts
- Prediction visualizations
- Performance metrics and reports

## 🎓 Learning Path

**Recommended order for beginners:**
1. Start with `01_quick_start.ipynb`
2. Explore with `06_data_exploration.ipynb`
3. Learn training with `02_model_training_guide.ipynb`
4. Compare models with `03_model_comparison.ipynb`
5. Deploy with `04_inference_and_deployment.ipynb`

**For advanced users:**
1. Skip to `05_custom_training.ipynb`
2. Reference `07_troubleshooting_guide.ipynb` as needed
3. Create your own notebooks building on examples

## 🐛 Troubleshooting

### Notebook won't start
```bash
# Clear Jupyter cache
rm -rf ~/.jupyter

# Reinstall Jupyter
pip install --upgrade jupyter
```

### Import errors
```bash
# Ensure you're in the correct directory
import sys
sys.path.insert(0, '../')

# Or install project in development mode
pip install -e ..
```

### Out of memory
- Reduce batch size in notebooks
- Use smaller dataset subset for testing
- Close other applications

## 📚 Additional Resources

- **MedEye Documentation:** See [../README.md](../README.md)
- **API Reference:** See [../README.md#api-reference](../README.md#api-reference)
- **Contributing:** See [../CONTRIBUTING.md](../CONTRIBUTING.md)
- **Configuration:** See [../README.md#configuration](../README.md#configuration)

## 🤝 Contributing Examples

Have a useful example or notebook? We'd love to include it!

1. Create your notebook with clear explanations
2. Test it thoroughly
3. Add to this README
4. Submit a PR with your contribution

---

**Happy Learning!** 🚀

For questions or issues with the examples, please:
- Open an issue on GitHub
- Check existing examples for solutions
- See [../CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines

**Made with ❤️ by the MedEye Team**
