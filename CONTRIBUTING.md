# Contributing to MedEye

Thank you for your interest in contributing to **MedEye**! We welcome contributions from everyone. This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Commit Messages](#commit-messages)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)

---

## 🤝 Code of Conduct

Please read our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing. All participants are expected to follow these guidelines to create a welcoming and inclusive community.

**TL;DR**: Be respectful, inclusive, and professional. No harassment, discrimination, or hostile behavior.

---

## 🚀 Getting Started

### Fork and Clone

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/MedEye-AI.git
   cd MedEye-AI
   ```
3. **Add** upstream remote:
   ```bash
   git remote add upstream https://github.com/MuhammedDaud/MedEye-AI.git
   ```

### Create a Branch

Always create a new branch for your work:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

**Branch naming convention:**
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions

---

## 🛠️ Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment manager (venv, conda, or pipenv)

### Step 1: Create Virtual Environment

```bash
# Using venv
python -m venv medeye_env
source medeye_env/bin/activate  # On Windows: medeye_env\Scripts\activate

# Or using conda
conda create -n medeye python=3.9
conda activate medeye
```

### Step 2: Install Dependencies

```bash
# Install project requirements
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 isort
```

### Step 3: Verify Setup

```bash
# Run tests
pytest tests/

# Check your code
python -c "import tensorflow; import cv2; import sklearn; print('✓ Setup successful')"
```

---

## 💡 How to Contribute

### Types of Contributions

We welcome all kinds of contributions:

#### 🐛 **Bug Fixes**
- Fix issues identified in the project
- Use the `fix/` branch naming convention
- Include test case for the bug

#### ✨ **New Features**
- Add new models, utilities, or functionality
- Use the `feature/` branch naming convention
- Include comprehensive documentation
- Add unit tests for new code

#### 📚 **Documentation**
- Improve README, docstrings, or guides
- Fix typos and clarify explanations
- Add examples and tutorials
- Use the `docs/` branch naming convention

#### 🧪 **Tests**
- Improve test coverage
- Add tests for untested code paths
- Use the `test/` branch naming convention

#### ♻️ **Refactoring**
- Improve code quality and efficiency
- Refactor for clarity and maintainability
- Use the `refactor/` branch naming convention

---

## 🔄 Pull Request Process

### Before Creating a PR

1. **Update** your branch with latest upstream:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run** all tests locally:
   ```bash
   pytest tests/ -v
   ```

3. **Format** your code:
   ```bash
   black .
   isort .
   flake8 .
   ```

### Creating a Pull Request

1. **Push** your branch:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open** a PR on GitHub with:
   - Clear title: `[Type] Brief description`
   - Detailed description of changes
   - Reference to related issues (e.g., "Fixes #123")
   - Screenshots for UI changes
   - Test results

3. **PR Title Format**:
   ```
   [Feature] Add EfficientNetB4 model support
   [Fix] Resolve dataset validation issue
   [Docs] Update installation guide
   [Test] Add data_utils unit tests
   [Refactor] Improve model_utils efficiency
   ```

### PR Description Template

```markdown
## Description
Brief explanation of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Test addition

## Related Issue
Fixes #(issue number)

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
How was this tested?

## Checklist
- [ ] Code follows PEP 8 guidelines
- [ ] Tests pass locally (`pytest tests/`)
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
- [ ] Commit messages are clear and descriptive
```

### Review Process

- Maintainers will review your PR
- Address feedback and push updates
- PR may require approval before merging
- Continuous integration must pass

---

## 📝 Coding Standards

### Python Style Guide

Follow **PEP 8** with these specific guidelines:

#### Imports
```python
# Standard library imports first
import os
import sys
from pathlib import Path

# Third-party imports next
import numpy as np
import pandas as pd
import tensorflow as tf

# Local imports last
from config import CLASS_NAMES, NUM_CLASSES
from data_utils import ImageLoader
```

#### Functions and Classes
```python
def load_image(filepath: str, target_size: tuple) -> np.ndarray:
    """
    Load and preprocess image from file.
    
    Args:
        filepath: Path to image file
        target_size: Target dimensions (height, width)
        
    Returns:
        Preprocessed image as numpy array
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image dimensions are invalid
    """
    # Implementation
    pass

class ImageValidator:
    """Validates image files for integrity and format."""
    
    def is_valid_file(self, filepath: str) -> bool:
        """Check if image file is valid."""
        pass
```

#### Documentation
- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Include type hints for parameters and return values
- Document exceptions raised

#### Code Quality
```python
# ✅ Good
result = process_data(config, verbose=True)
classification_report = get_detailed_metrics(predictions, labels)

# ❌ Avoid
res = proc_data(cfg, v=1)
cr = get_metrics(p, l)
```

### Key Standards

- **Line length**: Max 100 characters
- **Naming**: 
  - Functions/variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_CASE`
- **Imports**: Alphabetically sorted within groups
- **Formatting**: Use `black` formatter
- **Linting**: Pass `flake8` checks
- **Sorting**: Use `isort` for consistent imports

---

## 🧪 Testing Guidelines

### Test Structure

```python
# tests/test_module.py
import pytest
from module import FunctionUnderTest

class TestFunctionUnderTest:
    """Test cases for FunctionUnderTest."""
    
    def setup_method(self):
        """Setup before each test."""
        self.test_data = [1, 2, 3, 4, 5]
    
    def test_valid_input(self):
        """Test with valid input."""
        result = FunctionUnderTest(self.test_data)
        assert result == expected_value
    
    def test_invalid_input(self):
        """Test with invalid input."""
        with pytest.raises(ValueError):
            FunctionUnderTest(None)
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test
pytest tests/test_data_utils.py::TestImageValidator::test_valid_extension

# Run with verbose output
pytest tests/ -v

# Run with markers
pytest tests/ -m "not slow"
```

### Test Requirements

- ✅ Every new feature must have tests
- ✅ Bug fixes should include test for the bug
- ✅ Aim for >80% code coverage
- ✅ Tests should be independent and repeatable
- ✅ Use meaningful test names describing what is tested

---

## 📌 Commit Messages

Follow the conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `style:` - Formatting (no code change)
- `refactor:` - Code refactoring
- `test:` - Test addition/modification
- `chore:` - Build/dependencies changes
- `perf:` - Performance improvement

### Examples

```
feat(model): Add EfficientNetB4 support

- Implement EfficientNetB4 in ModelBuilder
- Add configuration parameters
- Update model comparison table
- Add unit tests

Fixes #123
```

```
fix(data): Resolve corrupted image detection

ImageValidator was incorrectly flagging valid images.
Improved dimension validation logic.

Fixes #456
```

```
docs: Update installation guide for GPU setup
```

---

## 🐛 Reporting Bugs

### Bug Report Template

When reporting a bug, please use this format:

```markdown
## Description
Brief description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [Windows/Linux/macOS]
- Python: [e.g., 3.9.1]
- TensorFlow: [e.g., 2.10.0]
- Other: [Any other relevant info]

## Logs
```
<error logs or output>
```

## Screenshots
[If applicable]

## Possible Solution
[If you have a suggested fix]
```

### Bug Report Best Practices

- ✅ Use clear, descriptive titles
- ✅ Provide minimal reproduction case
- ✅ Include environment details
- ✅ Attach error logs and stack traces
- ✅ Specify Python and package versions
- ✅ Explain expected vs actual behavior
- ✅ Include steps to reproduce

---

## 💭 Suggesting Features

### Feature Request Template

```markdown
## Description
Clear description of the feature

## Motivation
Why would this feature be useful?

## Proposed Solution
How should this work?

## Example Usage
```python
# Show example code
```

## Alternative Solutions
Are there other approaches?

## Additional Context
Any other relevant information
```

### Feature Request Best Practices

- ✅ Provide clear use cases
- ✅ Explain why it's beneficial
- ✅ Show example code if applicable
- ✅ Consider backward compatibility
- ✅ Discuss performance implications

---

## 🎓 Learning Resources

- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [TensorFlow Contributing Guide](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md)
- [GitHub Fork and Pull Request Workflow](https://docs.github.com/en/get-started/quickstart/contributing-to-projects)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

## ❓ Questions?

- **Documentation**: Check [README.md](README.md)
- **GitHub Issues**: Search existing issues
- **GitHub Discussions**: Ask questions in discussions
- **Contact**: Reach out to maintainers via email

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## 🙏 Thank You!

Thank you for considering contributing to MedEye! Your contributions, whether code, documentation, or feedback, help make this project better for everyone.

**Made with ❤️ by the MedEye Community**
