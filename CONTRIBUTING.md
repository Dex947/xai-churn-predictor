# Contributing to Customer Churn Prediction System

Thank you for your interest in contributing to the Customer Churn Prediction System! This document provides guidelines for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## 🤝 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors. We expect respectful, professional conduct from all participants.

### Our Standards

**Positive behaviors include:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Accepting constructive criticism gracefully
- Focusing on what's best for the project
- Showing empathy towards other contributors

**Unacceptable behaviors include:**
- Harassment, trolling, or insulting comments
- Publishing others' private information
- Any conduct that could be considered unprofessional

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of machine learning
- Familiarity with scikit-learn, XGBoost, or similar libraries

### Fork and Clone

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/churn-prediction-xai.git
   cd churn-prediction-xai
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/churn-prediction-xai.git
   ```

## 🛠️ How to Contribute

### Types of Contributions

We welcome various types of contributions:

#### 🐛 Bug Fixes
- Report bugs via GitHub Issues
- Include steps to reproduce
- Fix bugs and submit pull requests

#### ✨ New Features
- Discuss new features in GitHub Issues first
- Get approval before starting major work
- Document new features thoroughly

#### 📚 Documentation
- Improve README, docstrings, or guides
- Fix typos or clarify explanations
- Add examples or tutorials

#### 🧪 Tests
- Add test coverage
- Improve existing tests
- Add edge case testing

#### 🎨 Code Quality
- Refactor for better readability
- Improve performance
- Add type hints

## 💻 Development Setup

### 1. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n churn-pred python=3.10
conda activate churn-pred
```

### 2. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy
```

### 3. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_pipeline.py -v
```

### 4. Verify Setup

```bash
# Run the pipeline
python main.py --skip-download

# Launch dashboard
streamlit run app.py
```

## 📝 Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

```python
# Good: Clear, documented, type-hinted
def preprocess_data(
    df: pd.DataFrame,
    target_column: str = "Churn",
    test_size: float = 0.2
) -> Dict[str, pd.DataFrame]:
    """
    Preprocess the customer churn dataset.

    Args:
        df: Raw DataFrame with customer data
        target_column: Name of the target variable
        test_size: Proportion for test split

    Returns:
        Dictionary with train and test DataFrames
    """
    # Implementation here
    pass
```

### Code Formatting

**Use Black for formatting:**
```bash
black src/ tests/ --line-length 100
```

**Use flake8 for linting:**
```bash
flake8 src/ tests/ --max-line-length 100
```

**Use mypy for type checking:**
```bash
mypy src/ --ignore-missing-imports
```

### Documentation Standards

**Docstrings:** Use Google-style docstrings

```python
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """
    Train a machine learning model.

    Args:
        X_train: Training features of shape (n_samples, n_features)
        y_train: Training labels of shape (n_samples,)

    Returns:
        Trained model instance

    Raises:
        ValueError: If input data is invalid
    """
```

**Comments:** Explain "why", not "what"

```python
# Good: Explains reasoning
# Use SMOTE to balance classes because our dataset has 73% negative class
X_balanced, y_balanced = smote.fit_resample(X, y)

# Bad: States the obvious
# Apply SMOTE
X_balanced, y_balanced = smote.fit_resample(X, y)
```

### Configuration

- **Never hardcode values** - use `config/config.yaml`
- **Use logging** - not print statements
- **Handle errors gracefully** - with try-except blocks

## 🧪 Testing Guidelines

### Writing Tests

**Test Structure:**

```python
import pytest
from src.preprocessing import DataPreprocessor

class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': ['A', 'B', 'C'],
            'target': [0, 1, 0]
        })

    def test_handle_missing_values(self, sample_data):
        """Test missing value handling."""
        preprocessor = DataPreprocessor()
        result = preprocessor.handle_missing_values(sample_data)
        assert result.isnull().sum().sum() == 0

    def test_encode_categorical(self, sample_data):
        """Test categorical encoding."""
        preprocessor = DataPreprocessor()
        result = preprocessor.encode_categorical(sample_data)
        assert result.shape[1] > sample_data.shape[1]
```

### Test Coverage

- **Minimum 70% coverage** for new code
- **Test edge cases** and error conditions
- **Mock external dependencies** (API calls, file I/O)

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test
pytest tests/test_preprocessing.py::TestDataPreprocessor::test_encode_categorical -v

# Run tests in parallel
pytest tests/ -n auto
```

## 📥 Pull Request Process

### Before Submitting

1. **Sync with upstream:**
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes:**
   - Write code
   - Add tests
   - Update documentation

4. **Run quality checks:**
   ```bash
   black src/ tests/
   flake8 src/ tests/
   pytest tests/ --cov=src
   ```

5. **Commit changes:**
   ```bash
   git add .
   git commit -m "feat: Add new feature description"
   ```

### Commit Message Convention

We use **Conventional Commits**:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Examples:**
```bash
feat(models): Add gradient boosting classifier
fix(preprocessing): Handle missing values in TotalCharges
docs(README): Update installation instructions
test(evaluation): Add ROC curve calculation tests
```

### Submitting PR

1. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub with:
   - Clear title and description
   - Link to related issues
   - Screenshots (if UI changes)
   - Test results

3. **PR Template:**
   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation
   - [ ] Refactoring

   ## Checklist
   - [ ] Tests pass locally
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] No new warnings

   ## Related Issues
   Fixes #123
   ```

### Review Process

- **At least one approval** required
- **All tests must pass** (CI/CD)
- **No merge conflicts** with main
- **Code review feedback** addressed

## 🐛 Issue Reporting

### Bug Reports

**Use this template:**

```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step one
2. Step two
3. See error

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: Windows 10
- Python: 3.10
- Package versions: (from requirements.txt)

## Screenshots
If applicable

## Additional Context
Any other relevant information
```

### Feature Requests

**Use this template:**

```markdown
## Feature Description
Clear description of the proposed feature

## Motivation
Why this feature is needed

## Proposed Solution
How it could be implemented

## Alternatives Considered
Other approaches you've thought about

## Additional Context
Any other relevant information
```

## 🏗️ Project Structure

### Where to Add Code

```
src/
├── ingestion/          # Add data loading features
├── preprocessing/      # Add preprocessing logic
├── models/            # Add new models
├── evaluation/        # Add evaluation metrics
├── explainability/    # Add XAI methods
├── visualization/     # Add plotting functions
└── utils/            # Add utility functions
```

### Adding New Models

1. **Define in config** (`config/config.yaml`):
   ```yaml
   models:
     new_model:
       enabled: true
       params:
         param1: value1
   ```

2. **Add to ModelTrainer** (`src/models/model_trainer.py`):
   ```python
   elif model_name == "new_model":
       from sklearn.ensemble import SomeClassifier
       model = SomeClassifier(**params)
   ```

3. **Add tests** (`tests/test_models.py`)
4. **Update documentation**

## 📚 Additional Resources

### Learning Resources

- **scikit-learn**: https://scikit-learn.org/
- **XGBoost**: https://xgboost.readthedocs.io/
- **SHAP**: https://shap.readthedocs.io/
- **Streamlit**: https://docs.streamlit.io/

### Project Documentation

- [README.md](README.md) - Quick start guide
- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) - Technical details
- [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) - Performance analysis

## 💬 Communication

### Questions?

- **GitHub Issues**: For bugs and features
- **Discussions**: For general questions
- **Documentation**: Check existing docs first

### Getting Help

1. Check documentation
2. Search existing issues
3. Ask in Discussions
4. Create new issue if needed

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Acknowledged in documentation

Thank you for contributing! 🚀

---

## Quick Checklist for Contributors

- [ ] Forked and cloned repository
- [ ] Created virtual environment
- [ ] Installed dependencies
- [ ] Created feature branch
- [ ] Made changes with tests
- [ ] Ran code quality checks
- [ ] Committed with conventional commits
- [ ] Pushed to fork
- [ ] Created pull request
- [ ] Addressed review feedback

---

*Last Updated: October 2025*
*For questions, open an issue on GitHub*
