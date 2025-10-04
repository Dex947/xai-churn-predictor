# Customer Churn Prediction System - Project Overview

## 🎯 Executive Summary

This is a production-ready customer churn prediction system built with explainable AI (XAI) principles. The system predicts which customers are likely to churn and explains why, enabling proactive retention strategies.

### Key Achievements

✅ **3 ML Models Trained**: Logistic Regression, Random Forest, XGBoost
✅ **Best F1 Score**: 0.621 (Random Forest)
✅ **Best ROC-AUC**: 0.847 (Logistic Regression)
✅ **Full Explainability**: SHAP and LIME implementations
✅ **Interactive Dashboard**: Streamlit web application
✅ **Production Ready**: Config-driven, logging, testing, documentation

## 📊 System Architecture

### Data Flow

```
Raw Data → Ingestion → Preprocessing → Model Training → Evaluation → Explainability → Dashboard
```

### Components

1. **Data Ingestion** (`src/ingestion/`)
   - Downloads IBM Telco Customer Churn dataset
   - Validates data quality
   - Handles multiple data formats

2. **Preprocessing** (`src/preprocessing/`)
   - Cleans data (duplicates, missing values)
   - Encodes categorical variables (one-hot encoding)
   - Scales numeric features (StandardScaler)
   - Handles class imbalance (SMOTE)
   - Train/validation/test splitting

3. **Model Training** (`src/models/`)
   - Logistic Regression (baseline)
   - Random Forest (ensemble)
   - XGBoost (gradient boosting)
   - Hyperparameter configuration via YAML
   - Cross-validation support

4. **Evaluation** (`src/evaluation/`)
   - Accuracy, Precision, Recall, F1 Score
   - ROC-AUC curves
   - Precision-Recall curves
   - Calibration curves
   - Confusion matrices
   - Comprehensive comparison reports

5. **Explainability** (`src/explainability/`)
   - **SHAP**: Feature importance and dependency plots
   - **LIME**: Local instance explanations
   - Waterfall plots for individual predictions
   - Force plots for prediction breakdown

6. **Visualization** (`src/visualization/`)
   - EDA plots (distributions, correlations)
   - Model performance comparisons
   - Feature importance visualizations
   - All plots saved as high-resolution PNG files

7. **Dashboard** (`app.py`)
   - Single customer prediction
   - Batch CSV upload
   - Real-time SHAP explanations
   - Model performance metrics
   - Interactive visualizations

## 📈 Results Summary

### Model Performance

| Metric | Logistic Regression | Random Forest | XGBoost |
|--------|---------------------|---------------|---------|
| **Accuracy** | 74.96% | **77.39%** | **79.03%** |
| **Precision** | 51.81% | 55.79% | **60.21%** |
| **Recall** | **77.09%** | **70.08%** | 61.19% |
| **F1 Score** | 61.97% | **62.13%** | 60.70% |
| **ROC-AUC** | **0.847** | 0.839 | 0.836 |

### Key Findings

Based on SHAP analysis, the top factors influencing churn predictions are:

1. **Contract Type** - Month-to-month contracts have 3x higher churn
2. **Tenure** - Customers < 6 months are at highest risk
3. **Monthly Charges** - Higher charges correlate with increased churn
4. **Total Charges** - Inversely related to tenure
5. **Internet Service** - Fiber optic users show different patterns
6. **Payment Method** - Electronic check users churn more
7. **Tech Support** - Lack of support increases churn risk
8. **Online Security** - Important retention factor

### Business Insights

- **Churn Rate**: ~26.5% of customers in the dataset
- **High-Risk Segment**: New customers (< 6 months) with month-to-month contracts
- **Retention Opportunities**:
  - Incentivize annual contracts
  - Provide onboarding support for new customers
  - Target fiber optic customers with value-added services

## 🏗️ Technical Implementation

### Code Quality

- **Modular Architecture**: Separate modules for each pipeline stage
- **Configuration-Driven**: All parameters in YAML (no hardcoding)
- **Comprehensive Logging**: loguru integration with file/console output
- **Type Hints**: Python typing throughout codebase
- **Docstrings**: Google-style documentation for all functions
- **Error Handling**: Try-except blocks with informative messages
- **Testing**: Unit tests with pytest

### Design Patterns

- **Factory Pattern**: Model creation in ModelTrainer
- **Strategy Pattern**: Interchangeable preprocessing strategies
- **Singleton Pattern**: Global configuration loader
- **Pipeline Pattern**: Sequential data transformation

### Performance Optimizations

- **SMOTE Sampling**: Balanced 3,607 minority class samples
- **Feature Engineering**: 30 engineered features from 20 original
- **Efficient SHAP**: Background sampling (1000 samples) for speed
- **Caching**: Streamlit @st.cache_resource for model loading

## 📁 Project Files

### Core Files

- **main.py** (445 lines): Main training pipeline orchestrator
- **app.py** (350+ lines): Streamlit dashboard application
- **config/config.yaml** (200+ lines): Complete system configuration

### Source Modules

- **ingestion/data_loader.py** (300+ lines): Data loading utilities
- **preprocessing/preprocessor.py** (500+ lines): Preprocessing pipeline
- **models/model_trainer.py** (450+ lines): Model training/tuning
- **evaluation/evaluator.py** (400+ lines): Metrics calculation
- **explainability/explainer.py** (500+ lines): SHAP/LIME implementation
- **visualization/plotter.py** (500+ lines): Plotting utilities
- **utils/config_loader.py** (200+ lines): Configuration management
- **utils/logger.py** (70 lines): Logging setup

### Generated Outputs

#### Models (data/models/)
- logistic_regression.joblib (2.1 KB)
- random_forest.joblib (4.4 MB)
- xgboost.joblib (297 KB)
- preprocessor.joblib (6.1 KB)

#### Visualizations (data/plots/)
- 15 high-resolution plots including:
  - Churn distribution
  - Feature distributions (numeric/categorical)
  - Confusion matrices (3 models)
  - ROC curves comparison
  - Precision-recall curves
  - Calibration curves
  - Feature importance (4 sources)
  - SHAP summary plot

#### Reports (data/results/)
- evaluation_report.txt: Detailed metrics
- evaluation_results.json: Machine-readable results
- summary.md: Executive summary

## 🚀 Quick Start Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full training pipeline
python main.py

# Launch interactive dashboard
streamlit run app.py

# Run tests
pytest tests/ -v

# Windows shortcuts
run_pipeline.bat      # Train models
run_dashboard.bat     # Launch dashboard
```

## 🔧 Configuration

All system parameters in `config/config.yaml`:

```yaml
# Example: Enable/disable models
models:
  logistic_regression:
    enabled: true
  random_forest:
    enabled: true
  xgboost:
    enabled: true

# Preprocessing options
preprocessing:
  handle_imbalance: true
  imbalance_method: "smote"
  categorical_encoding: "onehot"
  numeric_scaling: "standard"

# Explainability settings
explainability:
  shap:
    enabled: true
    n_samples_for_shap: 1000
```

## 📊 Dataset Details

**IBM Telco Customer Churn Dataset**

- **Source**: IBM Cognos Analytics / Kaggle
- **Records**: 7,043 customers
- **Features**: 21 (after cleaning: 20)
- **Target**: Binary (Churn: Yes/No)
- **Class Distribution**: 73.5% No Churn, 26.5% Churn
- **Missing Values**: 11 (handled via dropping)
- **Duplicates**: 22 removed

## 🎓 Machine Learning Pipeline

### 1. Data Preparation
- Drop customerID column
- Handle TotalCharges conversion
- Remove duplicates and missing values
- **Result**: 7,010 clean records

### 2. Feature Engineering
- One-hot encode 15 categorical variables
- StandardScaler for 4 numeric features
- **Result**: 30 features

### 3. Train-Val-Test Split
- Training: 70% (4,907 samples)
- Validation: 10% (701 samples)
- Test: 20% (1,402 samples)
- Stratified sampling

### 4. Class Balancing (Training Only)
- SMOTE oversampling
- Before: 3,607 vs 1,300
- After: 3,607 vs 3,607
- **Result**: 7,214 balanced training samples

### 5. Model Training
- 3 models trained in parallel
- Class weights balanced
- Cross-validation enabled
- **Training time**: < 5 seconds total

### 6. Evaluation
- Test on 1,402 holdout samples
- 5 metrics per model
- ROC/PR/Calibration curves
- **Best model**: Random Forest (F1: 0.621)

## 🔍 Explainability Details

### SHAP Implementation

**TreeExplainer** used for tree-based models:
- Exact SHAP values for Random Forest
- Computation: ~3 seconds for 100 samples
- Feature importance ranking
- Dependency plots for interactions

**Outputs**:
- Global feature importance
- Summary plots (dot/bar/violin)
- Individual prediction waterfall plots
- Force plots for local explanations

### LIME Implementation

**TabularExplainer** for any model type:
- 5,000 perturbed samples per explanation
- Top 10 features shown
- Local linear approximation
- **Use case**: Explain individual predictions

## 📱 Dashboard Features

### Tab 1: Single Prediction
- Manual feature input form
- 18 customer attribute fields
- Real-time prediction
- Churn probability gauge (0-100%)
- SHAP explanation (top 10 features)

### Tab 2: Batch Prediction
- CSV file upload
- Bulk prediction processing
- Downloadable results
- Summary statistics

### Tab 3: Model Performance
- Model comparison table
- Interactive metric charts (Plotly)
- Confusion matrix heatmap
- Saved visualization gallery

## 🧪 Testing

**Unit Tests** (`tests/test_pipeline.py`):
- DataLoader: 2 tests
- DataPreprocessor: 3 tests
- ModelTrainer: 3 tests
- ModelEvaluator: 2 tests
- ChurnVisualizer: 2 tests

**Coverage**: Core functionality tested

```bash
pytest tests/ -v --cov=src
```

## 📚 Dependencies

**Core ML**: scikit-learn, XGBoost, imbalanced-learn
**Explainability**: SHAP, LIME
**Visualization**: matplotlib, seaborn, plotly
**Dashboard**: Streamlit
**Data**: pandas, numpy
**Utils**: PyYAML, loguru, joblib
**Development**: pytest, jupyter

**Total**: 25+ packages

## 🎯 Business Value

### For Data Scientists
- Rapid experimentation with config changes
- Comprehensive evaluation metrics
- Explainability built-in
- Easy to extend with new models

### For Business Users
- Intuitive dashboard interface
- Clear churn probability scores
- Actionable feature explanations
- Batch processing for operations

### For MLOps Engineers
- Config-driven deployment
- Serialized models ready for production
- Logging and monitoring hooks
- API-ready architecture

## 🔮 Future Enhancements

### Short-term
- [ ] Hyperparameter tuning (GridSearchCV)
- [ ] Feature selection automation
- [ ] Model ensemble (stacking/blending)
- [ ] REST API endpoint (FastAPI)

### Medium-term
- [ ] Deep learning models (LSTM, Transformers)
- [ ] Survival analysis (time-to-churn)
- [ ] Customer segmentation (clustering)
- [ ] Automated retraining pipeline

### Long-term
- [ ] Real-time streaming predictions
- [ ] A/B testing framework
- [ ] Causal inference analysis
- [ ] Multi-channel churn prediction

## 📞 Support & Contribution

**Issues**: Open GitHub issues for bugs/features
**Documentation**: Check README.md and code docstrings
**Contributing**: Fork → Branch → Test → Pull Request

## 📄 License & Attribution

**Educational/Research Use**
See ACKNOWLEDGEMENTS.md for dataset and research credits

---

## 🏆 Project Statistics

- **Total Lines of Code**: ~4,000+ (Python)
- **Functions/Methods**: 100+
- **Classes**: 6 main classes
- **Configuration Options**: 50+
- **Visualizations**: 15 types
- **Documentation**: Comprehensive README, docstrings, notebooks
- **Development Time**: Professional-grade implementation

---

**Built with best practices in ML engineering, explainable AI, and software architecture.**