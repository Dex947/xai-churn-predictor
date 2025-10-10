# Changelog

All notable changes to the XAI Churn Predictor project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2025-10-03

### Added
- **Core ML Pipeline**: Complete end-to-end churn prediction system
  - Data ingestion module with automatic dataset download
  - Preprocessing pipeline with SMOTE balancing
  - Three ML models: Logistic Regression, Random Forest, XGBoost
  - Comprehensive evaluation with 5 metrics (accuracy, precision, recall, F1, ROC-AUC)
  
- **Explainable AI (XAI)**:
  - SHAP integration for global and local explanations
  - LIME implementation for instance-level explanations
  - Feature importance visualization from multiple sources
  
- **Interactive Dashboard**:
  - Streamlit web application with 3 tabs
  - Single customer prediction with real-time SHAP explanations
  - Batch CSV upload and processing
  - Model performance visualization
  
- **Visualization Suite**:
  - 15+ plot types for EDA and model evaluation
  - Confusion matrices, ROC curves, calibration plots
  - Feature importance and SHAP summary plots
  
- **Documentation**:
  - Comprehensive README with visual results
  - PROJECT_OVERVIEW with technical architecture
  - RESULTS_SUMMARY with detailed analysis
  - INDEX for navigation
  - ACKNOWLEDGEMENTS for attribution
  
- **Testing**:
  - Unit tests for all core modules (pytest)
  - Test coverage for data loading, preprocessing, training, evaluation
  
- **Configuration**:
  - YAML-based configuration system
  - All parameters externalized (no hardcoding)
  - Modular and extensible design

### Technical Details
- **Dataset**: IBM Telco Customer Churn (7,043 records)
- **Best Model**: Random Forest (F1: 0.6213)
- **Best ROC-AUC**: Logistic Regression (0.8471)
- **Features**: 30 engineered features from 20 original
- **Class Balancing**: SMOTE oversampling on training set

### Performance
- Training time: < 5 seconds (all 3 models)
- Inference time: < 50ms per prediction
- SHAP computation: ~3 seconds for 100 samples

---

## [Unreleased]

### Planned Enhancements
- Hyperparameter tuning with GridSearchCV
- Feature selection automation
- Model ensemble (stacking/blending)
- REST API endpoint (FastAPI)
- Deep learning models (LSTM, Transformers)
- Automated retraining pipeline

---

## Audit Log

### [2025-10-10T15:16:10+01:00] - Code Quality Audit
**Auditor**: AI Systems Engineer  
**Scope**: Comprehensive codebase review per Global rules

#### Issues Identified & Fixed
1. ✅ Missing CHANGELOG.md - **FIXED** (created with full version history)
2. ✅ Missing memory.json - **FIXED** (created with project context and decisions)
3. ✅ Unused import in config_loader.py - **FIXED** (removed `import os`)
4. ✅ Hard-coded values in app.py - **FIXED** (moved to constants.py)
5. ✅ Inconsistent error handling in app.py - **FIXED** (specific exception types)
6. ✅ Magic numbers in plotter.py - **FIXED** (all replaced with constants)
7. ✅ Missing return type hint in logger.py - **FIXED** (added `-> logger`)
8. ℹ️ Long function in main.py - **DOCUMENTED** (acceptable for pipeline orchestrator)

#### Files Modified
- `src/utils/config_loader.py` - Removed unused import
- `src/utils/logger.py` - Added return type hint
- `src/utils/constants.py` - **NEW FILE** - Centralized all magic numbers
- `src/utils/__init__.py` - Exported constants module
- `app.py` - Replaced hard-coded values with constants, improved error handling
- `src/visualization/plotter.py` - Replaced all magic numbers with constants
- `CHANGELOG.md` - **NEW FILE** - Version tracking
- `memory.json` - **NEW FILE** - Project context storage

#### Improvements Applied
- **Code Quality**: Eliminated all code smells (unused imports, magic numbers)
- **Error Handling**: Specific exception types for better debugging
- **Maintainability**: Centralized configuration in constants.py
- **Documentation**: Created comprehensive CHANGELOG and memory files
- **Type Safety**: Added missing type hints

---

**Maintenance Notes**:
- Update this file with every code change
- Include timestamp, rationale, and file paths
- Reference issue numbers or PR links when applicable
