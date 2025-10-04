# Customer Churn Prediction System - Complete Index

Welcome to the Customer Churn Prediction System! This document provides a complete navigation guide to all project resources.

---

## 📚 Quick Navigation

### For Business Users
1. **[RESULTS_SUMMARY.md](RESULTS_SUMMARY.md)** - Executive summary with visualizations
2. **[README.md](README.md)** - Quick start guide with visual results
3. **Dashboard** - Run `streamlit run app.py` for interactive interface

### For Data Scientists
1. **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - Technical deep dive
2. **[notebooks/01_exploratory_data_analysis.ipynb](notebooks/01_exploratory_data_analysis.ipynb)** - EDA notebook
3. **[config/config.yaml](config/config.yaml)** - All configuration options

### For Developers
1. **[src/](src/)** - Source code modules
2. **[tests/test_pipeline.py](tests/test_pipeline.py)** - Unit tests
3. **[main.py](main.py)** - Training pipeline

---

## 📖 Documentation Files

### Primary Documentation

| File | Purpose | Audience | Size |
|------|---------|----------|------|
| [README.md](README.md) | Main documentation with visualizations | Everyone | 4.4 KB |
| [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) | Detailed results analysis | Business/Data Science | 13 KB |
| [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) | Technical architecture | Developers/DS | 12 KB |
| [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) | Dataset sources and credits | Researchers | 3.6 KB |
| [INDEX.md](INDEX.md) | This file - navigation guide | Everyone | - |

### Configuration Files

| File | Purpose | Format |
|------|---------|--------|
| [config/config.yaml](config/config.yaml) | System configuration | YAML |
| [requirements.txt](requirements.txt) | Python dependencies | Text |
| [.gitignore](.gitignore) | Git exclusions | Text |

---

## 🎨 Visualizations

### README Visualizations (docs/images/)

| File | Description | Size | Format |
|------|-------------|------|--------|
| `model_comparison_metrics.png` | All metrics bar chart | 176 KB | PNG |
| `performance_heatmap.png` | Metrics heatmap | 156 KB | PNG |
| `best_per_metric.png` | Winner per metric | 141 KB | PNG |
| `confusion_matrices_combined.png` | All confusion matrices | 147 KB | PNG |
| `roc_auc_ranking.png` | ROC-AUC comparison | 115 KB | PNG |
| `f1_score_breakdown.png` | Precision/Recall/F1 | 134 KB | PNG |
| `metrics_table.png` | Performance table | 123 KB | PNG |

**Total:** 7 visualizations, ~1 MB

### Pipeline Visualizations (data/plots/)

Generated during training pipeline:

| Category | Files | Count |
|----------|-------|-------|
| EDA Plots | `churn_distribution.png`, `numeric_distributions.png`, `categorical_distributions.png` | 3 |
| Confusion Matrices | `confusion_matrix_*.png` (per model) | 3 |
| ROC/PR Curves | `roc_curves.png`, `precision_recall_curves.png`, `calibration_curves.png` | 3 |
| Feature Importance | `feature_importance_*.png` (per model + SHAP) | 4 |
| Comparisons | `model_comparison.png` | 1 |
| SHAP | `shap_summary.png` | 1 |

**Total:** 15 visualizations, ~3.4 MB

---

## 🧠 Source Code

### Module Structure

```
src/
├── ingestion/          # Data loading
│   ├── __init__.py
│   └── data_loader.py  (300+ lines)
├── preprocessing/      # Data preprocessing
│   ├── __init__.py
│   └── preprocessor.py (500+ lines)
├── models/            # ML models
│   ├── __init__.py
│   └── model_trainer.py (450+ lines)
├── evaluation/        # Model evaluation
│   ├── __init__.py
│   └── evaluator.py (400+ lines)
├── explainability/    # SHAP & LIME
│   ├── __init__.py
│   └── explainer.py (500+ lines)
├── visualization/     # Plotting
│   ├── __init__.py
│   └── plotter.py (500+ lines)
└── utils/            # Utilities
    ├── __init__.py
    ├── config_loader.py (200+ lines)
    └── logger.py (70 lines)
```

**Total:** ~3,000 lines of production code

### Main Scripts

| File | Purpose | Lines | Description |
|------|---------|-------|-------------|
| [main.py](main.py) | Training pipeline | 445 | Complete ML pipeline orchestrator |
| [app.py](app.py) | Streamlit dashboard | 350+ | Interactive web application |
| [generate_readme_visuals.py](generate_readme_visuals.py) | Visualization generator | 325 | Creates README charts |

### Batch Scripts (Windows)

| File | Purpose |
|------|---------|
| `run_pipeline.bat` | Execute training pipeline |
| `run_dashboard.bat` | Launch Streamlit dashboard |

---

## 🗂️ Data Files

### Models (data/models/)

| File | Model Type | Size | Description |
|------|-----------|------|-------------|
| `logistic_regression.joblib` | Linear | 2.1 KB | Highest ROC-AUC (0.8471) |
| `random_forest.joblib` | Ensemble | 4.4 MB | Best F1 Score (0.6213) |
| `xgboost.joblib` | Gradient Boosting | 297 KB | Best Accuracy (79.03%) |
| `preprocessor.joblib` | Transformer | 6.1 KB | For inference |

**Total:** 4 files, 4.7 MB

### Results (data/results/)

| File | Format | Size | Content |
|------|--------|------|---------|
| `evaluation_report.txt` | Text | 2.1 KB | Detailed metrics report |
| `evaluation_results.json` | JSON | 1.6 KB | Machine-readable metrics |
| `summary.md` | Markdown | 1.6 KB | Auto-generated summary |

### Raw Data (data/raw/)

| File | Source | Records | Size |
|------|--------|---------|------|
| `Telco-Customer-Churn.csv` | IBM Cognos | 7,043 | ~1 MB |

---

## 📊 Results Quick Reference

### Model Performance

| Model | Accuracy | F1 | ROC-AUC | Best For |
|-------|----------|----|---------| ---------|
| **Logistic Regression** | 74.96% | 0.6197 | **0.8471** ⭐ | Catching all churners |
| **Random Forest** | 77.39% | **0.6213** ⭐ | 0.8389 | Balanced performance |
| **XGBoost** | **79.03%** ⭐ | 0.6070 | 0.8361 | Minimizing false alarms |

### Top Churn Factors

1. **Contract Type** (Month-to-month = high risk)
2. **Tenure** (< 6 months = critical)
3. **Monthly Charges** (Higher = more churn)
4. **Internet Service** (Fiber optic = different pattern)
5. **Payment Method** (Electronic check = higher risk)

---

## 🧪 Testing

### Test Files

| File | Coverage | Description |
|------|----------|-------------|
| [tests/test_pipeline.py](tests/test_pipeline.py) | Core modules | Unit tests for all components |

**Run tests:**
```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

---

## 🚀 Quick Start Commands

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
# Full pipeline
python main.py

# Skip download
python main.py --skip-download

# Windows
run_pipeline.bat
```

### Dashboard
```bash
# Launch app
streamlit run app.py

# Windows
run_dashboard.bat
```

### Testing
```bash
pytest tests/ -v
```

### Visualizations
```bash
python generate_readme_visuals.py
```

---

## 📋 File Checklist

### Documentation ✅
- [x] README.md (with visualizations)
- [x] RESULTS_SUMMARY.md
- [x] PROJECT_OVERVIEW.md
- [x] ACKNOWLEDGEMENTS.md
- [x] INDEX.md (this file)

### Source Code ✅
- [x] 6 modules (ingestion, preprocessing, models, evaluation, explainability, visualization)
- [x] main.py (pipeline)
- [x] app.py (dashboard)
- [x] tests/test_pipeline.py
- [x] generate_readme_visuals.py

### Configuration ✅
- [x] config/config.yaml
- [x] requirements.txt
- [x] .gitignore

### Data ✅
- [x] Raw dataset (7,043 records)
- [x] 4 trained models
- [x] Preprocessor
- [x] Evaluation results

### Visualizations ✅
- [x] 7 README charts
- [x] 15 pipeline plots
- [x] 22 total visualizations

### Scripts ✅
- [x] run_pipeline.bat
- [x] run_dashboard.bat

---

## 🎯 Recommended Reading Order

### For First-Time Users
1. Start with **[README.md](README.md)** - Overview and quick start
2. Review **[RESULTS_SUMMARY.md](RESULTS_SUMMARY.md)** - Detailed results
3. Explore **Streamlit Dashboard** - Interactive experience

### For Data Scientists
1. Read **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - Technical details
2. Review **[config/config.yaml](config/config.yaml)** - Configuration options
3. Explore **[notebooks/](notebooks/)** - EDA analysis
4. Study **[src/](src/)** - Implementation details

### For Developers
1. Review **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - Architecture
2. Examine **[main.py](main.py)** - Pipeline structure
3. Study **[src/](src/)** modules - Code organization
4. Check **[tests/](tests/)** - Testing approach

---

## 💡 Tips & Tricks

### Customizing Models
1. Edit `config/config.yaml`
2. Enable/disable models
3. Adjust hyperparameters
4. Run `python main.py`

### Adding Visualizations
1. Use `src/visualization/plotter.py`
2. Add new plotting functions
3. Call from `main.py`

### Extending Features
1. Modify `src/preprocessing/preprocessor.py`
2. Add feature engineering logic
3. Update config if needed

### Dashboard Customization
1. Edit `app.py`
2. Modify Streamlit components
3. Add new tabs or features

---

## 📊 Project Statistics

| Metric | Value |
|--------|-------|
| **Total Files** | 50+ |
| **Code Lines** | 4,000+ |
| **Documentation** | 5 files, 33 KB |
| **Visualizations** | 22 charts |
| **Models** | 3 trained |
| **Test Coverage** | Core functionality |
| **Dependencies** | 25+ packages |
| **Total Size** | ~10 MB |

---

## 🔗 External Resources

### Datasets
- [IBM Telco Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [IBM Cognos Samples](https://www.ibm.com/docs/en/cognos-analytics)

### Research Papers
- See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) for full list

### Libraries
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [SHAP](https://shap.readthedocs.io/)
- [Streamlit](https://streamlit.io/)

---

## 🆘 Troubleshooting

### Common Issues

**Import Errors:**
```bash
pip install -r requirements.txt
```

**Missing Data:**
```bash
python main.py  # Downloads automatically
```

**Dashboard Error:**
```bash
streamlit run app.py --server.port 8502  # Try different port
```

**Test Failures:**
```bash
pytest tests/ -v  # See detailed output
```

---

## 📞 Support

### Getting Help
1. Check documentation files
2. Review error logs in `logs/`
3. Run tests: `pytest tests/ -v`
4. Check configuration: `config/config.yaml`

### Reporting Issues
1. Check existing documentation
2. Review troubleshooting section
3. Gather error messages and logs
4. Include system information

---

## 🎓 Learning Resources

### Understanding the System
- **Business Impact**: See RESULTS_SUMMARY.md
- **Technical Details**: See PROJECT_OVERVIEW.md
- **Data Analysis**: See notebooks/
- **Code Examples**: See src/ modules

### ML Concepts
- **Classification**: Binary prediction (churn yes/no)
- **SHAP**: Feature importance explanation
- **SMOTE**: Handling imbalanced classes
- **ROC-AUC**: Model discrimination ability

---

## ✅ Project Completion Checklist

### Phase 1: Setup ✅
- [x] Project structure created
- [x] Dependencies installed
- [x] Configuration setup

### Phase 2: Data & EDA ✅
- [x] Dataset downloaded
- [x] EDA notebook created
- [x] Visualizations generated

### Phase 3: Pipeline ✅
- [x] Preprocessing implemented
- [x] Models trained
- [x] Evaluation completed

### Phase 4: Explainability ✅
- [x] SHAP integrated
- [x] LIME implemented
- [x] Feature importance analyzed

### Phase 5: Dashboard ✅
- [x] Streamlit app built
- [x] Single prediction
- [x] Batch processing
- [x] Visualizations integrated

### Phase 6: Documentation ✅
- [x] README with visualizations
- [x] Technical overview
- [x] Results summary
- [x] Navigation index

### Phase 7: Testing ✅
- [x] Unit tests written
- [x] Pipeline tested
- [x] Dashboard validated

---

## 🏆 Project Achievements

✅ **3 Production Models** - Trained and evaluated
✅ **22 Visualizations** - Comprehensive charts
✅ **Full Documentation** - 5 detailed files
✅ **Interactive Dashboard** - Real-time predictions
✅ **Explainable AI** - SHAP & LIME integrated
✅ **Production Ready** - Config-driven, logged, tested

---

**🚀 System Status: COMPLETE & READY FOR USE**

*Last Updated: October 3, 2025*
*Version: 1.0.0*
