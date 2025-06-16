# AI Explainer Pro - Model Insights & Evaluation Suite

A comprehensive tool for model interpretation and explainability assessment with robust SHAP explanations, LIME analysis, and interactive visualizations.

## 🚀 Features

- **Multiple Explanation Methods**: SHAP, LIME, and Feature Importance
- **Robust SHAP Implementation**: Automatic fallbacks for all model types
- **Interactive UI**: User-friendly Streamlit interface
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Multiple Datasets**: Iris, Wine, and Breast Cancer datasets
- **Various Models**: Random Forest, Gradient Boosting, Logistic Regression, SVM
- **Real-time Predictions**: Interactive prediction with explanations

## 📁 Project Structure

```
MLEvaluation/
├── src/                          # Source code
│   ├── __init__.py
│   ├── app.py                    # Main Streamlit application
│   ├── components/               # UI components
│   │   ├── __init__.py
│   │   └── ui_components.py      # Streamlit UI components
│   └── utils/                    # Utility modules
│       ├── __init__.py
│       ├── config.py             # Configuration settings
│       ├── logging_config.py     # Logging configuration
│       ├── data_handler.py       # Data loading and processing
│       ├── model_handler.py      # Model creation and training
│       └── explanation_handler.py # SHAP, LIME, and explanations
├── tests/                        # Test files
│   ├── __init__.py
│   ├── test_functionality.py            # Original comprehensive tests
│   ├── test_functionality_improved.py   # Improved tests with fixes
│   ├── test_main_py_exact.py            # Targeted main.py logic tests
│   ├── test_corrected_shap.py           # SHAP value correctness tests
│   ├── test_improved_logging.py         # Logging improvement tests
│   └── test_*.py                        # Other test files
├── scripts/                      # Utility scripts
│   ├── analyze_shap_values.py           # SHAP analysis scripts
│   ├── debug_format_error.py            # Debug utilities
│   ├── final_validation.py              # Final validation script
│   └── *.py                             # Other utility scripts
├── docs/                         # Documentation
│   ├── README.md                        # This file
│   ├── COMPLETE_FIXES_SUMMARY.md        # Comprehensive fix documentation
│   ├── SHAP_FIXES_SUMMARY.md            # SHAP-specific fixes
│   └── *.md                             # Other documentation
├── logs/                         # Application logs
├── .venv/                        # Virtual environment
├── main.py                       # Legacy main file (for compatibility)
├── run_app.py                    # Application runner script
└── requirements.txt              # Python dependencies
```

## 🔧 Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd MLEvaluation
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### Running the Application

**Option 1: Using the run script (Recommended)**
```bash
python run_app.py
```

**Option 2: Direct Streamlit command**
```bash
streamlit run src/app.py
```

**Option 3: Legacy main file**
```bash
streamlit run main.py
```

### Using the Application

1. **Select Dataset**: Choose from Iris, Wine, or Breast Cancer datasets
2. **Select Model**: Choose from Random Forest, Gradient Boosting, Logistic Regression, or SVM
3. **View Performance**: See training and test accuracy metrics
4. **Make Predictions**: Adjust feature values and see real-time predictions
5. **Explain Predictions**: Choose from SHAP, LIME, or Feature Importance explanations

## 🧪 Testing

### Run All Tests
```bash
# Improved comprehensive test suite (Recommended)
python tests/test_functionality_improved.py

# Original test suite (for comparison)
python tests/test_functionality.py

# Targeted main.py logic tests
python tests/test_main_py_exact.py
```

### Run Validation Scripts
```bash
# Final validation of all fixes
python scripts/final_validation.py

# Analyze SHAP value correctness
python scripts/analyze_shap_values.py
```

## 🔍 Key Improvements

### SHAP Fixes
- **100% Success Rate**: All model-dataset combinations now work
- **Automatic Fallbacks**: PermutationExplainer fallback for unsupported cases
- **Correct Probability Explanations**: Fixed to use `predict_proba` instead of `predict`
- **Multi-class Support**: Proper handling of multi-class Gradient Boosting
- **Pipeline Support**: SVM and Logistic Regression pipelines now work

### Architecture Improvements
- **Modular Design**: Clean separation of concerns
- **Comprehensive Logging**: Multi-level logging with detailed context
- **Error Handling**: Robust error handling with user-friendly messages
- **Type Safety**: Proper data type handling for all operations
- **Configuration Management**: Centralized configuration

### User Experience
- **Clear Feedback**: Informative messages about explainer selection
- **Error Guidance**: Actionable error messages and suggestions
- **Interactive UI**: Intuitive interface with real-time updates
- **Performance Metrics**: Clear display of model performance

## 📊 Supported Combinations

| Dataset | Random Forest | Gradient Boosting | Logistic Regression | SVM |
|---------|---------------|-------------------|-------------------|-----|
| Iris (3-class) | ✅ TreeExplainer | ✅ PermutationExplainer* | ✅ General Explainer | ✅ PermutationExplainer* |
| Wine (3-class) | ✅ TreeExplainer | ✅ PermutationExplainer* | ✅ General Explainer | ✅ PermutationExplainer* |
| Breast Cancer (2-class) | ✅ TreeExplainer | ✅ TreeExplainer | ✅ General Explainer | ✅ PermutationExplainer* |

*Automatic fallback due to model limitations

## 🔧 Configuration

### Environment Variables
- `LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)

### Configuration Files
- `src/utils/config.py`: Main configuration settings
- `src/utils/logging_config.py`: Logging configuration

## 📝 Logging

The application uses comprehensive logging with multiple levels:

- **DEBUG**: Detailed technical information
- **INFO**: General application flow
- **WARNING**: Important notices
- **ERROR**: Error conditions

Log files are created in the `logs/` directory:
- `ml_evaluation_YYYYMMDD.log`: All log messages
- `ml_evaluation_errors_YYYYMMDD.log`: Error messages only

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure everything works
5. Submit a pull request

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

- **COMPLETE_FIXES_SUMMARY.md**: Comprehensive overview of all improvements
- **SHAP_FIXES_SUMMARY.md**: Detailed SHAP fix documentation
- **LOGGING_DOCUMENTATION.md**: Logging system documentation

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root directory
2. **Module Not Found**: Check that virtual environment is activated
3. **SHAP Errors**: The application now handles all SHAP errors automatically with fallbacks
4. **Memory Issues**: Reduce sample sizes in configuration if needed

### Getting Help

- Check the logs in `logs/` directory for detailed error information
- Run validation scripts to identify specific issues
- Review documentation in `docs/` directory

## 📈 Performance

- **SHAP Success Rate**: 100% (12/12 model-dataset combinations)
- **Test Coverage**: Comprehensive test suite with multiple validation approaches
- **Error Handling**: Robust fallback mechanisms for all scenarios

## 🏆 Validation Results

```
✅ ALL TESTS PASSED!
🎉 SHAP Success Rate: 12/12 (100%)
🎯 All models and explanation methods working correctly!
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with Streamlit for the interactive interface
- Uses SHAP and LIME for model explanations
- Scikit-learn for machine learning models
- Comprehensive testing ensures reliability

---

**AI Explainer Pro** - Making machine learning models interpretable and trustworthy! 🚀
