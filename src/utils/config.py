"""
Configuration settings for ML Evaluation App
"""

import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"
DOCS_DIR = PROJECT_ROOT / "docs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
LOGS_DIR = PROJECT_ROOT / "logs"

# App configuration
APP_TITLE = "AI Explainer Pro - Model Insights & Evaluation"
APP_LAYOUT = "wide"
APP_VERSION = "1.0.0"

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100

# SHAP configuration
SHAP_SAMPLE_SIZE = 50
SHAP_BACKGROUND_SIZE = 100

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"

# Dataset configurations
DATASETS = {
    'iris': {
        'name': 'Iris Dataset',
        'description': 'Classic flower classification dataset',
        'type': 'multiclass',
        'features': 4,
        'classes': 3
    },
    'wine': {
        'name': 'Wine Dataset',
        'description': 'Wine origin classification dataset',
        'type': 'multiclass',
        'features': 13,
        'classes': 3
    },
    'breast_cancer': {
        'name': 'Breast Cancer Dataset',
        'description': 'Binary classification for cancer diagnosis',
        'type': 'binary',
        'features': 30,
        'classes': 2
    }
}

# Model configurations
MODELS = {
    'random_forest': {
        'name': 'Random Forest',
        'description': 'Ensemble of decision trees',
        'type': 'tree-based',
        'supports_feature_importance': True
    },
    'gradient_boosting': {
        'name': 'Gradient Boosting',
        'description': 'Boosted ensemble method',
        'type': 'tree-based',
        'supports_feature_importance': True
    },
    'logistic_regression': {
        'name': 'Logistic Regression',
        'description': 'Linear classification model',
        'type': 'linear',
        'supports_feature_importance': False
    },
    'svm': {
        'name': 'Support Vector Machine',
        'description': 'Support vector classification',
        'type': 'kernel',
        'supports_feature_importance': False
    }
}
