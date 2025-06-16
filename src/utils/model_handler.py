"""
Model handling utilities for ML Evaluation App
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Dict, Any, Tuple
from .config import MODELS, RANDOM_STATE, N_ESTIMATORS
from .logging_config import get_logger

logger = get_logger('model')


def create_model(model_name: str):
    """
    Create a model instance based on the model name
    
    Args:
        model_name: Name of the model to create
        
    Returns:
        Configured model instance
    """
    logger.info(f"Creating model: {model_name}")
    
    if model_name == 'random_forest':
        return RandomForestClassifier(
            n_estimators=N_ESTIMATORS, 
            random_state=RANDOM_STATE
        )
    elif model_name == 'gradient_boosting':
        return GradientBoostingClassifier(
            n_estimators=N_ESTIMATORS, 
            random_state=RANDOM_STATE
        )
    elif model_name == 'logistic_regression':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                random_state=RANDOM_STATE, 
                max_iter=1000
            ))
        ])
    elif model_name == 'svm':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(
                probability=True, 
                random_state=RANDOM_STATE
            ))
        ])
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def train_model(model, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    """
    Train a model and return training information
    
    Args:
        model: Model instance to train
        X_train: Training features
        y_train: Training targets
        
    Returns:
        Dictionary with training information
    """
    model_name = get_model_name(model)
    logger.info(f"Training {model_name} model")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Calculate training score
    train_score = model.score(X_train, y_train)
    
    logger.info(f"{model_name} training completed - Score: {train_score:.4f}")
    
    return {
        'model': model,
        'train_score': train_score,
        'model_name': model_name
    }


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluate a trained model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary with evaluation metrics
    """
    model_name = get_model_name(model)
    logger.info(f"Evaluating {model_name} model")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Metrics
    test_score = accuracy_score(y_test, y_pred)
    
    logger.info(f"{model_name} evaluation completed - Accuracy: {test_score:.4f}")
    
    return {
        'test_score': test_score,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }


def get_model_name(model) -> str:
    """Get the name of a model"""
    if isinstance(model, RandomForestClassifier):
        return 'random_forest'
    elif isinstance(model, GradientBoostingClassifier):
        return 'gradient_boosting'
    elif isinstance(model, Pipeline):
        classifier = model.named_steps.get('classifier')
        if isinstance(classifier, LogisticRegression):
            return 'logistic_regression'
        elif isinstance(classifier, SVC):
            return 'svm'
    return 'unknown'


def is_pipeline_model(model) -> bool:
    """Check if model is a pipeline"""
    return hasattr(model, 'named_steps')


def is_tree_based_model(model) -> bool:
    """Check if model is tree-based"""
    return isinstance(model, (RandomForestClassifier, GradientBoostingClassifier))


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Get information about all available models"""
    return MODELS.copy()


def validate_model_name(model_name: str) -> bool:
    """Check if model name is valid"""
    return model_name in MODELS
