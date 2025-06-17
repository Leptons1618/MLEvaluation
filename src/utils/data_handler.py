"""
Data handling utilities for ML Evaluation App
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
from utils.config import DATASETS, RANDOM_STATE, TEST_SIZE
from utils.logging_config import get_logger

logger = get_logger('data')


def load_dataset(dataset_name: str) -> Dict[str, Any]:
    """
    Load and prepare a dataset for ML evaluation
    
    Args:
        dataset_name: Name of the dataset ('iris', 'wine', 'breast_cancer')
        
    Returns:
        Dictionary containing dataset information and splits
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Load dataset
    if dataset_name == 'iris':
        dataset = load_iris()
    elif dataset_name == 'wine':
        dataset = load_wine()
    elif dataset_name == 'breast_cancer':
        dataset = load_breast_cancer()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create DataFrame
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    logger.info(f"Dataset {dataset_name} loaded: {len(X)} samples, {len(X.columns)} features")
    logger.debug(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    return {
        'name': dataset_name,
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': dataset.feature_names,
        'target_names': dataset.target_names,
        'config': DATASETS[dataset_name]
    }


def ensure_dataframe(X: Any, feature_names: list) -> pd.DataFrame:
    """
    Ensure input is a DataFrame with proper feature names
    
    Args:
        X: Input data (DataFrame, numpy array, etc.)
        feature_names: List of feature names
        
    Returns:
        DataFrame with proper column names
    """
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X, columns=feature_names)


def get_available_datasets() -> Dict[str, Dict[str, Any]]:
    """Get information about all available datasets"""
    return DATASETS.copy()


def validate_dataset(dataset_name: str) -> bool:
    """Check if dataset name is valid"""
    return dataset_name in DATASETS
