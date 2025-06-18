"""
Data preparation utilities for uploaded datasets
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, Optional, Tuple, List
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from utils.logging_config import get_logger

logger = get_logger('data_preparation')


def load_uploaded_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load uploaded file and return DataFrame
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Tuple of (DataFrame, error_message)
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            logger.info(f"Successfully loaded CSV file: {uploaded_file.name}")
            return df, None
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
            logger.info(f"Successfully loaded Excel file: {uploaded_file.name}")
            return df, None
        else:
            error_msg = "Unsupported file format. Please upload CSV or Excel files."
            logger.error(f"Unsupported file format: {uploaded_file.name}")
            return None, error_msg
    except Exception as e:
        error_msg = f"Error loading file: {str(e)}"
        logger.error(f"Error loading {uploaded_file.name}: {str(e)}")
        return None, error_msg


def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze uploaded dataset and return summary statistics
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing analysis results
    """
    analysis = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
        'datetime_columns': list(df.select_dtypes(include=['datetime64']).columns)
    }
    
    # Basic statistics for numeric columns
    if analysis['numeric_columns']:
        analysis['numeric_stats'] = df[analysis['numeric_columns']].describe().to_dict()
    
    # Value counts for categorical columns (top 10)
    categorical_info = {}
    for col in analysis['categorical_columns']:
        value_counts = df[col].value_counts().head(10)
        categorical_info[col] = {
            'unique_count': df[col].nunique(),
            'top_values': value_counts.to_dict()
        }
    analysis['categorical_info'] = categorical_info
    
    logger.info(f"Dataset analysis completed: {analysis['shape'][0]} rows, {analysis['shape'][1]} columns")
    return analysis


def paginate_dataframe(df: pd.DataFrame, page_size: int = 50, page_number: int = 1) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Paginate DataFrame for display
    
    Args:
        df: Input DataFrame
        page_size: Number of rows per page
        page_number: Current page number (1-indexed)
        
    Returns:
        Tuple of (paginated_df, pagination_info)
    """
    total_rows = len(df)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)  # Ceiling division
    
    # Ensure page_number is within valid range
    page_number = max(1, min(page_number, total_pages))
    
    start_idx = (page_number - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    
    paginated_df = df.iloc[start_idx:end_idx]
    
    pagination_info = {
        'current_page': page_number,
        'total_pages': total_pages,
        'page_size': page_size,
        'total_rows': total_rows,
        'start_row': start_idx + 1,
        'end_row': end_idx,
        'showing_rows': len(paginated_df)
    }
    
    return paginated_df, pagination_info


def detect_target_column(df: pd.DataFrame, analysis: Dict[str, Any]) -> List[str]:
    """
    Suggest potential target columns based on data analysis
    
    Args:
        df: Input DataFrame
        analysis: Dataset analysis results
        
    Returns:
        List of suggested target column names
    """
    suggestions = []
    
    # Common target column names
    common_targets = ['target', 'label', 'class', 'y', 'output', 'result', 'prediction']
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Check for common target names
        if any(target in col_lower for target in common_targets):
            suggestions.append(col)
        
        # Check for categorical columns with reasonable number of unique values (2-20)
        elif col in analysis['categorical_columns']:
            unique_count = analysis['categorical_info'][col]['unique_count']
            if 2 <= unique_count <= 20:
                suggestions.append(col)
        
        # Check for numeric columns that might be classifications (small range of integers)
        elif col in analysis['numeric_columns']:
            if df[col].dtype in ['int64', 'int32']:
                unique_vals = df[col].unique()
                if len(unique_vals) <= 20 and all(isinstance(x, (int, np.integer)) for x in unique_vals):
                    suggestions.append(col)
    
    return suggestions


def prepare_dataset_for_ml(df: pd.DataFrame, target_column: str, 
                          test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """
    Prepare dataset for machine learning with improved error handling
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        test_size: Proportion of test set
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary containing prepared dataset components
    """
    try:
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical features
        categorical_columns = list(X.select_dtypes(include=['object', 'category']).columns)
        label_encoders = {}
        
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Handle target encoding if categorical
        target_encoder = None
        if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))
          # Check class distribution for stratification
        class_counts = pd.Series(y).value_counts()
        min_class_size = class_counts.min()
        num_classes = len(class_counts)
        total_samples = len(y)
        
        # Calculate minimum test size needed for stratification
        min_test_samples = max(num_classes, int(total_samples * 0.1))  # At least 1 per class or 10%
        actual_test_samples = int(total_samples * test_size)
        
        # Use stratification only if conditions are met
        use_stratify = (min_class_size >= 2 and actual_test_samples >= num_classes)
        
        if not use_stratify:
            if min_class_size < 2:
                logger.warning(f"Disabling stratification due to small class sizes (min: {min_class_size}). "
                             f"Consider removing classes with few samples or using different split strategy.")
            elif actual_test_samples < num_classes:
                logger.warning(f"Disabling stratification: test size ({actual_test_samples}) smaller than number of classes ({num_classes}). "
                             f"Consider using larger test_size or smaller dataset.")
        
        # Adjust test_size if necessary for very small datasets
        if actual_test_samples < max(1, num_classes // 2):
            adjusted_test_size = max(0.3, num_classes / total_samples)
            logger.warning(f"Adjusting test_size from {test_size:.2f} to {adjusted_test_size:.2f} for small dataset")
            test_size = adjusted_test_size
        
        # Split the data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state, 
                stratify=y if use_stratify else None
            )
        except ValueError as e:
            if "least populated class" in str(e) or "test_size" in str(e):
                # Fallback to non-stratified split with adjusted test size
                logger.warning(f"Split failed ({str(e)}), using non-stratified random split")
                # Use a simple 70/30 split for very small datasets
                fallback_test_size = min(0.3, max(0.1, 2/total_samples))
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=fallback_test_size, random_state=random_state, stratify=None
                )
                use_stratify = False
            else:
                raise e
        
        # Get feature names and target names
        feature_names = list(X.columns)
        
        if target_encoder:
            target_names = list(target_encoder.classes_)
        else:
            target_names = [str(i) for i in sorted(y.unique())]
        
        prepared_data = {
            'X': X,
            'y': y,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'target_names': target_names,
            'label_encoders': label_encoders,
            'target_encoder': target_encoder,
            'stratified': use_stratify,
            'config': {
                'name': 'User Uploaded Dataset',
                'type': 'classification',
                'features': len(feature_names),
                'classes': len(target_names),
                'description': f'User uploaded dataset with {len(feature_names)} features and {len(target_names)} classes'
            }
        }
        
        logger.info(f"Dataset prepared for ML: {X.shape[0]} samples, {X.shape[1]} features, {len(target_names)} classes, stratified: {use_stratify}")
        return prepared_data
        
    except Exception as e:
        logger.error(f"Error preparing dataset for ML: {str(e)}")
        raise e


def get_data_quality_report(df: pd.DataFrame, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate data quality report
    
    Args:
        df: Input DataFrame
        analysis: Dataset analysis results
        
    Returns:
        Dictionary containing data quality metrics
    """
    quality_report = {
        'completeness': {},
        'consistency': {},
        'validity': {},
        'recommendations': []
    }
    
    # Completeness check
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    quality_report['completeness']['missing_percentage'] = (missing_cells / total_cells) * 100
    quality_report['completeness']['complete_rows'] = df.dropna().shape[0]
    quality_report['completeness']['complete_percentage'] = (quality_report['completeness']['complete_rows'] / df.shape[0]) * 100
    
    # Consistency checks
    quality_report['consistency']['duplicates'] = analysis['duplicates']
    quality_report['consistency']['duplicate_percentage'] = (analysis['duplicates'] / df.shape[0]) * 100
    
    # Recommendations
    if quality_report['completeness']['missing_percentage'] > 10:
        quality_report['recommendations'].append("Consider handling missing values (>10% missing)")
    
    if quality_report['consistency']['duplicate_percentage'] > 5:
        quality_report['recommendations'].append("Consider removing duplicate rows (>5% duplicates)")
    
    if len(analysis['categorical_columns']) > 0:
        quality_report['recommendations'].append("Categorical columns detected - may need encoding")
    
    return quality_report
