"""
Comprehensive Arrow compatibility utility functions
"""

import pandas as pd
import numpy as np
from typing import Any, Union, List, Dict
import warnings
warnings.filterwarnings('ignore')


def make_dataframe_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a DataFrame Arrow-compatible by fixing common serialization issues
    
    Args:
        df: Input DataFrame
        
    Returns:
        Arrow-compatible DataFrame
    """
    df_copy = df.copy()
    
    for col in df_copy.columns:
        df_copy[col] = make_series_arrow_compatible(df_copy[col])
    
    return df_copy


def make_series_arrow_compatible(series: pd.Series) -> pd.Series:
    """
    Make a Series Arrow-compatible by handling mixed types and problematic values
    
    Args:
        series: Input Series
        
    Returns:
        Arrow-compatible Series
    """
    if series.empty:
        return series
    
    # Handle mixed types - convert to string if necessary
    if has_mixed_types(series):
        return series.astype(str)
    
    # Handle numeric types with NaN
    if series.dtype in ['int64', 'float64', 'Int64', 'Float64']:
        # Replace inf/-inf with NaN
        series = series.replace([np.inf, -np.inf], np.nan)
        
        # For integer columns with NaN, convert to nullable Int64
        if series.dtype == 'int64' and series.isnull().any():
            return series.astype('Int64')
    
    # Handle object columns that might have mixed types
    elif series.dtype == 'object':
        # Try to infer better type
        if is_numeric_series(series):
            try:
                # Try to convert to numeric
                numeric_series = pd.to_numeric(series, errors='coerce')
                if numeric_series.notna().any():
                    return numeric_series.astype('Float64')
            except:
                pass
        
        # Convert to string to ensure compatibility
        return series.astype(str)
    
    return series


def has_mixed_types(series: pd.Series) -> bool:
    """
    Check if a Series has mixed data types
    
    Args:
        series: Input Series
        
    Returns:
        True if series has mixed types
    """
    if series.empty:
        return False
    
    # Get non-null values
    non_null_values = series.dropna()
    if len(non_null_values) == 0:
        return False
    
    # Check if all values are of the same type
    first_type = type(non_null_values.iloc[0])
    return not all(isinstance(val, first_type) for val in non_null_values)


def is_numeric_series(series: pd.Series) -> bool:
    """
    Check if a Series can be converted to numeric
    
    Args:
        series: Input Series
        
    Returns:
        True if series can be converted to numeric
    """
    try:
        pd.to_numeric(series, errors='raise')
        return True
    except (ValueError, TypeError):
        return False


def safe_dataframe_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame for safe display in Streamlit (Arrow-compatible)
    
    Args:
        df: Input DataFrame
        
    Returns:
        Display-ready DataFrame
    """
    display_df = df.copy()
    
    # Make Arrow-compatible
    display_df = make_dataframe_arrow_compatible(display_df)
    
    # Limit precision for float columns
    for col in display_df.select_dtypes(include=['float64', 'Float64']).columns:
        display_df[col] = display_df[col].round(6)
    
    # Truncate long strings
    for col in display_df.select_dtypes(include=['object']).columns:
        display_df[col] = display_df[col].astype(str).str[:100]
    
    return display_df


def create_arrow_safe_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create an Arrow-safe summary DataFrame with statistics
    
    Args:
        df: Input DataFrame
        
    Returns:
        Arrow-safe summary DataFrame
    """
    summary_data = []
    
    for col in df.columns:
        col_data = {
            'Column': str(col),
            'Type': str(df[col].dtype),
            'Non-Null Count': int(df[col].count()),
            'Null Count': int(df[col].isnull().sum()),
            'Unique Values': int(df[col].nunique()),
        }
        
        # Add statistics for numeric columns
        if df[col].dtype in ['int64', 'float64', 'Int64', 'Float64']:
            try:
                col_data.update({
                    'Mean': float(safe_mean(df[col])),
                    'Median': float(safe_median(df[col])),
                    'Mode': float(safe_mode(df[col])) if isinstance(safe_mode(df[col]), (int, float)) else 0.0,
                    'Min': float(df[col].min()) if df[col].notna().any() else 0.0,
                    'Max': float(df[col].max()) if df[col].notna().any() else 0.0
                })
            except:
                col_data.update({
                    'Mean': 0.0,
                    'Median': 0.0,
                    'Mode': 0.0,
                    'Min': 0.0,
                    'Max': 0.0
                })
        else:
            # For non-numeric columns
            try:
                most_common = str(safe_mode(df[col]))
            except:
                most_common = 'N/A'
            
            col_data.update({
                'Mean': 0.0,
                'Median': 0.0,
                'Mode': most_common,
                'Min': 0.0,
                'Max': 0.0
            })
        
        summary_data.append(col_data)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Ensure all columns are Arrow-compatible
    for col in summary_df.columns:
        if summary_df[col].dtype == 'object':
            summary_df[col] = summary_df[col].astype(str)
    
    return summary_df


# Import the safe functions from the main module
try:
    from components.enhanced_ui_components import safe_mean, safe_median, safe_mode
except ImportError:
    # Fallback definitions if not available
    def safe_mean(series: pd.Series) -> float:
        try:
            clean_values = series.dropna().values
            if len(clean_values) == 0:
                return 0.0
            return float(np.mean(clean_values))
        except:
            return 0.0
    
    def safe_median(series: pd.Series) -> float:
        try:
            clean_values = series.dropna().values
            if len(clean_values) == 0:
                return 0.0
            return float(np.median(clean_values))
        except:
            return 0.0
    
    def safe_mode(series: pd.Series):
        try:
            if series.dtype in ['int64', 'float64', 'Int64', 'Float64']:
                clean_values = series.dropna().values
                if len(clean_values) == 0:
                    return 0.0
                mode_series = pd.Series(clean_values).mode()
                return float(mode_series.iloc[0]) if len(mode_series) > 0 else 0.0
            else:
                mode_series = series.mode()
                return mode_series.iloc[0] if len(mode_series) > 0 else 'Unknown'
        except:
            return 'Unknown' if series.dtype == 'object' else 0.0
