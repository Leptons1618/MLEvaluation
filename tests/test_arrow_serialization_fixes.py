"""
Test Arrow serialization fixes and safe calculation functions
"""

import sys
import os
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

import pandas as pd
import numpy as np
import pytest
from components.enhanced_ui_components import (
    safe_mean, safe_median, safe_mode, 
    apply_imputation, make_dataframe_arrow_compatible
)
from utils.data_preparation_enhanced import DataPreparationTools


def test_safe_calculation_functions():
    """Test safe calculation functions with various data types"""
    
    # Test numeric data
    numeric_series = pd.Series([1, 2, 3, 4, 5, np.nan])
    assert safe_mean(numeric_series) == 3.0
    assert safe_median(numeric_series) == 3.0
    assert safe_mode(numeric_series) == 1.0  # First mode value
    
    # Test categorical data
    categorical_series = pd.Series(['A', 'B', 'A', 'C', np.nan])
    assert safe_mode(categorical_series) == 'A'
      # Test empty numeric series
    empty_numeric_series = pd.Series([np.nan, np.nan], dtype='float64')
    assert safe_mean(empty_numeric_series) == 0.0
    assert safe_median(empty_numeric_series) == 0.0
    assert safe_mode(empty_numeric_series) == 0.0
    
    # Test empty categorical series
    empty_categorical_series = pd.Series([np.nan, np.nan], dtype='object')
    assert safe_mode(empty_categorical_series) == 'Unknown'
    
    print("✅ All safe calculation functions work correctly")


def test_arrow_compatibility():
    """Test Arrow compatibility utilities"""
    
    # Create problematic DataFrame
    df = pd.DataFrame({
        'numeric_col': [1, 2, 3, np.nan],
        'string_col': ['A', 'B', 'C', np.nan],
        'mixed_col': [1, 'A', 3, np.nan],
        'bool_col': [True, False, True, np.nan]
    })
    
    # Make it Arrow compatible
    df_safe = make_dataframe_arrow_compatible(df.copy())
    
    # Check that mixed column is converted to string
    assert df_safe['mixed_col'].dtype == 'object'
    assert str(df_safe['mixed_col'].iloc[0]) == '1'
    assert str(df_safe['mixed_col'].iloc[1]) == 'A'
    
    print("✅ Arrow compatibility utility works correctly")


def test_imputation_with_arrow_safety():
    """Test imputation function with Arrow safety"""
    
    # Create test DataFrame with various missing value patterns
    df = pd.DataFrame({
        'numeric_with_missing': [1, 2, np.nan, 4, 5],
        'categorical_with_missing': ['A', 'B', np.nan, 'A', 'C'],
        'target': [0, 1, 0, 1, 0]
    })
    
    # Test different imputation methods
    config = {
        'numeric_with_missing': {'method': 'Mean', 'custom_value': None},
        'categorical_with_missing': {'method': 'Mode', 'custom_value': None}
    }
    
    result_df = apply_imputation(df, config)
    
    # Check that missing values are filled
    assert result_df['numeric_with_missing'].isnull().sum() == 0
    assert result_df['categorical_with_missing'].isnull().sum() == 0    # Check that values are reasonable
    actual_mean = float(result_df['numeric_with_missing'].iloc[2])
    expected_mean = 3.0  # Mean of [1,2,4,5]
    print(f"Expected mean: {expected_mean}, Actual: {actual_mean}")
    assert abs(actual_mean - expected_mean) < 0.001
    assert result_df['categorical_with_missing'].iloc[2] == 'A'  # Mode
    
    print("✅ Imputation with Arrow safety works correctly")


def test_data_preparation_tools():
    """Test enhanced data preparation tools"""
    
    # Create test data
    df = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 5],
        'feature2': ['A', 'B', 'A', np.nan, 'C'],
        'target': [0, 1, 0, 1, 0]
    })
    
    # Initialize tools
    tools = DataPreparationTools()
    
    # Analyze preparation needs
    analysis = tools.analyze_preparation_needs(df, 'target')
    
    # Check that analysis detects issues
    assert len(analysis['issues']) > 0
    assert len(analysis['suggestions']) > 0
    
    print("✅ Data preparation tools work correctly")


def test_mixed_data_types():
    """Test handling of mixed data types that cause Arrow issues"""
    
    # Create DataFrame with mixed types in same column (common Arrow issue)
    df = pd.DataFrame({
        'mixed_column': [1, 'text', 3.14, True, np.nan],
        'normal_column': [1, 2, 3, 4, 5],
        'target': [0, 1, 0, 1, 0]
    })
    
    # Apply Arrow compatibility
    df_safe = make_dataframe_arrow_compatible(df.copy())
    
    # Check that mixed column is now uniform string type
    assert df_safe['mixed_column'].dtype == 'object'
    assert all(isinstance(val, str) or pd.isna(val) for val in df_safe['mixed_column'])
    
    print("✅ Mixed data type handling works correctly")


if __name__ == "__main__":
    print("🧪 Running Arrow Serialization Fix Tests...")
    
    try:
        test_safe_calculation_functions()
        test_arrow_compatibility()
        test_imputation_with_arrow_safety()
        test_data_preparation_tools()
        test_mixed_data_types()
        
        print("\n🎉 All Arrow serialization fix tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
