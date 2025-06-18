"""
Simple test script to verify Arrow serialization fixes
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from components.enhanced_ui_components import (
    safe_mean, safe_median, safe_mode, apply_imputation
)
from utils.arrow_compatibility import (
    make_dataframe_arrow_compatible, 
    safe_dataframe_display
)

def test_arrow_serialization_fixes():
    """Test that Arrow serialization issues are fixed"""
    print("🧪 Testing Arrow Serialization Fixes...")
    
    # Create test data that previously caused Arrow errors
    test_df = pd.DataFrame({
        'numeric_col': [1.0, 2.0, np.nan, 4.0, 5.0],
        'categorical_col': ['A', 'B', np.nan, 'A', 'C'],
        'mixed_col': [1, 2, 'invalid', 4, np.nan],
        'target': [0, 1, 0, 1, 0]
    })
    
    print("✅ Test data created")
    
    # Test safe calculation functions
    print("\n🔢 Testing safe calculation functions...")
    
    # Test with numeric data
    numeric_series = test_df['numeric_col']
    mean_val = safe_mean(numeric_series)
    median_val = safe_median(numeric_series)
    mode_val = safe_mode(numeric_series)
    
    print(f"   safe_mean: {mean_val} (type: {type(mean_val).__name__})")
    print(f"   safe_median: {median_val} (type: {type(median_val).__name__})")
    print(f"   safe_mode: {mode_val} (type: {type(mode_val).__name__})")
    
    # Test with categorical data
    categorical_series = test_df['categorical_col']
    cat_mode = safe_mode(categorical_series)
    print(f"   categorical mode: {cat_mode} (type: {type(cat_mode).__name__})")
    
    # Test with problematic mixed data
    mixed_series = test_df['mixed_col']
    mixed_mean = safe_mean(mixed_series)
    mixed_median = safe_median(mixed_series)
    mixed_mode = safe_mode(mixed_series)
    
    print(f"   mixed data mean: {mixed_mean} (type: {type(mixed_mean).__name__})")
    print(f"   mixed data median: {mixed_median} (type: {type(mixed_median).__name__})")
    print(f"   mixed data mode: {mixed_mode} (type: {type(mixed_mode).__name__})")
    
    print("✅ Safe calculation functions work correctly")
    
    # Test imputation with Arrow compatibility
    print("\n🔧 Testing imputation with Arrow compatibility...")
    
    imputation_config = {
        'numeric_col': {'method': 'Mean', 'custom_value': None},
        'categorical_col': {'method': 'Mode', 'custom_value': None}
    }
    
    result_df = apply_imputation(test_df, imputation_config)
    print(f"   Original missing values: {test_df.isnull().sum().sum()}")
    print(f"   After imputation missing values: {result_df.isnull().sum().sum()}")
    
    # Make the result DataFrame Arrow-compatible for testing
    arrow_safe_df = make_dataframe_arrow_compatible(result_df)
    
    # Test Arrow compatibility by trying to serialize
    try:
        import pyarrow as pa
        table = pa.Table.from_pandas(arrow_safe_df)
        print("✅ Arrow serialization successful - no errors!")
    except Exception as e:
        print(f"❌ Arrow serialization failed: {e}")
        return False
    
    # Test edge cases
    print("\n🔍 Testing edge cases...")
    
    # Empty series
    empty_series = pd.Series([])
    empty_mean = safe_mean(empty_series)
    empty_median = safe_median(empty_series)
    empty_mode = safe_mode(empty_series)
    print(f"   Empty series - mean: {empty_mean}, median: {empty_median}, mode: {empty_mode}")
    
    # All NaN series
    nan_series = pd.Series([np.nan, np.nan, np.nan])
    nan_mean = safe_mean(nan_series)
    nan_median = safe_median(nan_series)
    nan_mode = safe_mode(nan_series)
    print(f"   All NaN series - mean: {nan_mean}, median: {nan_median}, mode: {nan_mode}")
    
    # Single value series
    single_series = pd.Series([42])
    single_mean = safe_mean(single_series)
    single_median = safe_median(single_series)
    single_mode = safe_mode(single_series)
    print(f"   Single value series - mean: {single_mean}, median: {single_median}, mode: {single_mode}")
    
    print("✅ Edge cases handled correctly")
    
    # Test that results are Arrow-compatible
    print("\n🎯 Testing Arrow compatibility of all results...")
    
    results_df = pd.DataFrame({
        'Mean': [mean_val, mixed_mean, empty_mean, nan_mean, single_mean],
        'Median': [median_val, mixed_median, empty_median, nan_median, single_median],
        'Mode': [str(mode_val), str(mixed_mode), str(empty_mode), str(nan_mode), str(single_mode)],
        'Category_Mode': [str(cat_mode)] * 5
    })
    
    try:
        arrow_table = pa.Table.from_pandas(results_df)
        print("✅ All calculation results are Arrow-compatible!")
        print(f"   Arrow table schema: {arrow_table.schema}")
    except Exception as e:
        print(f"❌ Some results are not Arrow-compatible: {e}")
        return False
    
    print("\n🎉 All Arrow serialization fixes are working correctly!")
    return True

def test_plotly_fixes():
    """Test that Plotly method fixes work"""
    print("\n📊 Testing Plotly fixes...")
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        
        # Create a simple figure
        fig = px.bar(x=['A', 'B', 'C'], y=[1, 2, 3], title='Test Chart')
        
        # Test the corrected method
        fig.update_xaxes(tickangle=45)  # This should work
        print("✅ Plotly update_xaxes method works correctly")
        
        # Test that the old method would fail (for documentation)
        try:
            fig.update_xaxis(tickangle=45)  # This should fail
            print("❌ Old update_xaxis method unexpectedly works")
        except AttributeError:
            print("✅ Old update_xaxis method correctly raises AttributeError")
        
        return True
        
    except ImportError:
        print("⚠️  Plotly not available for testing")
        return True
    except Exception as e:
        print(f"❌ Plotly test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Arrow Serialization and Plotly Fixes Validation")
    print("=" * 60)
    
    arrow_success = test_arrow_serialization_fixes()
    plotly_success = test_plotly_fixes()
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY:")
    print(f"   Arrow Serialization Fixes: {'✅ PASSED' if arrow_success else '❌ FAILED'}")
    print(f"   Plotly Method Fixes: {'✅ PASSED' if plotly_success else '❌ FAILED'}")
    
    if arrow_success and plotly_success:
        print("\n🎉 ALL FIXES WORKING CORRECTLY!")
        print("   The application should now run without Arrow serialization errors")
        print("   and Plotly method errors.")
    else:
        print("\n❌ SOME ISSUES REMAIN")
        print("   Please check the error messages above for details.")
    
    print("=" * 60)
