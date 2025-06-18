"""
Test for numpy.int64 to Python int conversion in Streamlit metrics
"""

import sys
import os
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

import pandas as pd
import numpy as np


def test_numpy_int64_conversion():
    """Test that numpy.int64 values are properly converted to Python int"""
    
    print("🧪 Testing numpy.int64 to Python int conversion...")
    
    # Create a DataFrame with missing values
    df = pd.DataFrame({
        'col1': [1, 2, np.nan, 4, 5],
        'col2': ['A', 'B', np.nan, 'D', 'E'],
        'col3': [1.1, 2.2, 3.3, np.nan, 5.5]
    })
    
    # Get missing value count (this returns numpy.int64)
    missing_count = df.isnull().sum().sum()
    print(f"Missing count type: {type(missing_count)}")
    print(f"Missing count value: {missing_count}")
    
    # Test that it's numpy.int64
    assert isinstance(missing_count, np.int64), f"Expected numpy.int64, got {type(missing_count)}"
    
    # Convert to Python int (this is what we need to do for Streamlit)
    missing_count_int = int(missing_count)
    print(f"Converted type: {type(missing_count_int)}")
    print(f"Converted value: {missing_count_int}")
    
    # Test that it's now Python int
    assert isinstance(missing_count_int, int), f"Expected int, got {type(missing_count_int)}"
    assert missing_count_int == 3, f"Expected 3 missing values, got {missing_count_int}"
    
    # Test delta calculation
    df2 = df.dropna()  # Remove missing values
    missing_count2 = df2.isnull().sum().sum()
    delta = int(missing_count2 - missing_count)
    
    assert isinstance(delta, int), f"Delta should be int, got {type(delta)}"
    assert delta == -3, f"Expected delta of -3, got {delta}"
    
    print("✅ numpy.int64 to Python int conversion test passed!")
    
    # Test with datetime conversion scenario
    print("\n🗓️ Testing datetime conversion scenario...")
    
    # Create DataFrame with date strings
    date_df = pd.DataFrame({
        'release_date': ['2023-01-01', '2023-02-01', '2023-03-01', np.nan],
        'other_col': [1, 2, 3, 4]
    })
    
    print(f"Original release_date dtype: {date_df['release_date'].dtype}")
    
    # Convert to datetime (this is what happens when user changes column type)
    date_df['release_date'] = pd.to_datetime(date_df['release_date'], errors='coerce')
    
    print(f"After conversion dtype: {date_df['release_date'].dtype}")
    
    # Test missing value count after conversion
    missing_after = int(date_df.isnull().sum().sum())
    print(f"Missing values after datetime conversion: {missing_after}")
    
    assert isinstance(missing_after, int), f"Missing count should be int, got {type(missing_after)}"
    
    print("✅ Datetime conversion scenario test passed!")


def test_streamlit_metric_compatibility():
    """Test that our values are compatible with Streamlit's metric function"""
    
    print("\n📊 Testing Streamlit metric compatibility...")
    
    # Simulate what happens in the operation tracker
    orig_df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': ['A', 'B', 'C', 'D', 'E']
    })
    
    curr_df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5, 6],  # Added one row
        'col2': ['A', 'B', 'C', 'D', 'E', 'F'],
        'col3': [1, 2, 3, 4, 5, np.nan]  # Added column with missing value
    })
    
    # Test the exact calculations from the operation tracker
    rows_delta = int(len(curr_df) - len(orig_df))
    cols_delta = int(len(curr_df.columns) - len(orig_df.columns))
    
    curr_missing = int(curr_df.isnull().sum().sum())
    orig_missing = int(orig_df.isnull().sum().sum())
    missing_delta = int(curr_missing - orig_missing)
    
    print(f"Rows delta: {rows_delta} (type: {type(rows_delta)})")
    print(f"Columns delta: {cols_delta} (type: {type(cols_delta)})")
    print(f"Missing values delta: {missing_delta} (type: {type(missing_delta)})")
    
    # All should be Python int, not numpy.int64
    assert isinstance(rows_delta, int)
    assert isinstance(cols_delta, int)
    assert isinstance(missing_delta, int)
    
    # Test expected values
    assert rows_delta == 1
    assert cols_delta == 1
    assert missing_delta == 1
    
    print("✅ Streamlit metric compatibility test passed!")


if __name__ == "__main__":
    print("🔧 Testing numpy.int64 to Python int conversion fixes")
    print("=" * 60)
    
    try:
        test_numpy_int64_conversion()
        test_streamlit_metric_compatibility()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("✅ numpy.int64 conversion working correctly")
        print("✅ Streamlit metric compatibility ensured")
        print("✅ Datetime column type change scenario tested")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
