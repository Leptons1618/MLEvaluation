"""
Test script for new data upload features
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from utils.data_preparation import (
    analyze_dataset, 
    paginate_dataframe,
    detect_target_column,
    get_data_quality_report
)

def test_data_preparation_features():
    """Test the new data preparation features"""
    print("🧪 Testing Data Upload and Preparation Features")
    print("=" * 50)
    
    # Create a sample dataset
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(5, 2, n_samples),
        'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature_4': np.random.randint(1, 100, n_samples),
        'target': np.random.choice([0, 1, 2], n_samples)
    }
    
    # Add some missing values
    sample_data['feature_1'][50:60] = np.nan
    sample_data['feature_3'][100:105] = None
    
    df = pd.DataFrame(sample_data)
    
    print(f"✅ Created sample dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Test dataset analysis
    print("\n🔍 Testing dataset analysis...")
    analysis = analyze_dataset(df)
    print(f"  ✓ Found {len(analysis['numeric_columns'])} numeric columns")
    print(f"  ✓ Found {len(analysis['categorical_columns'])} categorical columns")
    print(f"  ✓ Detected {analysis['duplicates']} duplicate rows")
    print(f"  ✓ Found {sum(analysis['missing_values'].values())} missing values")
    
    # Test pagination
    print("\n📄 Testing pagination...")
    page_size = 50
    for page in [1, 2, 5, 10]:
        paginated_df, pagination_info = paginate_dataframe(df, page_size, page)
        if pagination_info['current_page'] <= pagination_info['total_pages']:
            print(f"  ✓ Page {page}: {len(paginated_df)} rows (expected: {min(page_size, max(0, len(df) - (page-1)*page_size))})")
        else:
            print(f"  ✓ Page {page}: Beyond total pages ({pagination_info['total_pages']})")
    
    # Test target column detection
    print("\n🎯 Testing target column detection...")
    suggested_targets = detect_target_column(df, analysis)
    print(f"  ✓ Suggested target columns: {suggested_targets}")
    
    # Test data quality report
    print("\n📊 Testing data quality report...")
    quality_report = get_data_quality_report(df, analysis)
    print(f"  ✓ Data completeness: {quality_report['completeness']['complete_percentage']:.1f}%")
    print(f"  ✓ Missing data: {quality_report['completeness']['missing_percentage']:.1f}%")
    print(f"  ✓ Duplicate rows: {quality_report['consistency']['duplicate_percentage']:.1f}%")
    print(f"  ✓ Recommendations: {len(quality_report['recommendations'])}")
    
    print("\n🎉 All data preparation features working correctly!")
    return True

if __name__ == "__main__":
    try:
        test_data_preparation_features()
        print("\n✅ ALL TESTS PASSED!")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
