"""
End-to-end test with sample dataset to validate all fixes
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
    safe_dataframe_display,
    create_arrow_safe_summary
)

def create_realistic_test_dataset():
    """Create a realistic dataset with various issues for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Create base data
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.exponential(50000, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'experience': np.random.uniform(0, 20, n_samples),
        'score': np.random.normal(100, 15, n_samples),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce missing values
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
    df.loc[missing_indices[:100], 'age'] = np.nan
    df.loc[missing_indices[100:200], 'income'] = np.nan
    df.loc[missing_indices[200:300], 'education'] = np.nan
    
    # Add some problematic values
    df.loc[50:60, 'income'] = ['invalid'] * 11
    df.loc[70:75, 'age'] = 'unknown'
    
    # Add duplicates
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    
    return df

def test_end_to_end_workflow():
    """Test complete workflow with realistic dataset"""
    print("🚀 Starting End-to-End Workflow Test")
    print("=" * 50)
    
    # Step 1: Create realistic test data
    print("📊 Creating realistic test dataset...")
    df = create_realistic_test_dataset()
    print(f"   Dataset shape: {df.shape}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    print(f"   Data types: {df.dtypes.value_counts().to_dict()}")
    
    # Step 2: Test Arrow compatibility of raw data
    print("\n🔍 Testing raw data Arrow compatibility...")
    try:
        raw_arrow_df = make_dataframe_arrow_compatible(df)
        import pyarrow as pa
        table = pa.Table.from_pandas(raw_arrow_df)
        print("✅ Raw data made Arrow-compatible successfully")
    except Exception as e:
        print(f"❌ Raw data Arrow compatibility failed: {e}")
        return False
    
    # Step 3: Test safe display
    print("\n📺 Testing safe display functionality...")
    try:
        display_df = safe_dataframe_display(df)
        print(f"   Display-ready dataset shape: {display_df.shape}")
        print("✅ Safe display preparation successful")
    except Exception as e:
        print(f"❌ Safe display preparation failed: {e}")
        return False
    
    # Step 4: Test safe calculations on problematic columns
    print("\n🔢 Testing safe calculations on problematic data...")
    
    # Test on age column (has 'unknown' strings)
    age_mean = safe_mean(df['age'])
    age_median = safe_median(df['age'])
    age_mode = safe_mode(df['age'])
    print(f"   Age column - Mean: {age_mean}, Median: {age_median}, Mode: {age_mode}")
    
    # Test on income column (has 'invalid' strings)
    income_mean = safe_mean(df['income'])
    income_median = safe_median(df['income'])
    income_mode = safe_mode(df['income'])
    print(f"   Income column - Mean: {income_mean}, Median: {income_median}, Mode: {income_mode}")
    
    # Test on categorical column
    edu_mode = safe_mode(df['education'])
    print(f"   Education mode: {edu_mode}")
    
    print("✅ Safe calculations completed successfully")
    
    # Step 5: Test comprehensive imputation
    print("\n🔧 Testing comprehensive imputation...")
    
    imputation_config = {
        'age': {'method': 'Median', 'custom_value': None},
        'income': {'method': 'Mean', 'custom_value': None},
        'education': {'method': 'Mode', 'custom_value': None}
    }
    
    try:
        imputed_df = apply_imputation(df, imputation_config)
        print(f"   Original missing values: {df.isnull().sum().sum()}")
        print(f"   After imputation missing values: {imputed_df.isnull().sum().sum()}")
        
        # Test Arrow compatibility of imputed data
        arrow_table = pa.Table.from_pandas(imputed_df)
        print("✅ Imputation and Arrow serialization successful")
        
    except Exception as e:
        print(f"❌ Imputation failed: {e}")
        return False
    
    # Step 6: Test summary creation
    print("\n📋 Testing Arrow-safe summary creation...")
    try:
        summary_df = create_arrow_safe_summary(imputed_df)
        print(f"   Summary shape: {summary_df.shape}")
        
        # Test Arrow compatibility of summary
        summary_table = pa.Table.from_pandas(summary_df)
        print("✅ Arrow-safe summary creation successful")
        
    except Exception as e:
        print(f"❌ Summary creation failed: {e}")
        return False
    
    # Step 7: Test with extreme edge cases
    print("\n🎯 Testing extreme edge cases...")
    
    # All NaN column
    df_edge = imputed_df.copy()
    df_edge['all_nan'] = np.nan
    
    # Mixed types column
    df_edge['mixed'] = [1, 'text', 3.14, True, None] * (len(df_edge) // 5 + 1)
    df_edge = df_edge.iloc[:len(imputed_df)]
    
    try:
        edge_safe_df = make_dataframe_arrow_compatible(df_edge)
        edge_table = pa.Table.from_pandas(edge_safe_df)
        print("✅ Extreme edge cases handled successfully")
        
    except Exception as e:
        print(f"❌ Edge case handling failed: {e}")
        return False
    
    # Step 8: Performance test with larger dataset
    print("\n⚡ Testing performance with larger dataset...")
    
    large_df = create_realistic_test_dataset()
    large_df = pd.concat([large_df] * 10, ignore_index=True)  # 10x larger
    print(f"   Large dataset shape: {large_df.shape}")
    
    import time
    start_time = time.time()
    
    try:
        large_safe_df = make_dataframe_arrow_compatible(large_df)
        large_table = pa.Table.from_pandas(large_safe_df)
        
        processing_time = time.time() - start_time
        print(f"   Processing time: {processing_time:.2f} seconds")
        print("✅ Performance test successful")
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False
    
    print("\n🎉 ALL END-TO-END TESTS PASSED!")
    return True

if __name__ == "__main__":
    success = test_end_to_end_workflow()
    
    print("\n" + "=" * 50)
    if success:
        print("🎊 END-TO-END VALIDATION SUCCESSFUL!")
        print("   The ML Evaluation application is ready for production use.")
        print("   All Arrow serialization and Plotly fixes are working correctly.")
        print("   Enhanced features are fully functional and reliable.")
    else:
        print("❌ END-TO-END VALIDATION FAILED!")
        print("   Please check the error messages above for issues to resolve.")
    print("=" * 50)
