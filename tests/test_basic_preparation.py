"""
Quick test of basic data preparation stratification error handling
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from utils.data_preparation import prepare_dataset_for_ml

def test_basic_stratification_error_handling():
    """Test basic data preparation with stratification error"""
    print("Testing basic data preparation stratification error handling...")
    
    # Create problematic dataset with single-sample classes
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'target': ['A', 'A', 'A', 'A', 'B', 'B', 'C', 'D', 'E', 'F']  # C, D, E, F have only 1 sample each
    }
    df = pd.DataFrame(data)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Class distribution: {df['target'].value_counts().to_dict()}")
    
    try:
        prepared_data = prepare_dataset_for_ml(df, 'target', test_size=0.2)
        
        print(f"\n✅ Basic preparation successful!")
        print(f"  - Training samples: {len(prepared_data['X_train'])}")
        print(f"  - Test samples: {len(prepared_data['X_test'])}")
        print(f"  - Features: {len(prepared_data['feature_names'])}")
        print(f"  - Classes: {len(prepared_data['target_names'])}")
        print(f"  - Stratified: {prepared_data['stratified']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during basic preparation: {e}")
        return False

def test_basic_with_good_data():
    """Test basic data preparation with well-balanced data"""
    print("\n" + "="*50)
    print("Testing basic data preparation with good data...")
    
    # Create well-balanced dataset
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        'target': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'D', 'D', 'D']  # 3 samples each
    }
    df = pd.DataFrame(data)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution: {df['target'].value_counts().to_dict()}")
    
    try:
        prepared_data = prepare_dataset_for_ml(df, 'target', test_size=0.2)
        
        print(f"\n✅ Basic preparation successful!")
        print(f"  - Training samples: {len(prepared_data['X_train'])}")
        print(f"  - Test samples: {len(prepared_data['X_test'])}")
        print(f"  - Features: {len(prepared_data['feature_names'])}")
        print(f"  - Classes: {len(prepared_data['target_names'])}")
        print(f"  - Stratified: {prepared_data['stratified']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during basic preparation: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Basic Data Preparation")
    print("="*60)
    
    try:
        # Test 1: Problematic data (stratification error)
        success1 = test_basic_stratification_error_handling()
        
        # Test 2: Good data (should work with stratification)
        success2 = test_basic_with_good_data()
        
        print("\n" + "="*60)
        print("🎉 Test Summary:")
        print(f"  - Problematic data handling: {'✅ PASS' if success1 else '❌ FAIL'}")
        print(f"  - Good data handling: {'✅ PASS' if success2 else '❌ FAIL'}")
        
        if all([success1, success2]):
            print("\n🚀 Basic data preparation is handling stratification errors correctly!")
        else:
            print("\n⚠️ Some issues detected. Please review the logs above.")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
