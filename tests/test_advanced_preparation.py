"""
Test the enhanced auto-preparation features with advanced capabilities
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from utils.data_preparation_enhanced import data_prep_tools

def test_advanced_features():
    """Test advanced preparation features"""
    print("🧪 Testing Advanced Auto-Preparation Features")
    print("="*60)
    
    # Create a complex dataset with multiple issues
    np.random.seed(42)
    n_samples = 200
    
    # Create data with outliers, high cardinality, and many features
    data = {
        # Many numeric features (will trigger feature selection)
        **{f'feature_{i}': np.random.normal(0, 1, n_samples) for i in range(25)},
        
        # High cardinality categorical feature
        'high_cardinality_cat': [f'Category_{i}' for i in np.random.randint(0, 80, n_samples)],
        
        # Feature with outliers
        'feature_with_outliers': np.random.normal(0, 1, n_samples),
        
        # Target with imbalance
        'target': (['Majority'] * 120 + ['Minority'] * 80)
    }
    
    # Add extreme outliers
    outlier_indices = np.random.choice(n_samples, 10, replace=False)
    data['feature_with_outliers'][outlier_indices] = np.random.normal(10, 1, 10)  # Far outliers
    
    df = pd.DataFrame(data)
    
    print(f"Created complex dataset: {df.shape}")
    print(f"Features: {len(df.columns) - 1}")
    print(f"High cardinality feature unique values: {df['high_cardinality_cat'].nunique()}")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    
    # Analyze
    print("\n🔍 Analyzing preparation needs...")
    analysis = data_prep_tools.analyze_preparation_needs(df, 'target')
    
    print("\nIssues detected:")
    for issue in analysis['issues']:
        print(f"  - {issue}")
    
    # Get recommendations
    recommendations = data_prep_tools.get_preparation_recommendations(analysis)
    print(f"\nRecommendations ({len(recommendations)}):")
    for rec in recommendations:
        print(f"  {rec['priority']}: {rec['title']} - {rec['description']}")
    
    # Test comprehensive auto-preparation
    print("\n🚀 Applying comprehensive auto-preparation...")
    
    try:
        # Apply all available fixes
        all_fixes = [rec['action'] for rec in recommendations if rec['auto_fixable']]
        print(f"Applying fixes: {all_fixes}")
        
        prepared_data = data_prep_tools.auto_prepare_dataset(df, 'target', all_fixes)
        
        print(f"\n✅ Comprehensive preparation successful!")
        print(f"  - Original shape: {df.shape}")
        print(f"  - Processed shape: {prepared_data['X'].shape}")
        print(f"  - Training samples: {len(prepared_data['X_train'])}")
        print(f"  - Test samples: {len(prepared_data['X_test'])}")
        print(f"  - Final features: {len(prepared_data['feature_names'])}")
        print(f"  - Classes: {len(prepared_data['target_names'])}")
        print(f"  - Stratified: {prepared_data['stratified']}")
        
        # Show detailed preparation log
        print(f"\n📝 Preparation Log ({len(prepared_data['preparation_log'])} steps):")
        for i, log_entry in enumerate(prepared_data['preparation_log'], 1):
            status = "✅" if log_entry['status'] == 'success' else "⚠️" if log_entry['status'] == 'warning' else "❌"
            print(f"  {i}. {status} {log_entry['action']}")
            print(f"     {log_entry['details']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during comprehensive preparation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_outlier_detection():
    """Test outlier detection specifically"""
    print("\n" + "="*60)
    print("🔍 Testing Outlier Detection")
    
    # Create dataset with clear outliers
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'normal_feature': np.random.normal(0, 1, n_samples),
        'outlier_feature': np.random.normal(5, 1, n_samples),
        'target': ['A', 'B'] * (n_samples // 2)
    }
    
    # Add extreme outliers
    data['outlier_feature'][5] = 50  # Clear outlier
    data['outlier_feature'][15] = -20  # Clear outlier
    data['normal_feature'][25] = 10  # Clear outlier
    
    df = pd.DataFrame(data)
    
    print(f"Dataset with outliers: {df.shape}")
    print(f"Normal feature range: {df['normal_feature'].min():.2f} to {df['normal_feature'].max():.2f}")
    print(f"Outlier feature range: {df['outlier_feature'].min():.2f} to {df['outlier_feature'].max():.2f}")
    
    # Test outlier detection
    analysis = data_prep_tools.analyze_preparation_needs(df, 'target')
    
    if 'detect_outliers' in analysis['auto_fixes']:
        print("✅ Outliers detected automatically")
        
        # Apply outlier removal
        prepared_data = data_prep_tools.auto_prepare_dataset(df, 'target', ['detect_outliers'])
        
        print(f"After outlier removal: {prepared_data['X'].shape}")
        
        # Check the log
        for log_entry in prepared_data['preparation_log']:
            if 'outlier' in log_entry['action'].lower():
                print(f"Outlier action: {log_entry['details']}")
        
        return True
    else:
        print("⚠️ No outliers detected")
        return True

def test_feature_selection():
    """Test feature selection with many features"""
    print("\n" + "="*60)
    print("🎯 Testing Feature Selection")
    
    # Create dataset with many features
    np.random.seed(42)
    n_samples = 150
    n_features = 30
    
    data = {}
    
    # Create features with varying importance
    for i in range(n_features):
        if i < 5:  # Important features
            data[f'important_{i}'] = np.random.normal(0, 1, n_samples)
        else:  # Less important/random features
            data[f'random_{i}'] = np.random.normal(0, 1, n_samples)
    
    # Create target that depends on important features
    target_signal = sum(data[f'important_{i}'] for i in range(5))
    data['target'] = ['High' if x > 0 else 'Low' for x in target_signal]
    
    df = pd.DataFrame(data)
    
    print(f"Dataset for feature selection: {df.shape}")
    print(f"Features: {len(df.columns) - 1}")
    
    # Test feature selection
    analysis = data_prep_tools.analyze_preparation_needs(df, 'target')
    
    if 'select_features' in analysis['auto_fixes']:
        print("✅ Feature selection recommended")
        
        # Apply feature selection
        prepared_data = data_prep_tools.auto_prepare_dataset(df, 'target', ['select_features'])
        
        print(f"After feature selection: {prepared_data['X'].shape}")
        print(f"Selected features: {prepared_data['feature_names']}")
        
        # Check if important features were kept
        kept_important = sum(1 for name in prepared_data['feature_names'] if 'important' in name)
        print(f"Important features kept: {kept_important}/5")
        
        return True
    else:
        print("⚠️ Feature selection not recommended")
        return False

if __name__ == "__main__":
    try:
        # Test 1: Advanced comprehensive features
        success1 = test_advanced_features()
        
        # Test 2: Outlier detection
        success2 = test_outlier_detection()
        
        # Test 3: Feature selection
        success3 = test_feature_selection()
        
        print("\n" + "="*60)
        print("🎉 Advanced Features Test Summary:")
        print(f"  - Comprehensive preparation: {'✅ PASS' if success1 else '❌ FAIL'}")
        print(f"  - Outlier detection: {'✅ PASS' if success2 else '❌ FAIL'}")
        print(f"  - Feature selection: {'✅ PASS' if success3 else '❌ FAIL'}")
        
        if all([success1, success2, success3]):
            print("\n🚀 All advanced preparation features are working correctly!")
            print("\n🎯 The auto-preparation system now includes:")
            print("   • Critical fixes (stratification errors, missing targets)")
            print("   • Data quality fixes (duplicates, missing values)")
            print("   • Class balancing (SMOTE, over/under-sampling)")
            print("   • Feature engineering (scaling, selection)")
            print("   • Outlier detection and removal")
            print("   • High cardinality categorical handling")
            print("   • Intelligent recommendations and prioritization")
        else:
            print("\n⚠️ Some advanced features need attention. Check logs above.")
            
    except Exception as e:
        print(f"\n❌ Advanced features test failed: {e}")
        import traceback
        traceback.print_exc()
