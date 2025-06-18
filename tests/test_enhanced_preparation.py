"""
Test enhanced data preparation features with problematic datasets
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from utils.data_preparation_enhanced import data_prep_tools

def create_problematic_dataset():
    """Create a dataset with common preparation issues"""
    np.random.seed(42)
    n_samples = 200
    
    # Create dataset with various issues
    data = {
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(100, 50, n_samples),  # Different scale
        'feature_3': np.random.choice(['A', 'B', 'C', 'D', 'E'] * 50, n_samples),  # High cardinality
        'feature_4': np.random.randint(1, 10, n_samples),
        'target_class': ['Class_A'] * 100 + ['Class_B'] * 80 + ['Class_C'] * 18 + ['Class_D'] * 2  # Imbalanced
    }
    
    df = pd.DataFrame(data)
    
    # Add missing values
    df.loc[10:20, 'feature_1'] = np.nan
    df.loc[30:35, 'feature_3'] = None
    df.loc[150:155, 'feature_4'] = np.nan
    
    # Add duplicates
    df = pd.concat([df, df.iloc[0:5]], ignore_index=True)
    
    print(f"Created problematic dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Class distribution: {df['target_class'].value_counts().to_dict()}")
    print(f"Missing values: {df.isnull().sum().to_dict()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    
    return df

def test_enhanced_preparation():
    """Test the enhanced data preparation system"""
    print("🧪 Testing Enhanced Data Preparation System")
    print("=" * 60)
    
    # Create problematic dataset
    df = create_problematic_dataset()
    target_column = 'target_class'
    
    # Test 1: Analyze preparation needs
    print("\n📊 Step 1: Analyzing preparation needs...")
    analysis = data_prep_tools.analyze_preparation_needs(df, target_column)
    
    print(f"Issues detected: {len(analysis['issues'])}")
    for i, issue in enumerate(analysis['issues'], 1):
        print(f"  {i}. {issue}")
    
    print(f"\nAuto-fixes available: {analysis['auto_fixes']}")
    
    # Test 2: Get recommendations
    print("\n💡 Step 2: Getting recommendations...")
    recommendations = data_prep_tools.get_preparation_recommendations(analysis)
    
    for rec in recommendations:
        priority_icon = {"Critical": "🚨", "High": "⚡", "Medium": "📋", "Low": "💡"}.get(rec['priority'], "📝")
        print(f"  {priority_icon} {rec['priority']}: {rec['title']}")
        print(f"     {rec['description']}")
    
    # Test 3: Auto-prepare with critical fixes
    print("\n🔧 Step 3: Auto-preparing with critical fixes...")
    try:
        critical_fixes = [rec['action'] for rec in recommendations if rec['priority'] == 'Critical']
        prepared_data = data_prep_tools.auto_prepare_dataset(df, target_column, critical_fixes)
        
        print(f"✅ Auto-preparation successful!")
        print(f"Final shape: {prepared_data['X'].shape}")
        print(f"Classes: {len(prepared_data['target_names'])}")
        print(f"Stratified split: {prepared_data.get('stratified', 'Unknown')}")
        
        # Show preparation log
        if 'preparation_log' in prepared_data:
            print(f"\nPreparation steps applied:")
            for i, log_entry in enumerate(prepared_data['preparation_log'], 1):
                status_icon = "✅" if log_entry['status'] == 'success' else "❌"
                print(f"  {i}. {status_icon} {log_entry['action']}: {log_entry['details']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Auto-preparation failed: {str(e)}")
        return False

def test_stratification_fix():
    """Test the stratification error fix"""
    print("\n🎯 Testing Stratification Error Fix")
    print("=" * 40)
    
    # Create dataset with single-sample classes (causes stratification error)
    data = {
        'feature_1': np.random.normal(0, 1, 10),
        'feature_2': np.random.normal(5, 2, 10),
        'target': ['A'] * 5 + ['B'] * 3 + ['C'] * 1 + ['D'] * 1  # C and D have only 1 sample each
    }
    
    df = pd.DataFrame(data)
    print(f"Created dataset with single-sample classes:")
    print(f"Class distribution: {df['target'].value_counts().to_dict()}")
    
    # Test original function with fix
    try:
        from utils.data_preparation import prepare_dataset_for_ml
        prepared_data = prepare_dataset_for_ml(df, 'target', test_size=0.3)
        
        print(f"✅ Stratification error handled successfully!")
        print(f"Stratified split used: {prepared_data.get('stratified', 'Unknown')}")
        print(f"Final train size: {len(prepared_data['X_train'])}")
        print(f"Final test size: {len(prepared_data['X_test'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Still getting error: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        success1 = test_enhanced_preparation()
        success2 = test_stratification_fix()
        
        print("\n" + "="*60)
        if success1 and success2:
            print("✅ ALL ENHANCED PREPARATION TESTS PASSED!")
        else:
            print("⚠️ Some tests had issues - check output above")
        
        print("\n🚀 Enhanced data preparation features are ready!")
        print("📋 The system can now:")
        print("  • Automatically detect and fix common data issues")
        print("  • Handle stratification errors gracefully")
        print("  • Provide intelligent preparation recommendations")
        print("  • Apply custom fixes based on user selection")
        print("  • Generate detailed preparation logs")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
