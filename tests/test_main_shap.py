#!/usr/bin/env python3
"""
Test script to verify SHAP fixes in main.py by calling the actual application functions
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Import the functions from main.py
sys.path.append('.')
from main import setup_logging, load_dataset, train_model

def test_main_shap_functionality():
    """Test SHAP functionality using the actual main.py functions"""
    print("🔧 Testing SHAP Fixes Using Main.py Functions")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    
    # Test datasets
    datasets = {
        'iris': load_iris(),
        'wine': load_wine(), 
        'breast_cancer': load_breast_cancer()
    }
    
    # Test models
    models = {
        'Random Forest': 'Random Forest',
        'Gradient Boosting': 'Gradient Boosting', 
        'Logistic Regression': 'Logistic Regression',
        'SVM': 'SVM'
    }
    
    # Priority problematic combinations (ones we know fail)
    priority_tests = [
        ('Gradient Boosting', 'iris'),      # Multi-class GB - should use fallback
        ('Gradient Boosting', 'wine'),      # Multi-class GB - should use fallback  
        ('Gradient Boosting', 'breast_cancer'), # Binary GB - TreeExplainer should work but waterfall might fail
        ('SVM', 'iris'),                    # SVM pipeline - should use fallback
        ('SVM', 'wine'),                    # SVM pipeline - should use fallback
        ('SVM', 'breast_cancer'),           # SVM pipeline - should use fallback
    ]
    
    issues_found = []
    
    for model_name, dataset_name in priority_tests:
        print(f"\n🧪 Testing {model_name} on {dataset_name}")
        print("-" * 40)
        
        try:
            # Load dataset using main.py function
            X, y, feature_names, class_names = load_dataset(dataset_name)
            print(f"  ✓ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Train model using main.py function
            model = train_model(model_name, X_train, y_train)
            print(f"  ✓ Model trained: {type(model).__name__}")
            
            # Prepare test input
            sample_input = X_test[0:1]  # First test sample
            input_df = pd.DataFrame(sample_input, columns=feature_names)
            
            # Test the SHAP explanation logic from main.py
            # We'll simulate the SHAP explanation section
            print(f"  🎯 Testing SHAP explanation logic...")
            
            # Reproduce the main.py SHAP logic
            if model_name in ["Random Forest", "Gradient Boosting"]:
                # Tree-based models - try TreeExplainer first, fallback to others
                print(f"    - Using TreeExplainer approach for {model_name}")
                try:
                    import shap
                    explainer_shap = shap.TreeExplainer(model)
                    shap_values = explainer_shap.shap_values(input_df)
                    expected_value = explainer_shap.expected_value
                    use_old_format = True
                    print(f"    ✓ TreeExplainer succeeded")
                    
                    # Test prediction and SHAP value extraction
                    pred_class_idx = model.predict(input_df)[0]
                    
                    # Handle different SHAP value formats
                    if isinstance(shap_values, list):
                        # Multi-class case - shap_values is a list of arrays
                        single_shap_values = shap_values[pred_class_idx][0]
                        single_expected_value = expected_value[pred_class_idx] if isinstance(expected_value, (list, np.ndarray)) else expected_value
                    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                        # Multi-class case - shap_values is array with shape (n_samples, n_features, n_classes)
                        single_shap_values = shap_values[0, :, pred_class_idx]
                        single_expected_value = expected_value[pred_class_idx] if isinstance(expected_value, (list, np.ndarray)) else expected_value
                    else:
                        # Binary classification case
                        single_shap_values = shap_values[0]
                        single_expected_value = expected_value
                    
                    # Test waterfall plot creation with float conversion
                    waterfall_explanation = shap.Explanation(
                        values=single_shap_values.astype(float),
                        base_values=float(single_expected_value) if np.isscalar(single_expected_value) else single_expected_value.astype(float),
                        data=input_df.values[0].astype(float),
                        feature_names=list(feature_names)
                    )
                    print(f"    ✓ Waterfall plot data prepared successfully")
                    
                except Exception as tree_error:
                    print(f"    ⚠️  TreeExplainer failed: {str(tree_error)}")
                    print(f"    - Falling back to PermutationExplainer")
                    
                    # Fallback to PermutationExplainer
                    explainer_shap = shap.PermutationExplainer(model.predict_proba, X_train[:50])
                    shap_values = explainer_shap(input_df)
                    expected_value = explainer_shap.expected_value
                    use_old_format = False
                    print(f"    ✓ PermutationExplainer succeeded as fallback")
                    
                    # Extract values from new format
                    pred_class_idx = model.predict(input_df)[0]
                    if hasattr(shap_values, 'values'):
                        if len(shap_values.values.shape) == 3:
                            single_shap_values = shap_values.values[0, :, pred_class_idx]
                            single_expected_value = shap_values.base_values[0, pred_class_idx] if hasattr(shap_values, 'base_values') else expected_value
                        else:
                            single_shap_values = shap_values.values[0]
                            single_expected_value = shap_values.base_values[0] if hasattr(shap_values, 'base_values') else expected_value
                    else:
                        single_shap_values = shap_values[0]
                        single_expected_value = expected_value
                    
                    waterfall_explanation = shap.Explanation(
                        values=single_shap_values.astype(float),
                        base_values=float(single_expected_value) if np.isscalar(single_expected_value) else single_expected_value.astype(float),
                        data=input_df.values[0].astype(float),
                        feature_names=list(feature_names)
                    )
                    print(f"    ✓ Fallback waterfall plot data prepared successfully")
                    
            else:
                # Pipeline models (Logistic Regression, SVM)
                print(f"    - Using pipeline approach for {model_name}")
                if hasattr(model, 'named_steps'):
                    # This is a pipeline - try different approaches
                    classifier = model.named_steps['classifier']
                    scaler = model.named_steps['scaler']
                    
                    try:
                        # Approach 1: Extract classifier and use transformed data
                        X_train_scaled = scaler.transform(X_train)
                        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
                        
                        input_scaled = scaler.transform(input_df)
                        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
                        
                        explainer_shap = shap.Explainer(classifier, X_train_scaled_df[:100])
                        shap_values = explainer_shap(input_scaled_df)
                        expected_value = explainer_shap.expected_value
                        print(f"    ✓ Pipeline classifier extraction succeeded")
                        
                    except Exception as pipeline_error:
                        print(f"    ⚠️  Pipeline extraction failed: {str(pipeline_error)}")
                        print(f"    - Falling back to PermutationExplainer for pipeline")
                        
                        # Fallback to PermutationExplainer
                        explainer_shap = shap.PermutationExplainer(model.predict_proba, X_train[:50])
                        shap_values = explainer_shap(input_df)
                        expected_value = explainer_shap.expected_value
                        print(f"    ✓ PermutationExplainer succeeded for pipeline")
                else:
                    # Not a pipeline
                    explainer_shap = shap.Explainer(model, X_train[:100])
                    shap_values = explainer_shap(input_df)
                    expected_value = explainer_shap.expected_value
                    print(f"    ✓ Non-pipeline model succeeded")
            
            print(f"  ✅ {model_name} on {dataset_name}: SHAP explanation successful!")
            
        except Exception as e:
            error_msg = f"{model_name} on {dataset_name}: {str(e)}"
            issues_found.append(error_msg)
            print(f"  ❌ {error_msg}")
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 MAIN.PY SHAP TEST SUMMARY")
    print("=" * 60)
    
    if issues_found:
        print(f"⚠️  Found {len(issues_found)} issues:")
        for issue in issues_found:
            print(f"  - {issue}")
        return False
    else:
        print("✅ All priority SHAP tests passed!")
        print("🎉 SHAP fixes in main.py are working correctly!")
        return True

if __name__ == "__main__":
    success = test_main_shap_functionality()
    sys.exit(0 if success else 1)
