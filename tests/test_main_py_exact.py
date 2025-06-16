#!/usr/bin/env python3
"""
Test the exact SHAP logic from main.py to verify fixes work
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def test_main_py_shap_logic():
    """Test the exact SHAP logic from main.py"""
    print("🧪 Testing Main.py SHAP Logic")
    print("=" * 40)
    
    # Test case: Gradient Boosting on iris (multi-class, should use fallback)
    print("\n1. Testing Gradient Boosting on iris (multi-class)")
    print("-" * 50)
    
    # Load data exactly like main.py
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train model
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Create input exactly like main.py
    input_df = pd.DataFrame(X_test[0:1], columns=feature_names)
    model_choice = "Gradient Boosting"
    
    # Reproduce the exact SHAP logic from main.py
    try:
        # Tree-based models - try TreeExplainer first, fallback to others
        print(f"  Using TreeExplainer for {model_choice}")
        try:
            explainer_shap = shap.TreeExplainer(model)
            shap_values = explainer_shap.shap_values(input_df)
            expected_value = explainer_shap.expected_value
            use_old_format = True
            print(f"  TreeExplainer succeeded")
        except Exception as tree_error:
            print(f"  TreeExplainer failed: {str(tree_error)}")
            print(f"  Falling back to PermutationExplainer")
              # Fallback to PermutationExplainer for multi-class Gradient Boosting
            explainer_shap = shap.PermutationExplainer(model.predict_proba, X_train[:50])
            shap_values = explainer_shap(input_df)
            expected_value = None  # PermutationExplainer doesn't have expected_value
            use_old_format = False
            print(f"  PermutationExplainer succeeded as fallback")
        
        # Get prediction for the current input
        pred_class_idx = model.predict(input_df)[0]
        print(f"  Predicted class: {pred_class_idx}")
        
        # Handle different SHAP value formats with more robust logic
        if use_old_format:
            # Old format from TreeExplainer
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
        else:
            # New format from general Explainer
            if hasattr(shap_values, 'values'):
                if len(shap_values.values.shape) == 3:
                    # Multi-class case: shape is (n_samples, n_features, n_classes)
                    single_shap_values = shap_values.values[0, :, pred_class_idx]
                    if hasattr(shap_values, 'base_values'):
                        if shap_values.base_values.ndim > 1:
                            single_expected_value = shap_values.base_values[0, pred_class_idx]
                        else:
                            single_expected_value = shap_values.base_values[0]
                    else:
                        single_expected_value = expected_value if expected_value is not None else 0.0
                else:
                    # Binary or regression case: shape is (n_samples, n_features)
                    single_shap_values = shap_values.values[0]
                    single_expected_value = shap_values.base_values[0] if hasattr(shap_values, 'base_values') else (expected_value if expected_value is not None else 0.0)
            else:
                # Fallback case
                single_shap_values = shap_values[0]
                single_expected_value = expected_value if expected_value is not None else 0.0
        
        # SHAP waterfall plot - the exact code from main.py
        fig_shap, ax = plt.subplots(figsize=(10, 6))
        
        # Create a clean explanation object for waterfall plot with proper data types
        # Ensure all values are properly converted to float to avoid format issues
        safe_shap_values = np.array(single_shap_values, dtype=float)
        safe_expected_value = float(single_expected_value) if single_expected_value is not None else 0.0
        safe_input_data = np.array(input_df.values[0], dtype=float)
        
        waterfall_explanation = shap.Explanation(
            values=safe_shap_values,
            base_values=safe_expected_value,
            data=safe_input_data,
            feature_names=list(feature_names)  # Ensure it's a list
        )
        
        # Generate the waterfall plot
        shap.plots.waterfall(waterfall_explanation, show=False)
        plt.close(fig_shap)
        
        print("  ✅ SHAP waterfall plot generated successfully")
        print(f"  SHAP values shape: {safe_shap_values.shape}")
        print(f"  Expected value: {safe_expected_value}")
        print(f"  Sum of SHAP values: {np.sum(safe_shap_values):.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False

def test_svm_pipeline():
    """Test SVM pipeline SHAP logic"""
    print("\n2. Testing SVM on iris (pipeline, should use fallback)")
    print("-" * 50)
    
    # Load data
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(probability=True, random_state=42))
    ])
    model.fit(X_train, y_train)
    
    # Create input
    input_df = pd.DataFrame(X_test[0:1], columns=feature_names)
    model_choice = "SVM"
    
    try:
        # Pipeline models (Logistic Regression, SVM) - need to handle differently
        print(f"  Using general Explainer for {model_choice}")
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
                print(f"  Pipeline model SHAP values calculated with extracted classifier")
                
            except Exception as pipeline_error:
                print(f"  Pipeline extraction failed: {str(pipeline_error)}")
                print(f"  Falling back to PermutationExplainer for pipeline")
                
                # Fallback to PermutationExplainer
                explainer_shap = shap.PermutationExplainer(model.predict_proba, X_train[:50])
                shap_values = explainer_shap(input_df)
                expected_value = None  # PermutationExplainer doesn't have expected_value
                print(f"  PermutationExplainer succeeded for pipeline")
        else:
            # Not a pipeline - use original approach
            explainer_shap = shap.Explainer(model, X_train[:100])
            shap_values = explainer_shap(input_df)
            expected_value = explainer_shap.expected_value
            print(f"  Non-pipeline model succeeded")
        
        use_old_format = False
        
        # Get prediction for the current input
        pred_class_idx = model.predict(input_df)[0]
        print(f"  Predicted class: {pred_class_idx}")
        
        # New format from general Explainer
        if hasattr(shap_values, 'values'):
            if len(shap_values.values.shape) == 3:
                # Multi-class case: shape is (n_samples, n_features, n_classes)
                single_shap_values = shap_values.values[0, :, pred_class_idx]
                if hasattr(shap_values, 'base_values'):
                    if shap_values.base_values.ndim > 1:
                        single_expected_value = shap_values.base_values[0, pred_class_idx]
                    else:
                        single_expected_value = shap_values.base_values[0]
                else:
                    single_expected_value = expected_value if expected_value is not None else 0.0
            else:
                # Binary or regression case: shape is (n_samples, n_features)
                single_shap_values = shap_values.values[0]
                single_expected_value = shap_values.base_values[0] if hasattr(shap_values, 'base_values') else (expected_value if expected_value is not None else 0.0)
        else:
            # Fallback case
            single_shap_values = shap_values[0]
            single_expected_value = expected_value if expected_value is not None else 0.0
        
        # SHAP waterfall plot
        fig_shap, ax = plt.subplots(figsize=(10, 6))
        
        # Create a clean explanation object for waterfall plot with proper data types
        safe_shap_values = np.array(single_shap_values, dtype=float)
        safe_expected_value = float(single_expected_value) if single_expected_value is not None else 0.0
        safe_input_data = np.array(input_df.values[0], dtype=float)
        
        waterfall_explanation = shap.Explanation(
            values=safe_shap_values,
            base_values=safe_expected_value,
            data=safe_input_data,
            feature_names=list(feature_names)
        )
        
        # Generate the waterfall plot
        shap.plots.waterfall(waterfall_explanation, show=False)
        plt.close(fig_shap)
        
        print("  ✅ SHAP waterfall plot generated successfully")
        print(f"  SHAP values shape: {safe_shap_values.shape}")
        print(f"  Expected value: {safe_expected_value}")
        print(f"  Sum of SHAP values: {np.sum(safe_shap_values):.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing exact SHAP logic from main.py")
    
    success1 = test_main_py_shap_logic()
    success2 = test_svm_pipeline()
    
    print("\n" + "=" * 40)
    print("🏁 SUMMARY")
    print("=" * 40)
    
    if success1 and success2:
        print("✅ All main.py SHAP logic tests passed!")
        print("🎉 The fixes in main.py are working correctly!")
    else:
        print("⚠️  Some tests failed")
        print("📝 Issues may still need to be addressed")
    
    import sys
    sys.exit(0 if (success1 and success2) else 1)
