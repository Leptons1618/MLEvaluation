#!/usr/bin/env python3
"""
Simplified test script to verify SHAP fallback fixes directly
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
import warnings
warnings.filterwarnings('ignore')

def test_shap_fixes():
    """Test the SHAP fallback fixes for Gradient Boosting and SVM"""
    print("🔧 Testing SHAP Fallback Fixes")
    print("=" * 50)
    
    # Test datasets
    datasets = {
        'iris': load_iris(),
        'wine': load_wine(), 
        'breast_cancer': load_breast_cancer()
    }
    
    issues_found = []
    
    # Priority problematic combinations from our test suite
    priority_tests = [
        ('Gradient Boosting', 'iris'),      # Multi-class GB - should use fallback
        ('Gradient Boosting', 'wine'),      # Multi-class GB - should use fallback  
        ('Gradient Boosting', 'breast_cancer'), # Binary GB - TreeExplainer should work
        ('SVM', 'iris'),                    # SVM pipeline - should use fallback
        ('SVM', 'wine'),                    # SVM pipeline - should use fallback
        ('SVM', 'breast_cancer'),           # SVM pipeline - should use fallback
    ]
    
    for model_name, dataset_name in priority_tests:
        print(f"\n🧪 Testing {model_name} on {dataset_name}")
        print("-" * 40)
        
        try:
            # Load dataset
            data = datasets[dataset_name]
            X, y = data.data, data.target
            feature_names = data.feature_names
            
            print(f"  ✓ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Train model
            if model_name == 'Gradient Boosting':
                model = GradientBoostingClassifier(random_state=42)
            elif model_name == 'SVM':
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', SVC(probability=True, random_state=42))
                ])
            
            model.fit(X_train, y_train)
            print(f"  ✓ Model trained: {type(model).__name__}")
            
            # Prepare test input
            sample_input = X_test[0:1]  # First test sample
            input_df = pd.DataFrame(sample_input, columns=feature_names)
            
            # Test the SHAP explanation logic with fallbacks
            print(f"  🎯 Testing SHAP explanation with fallbacks...")
            
            # Implement the same logic as main.py
            if model_name == "Gradient Boosting":
                # Tree-based models - try TreeExplainer first, fallback to PermutationExplainer
                print(f"    - Trying TreeExplainer for {model_name}")
                try:
                    explainer_shap = shap.TreeExplainer(model)
                    shap_values = explainer_shap.shap_values(input_df)
                    expected_value = explainer_shap.expected_value
                    use_old_format = True
                    print(f"    ✓ TreeExplainer succeeded")
                    fallback_used = False
                    
                except Exception as tree_error:
                    print(f"    ⚠️  TreeExplainer failed: {str(tree_error)}")
                    print(f"    - Falling back to PermutationExplainer")
                    
                    # Fallback to PermutationExplainer for multi-class Gradient Boosting
                    explainer_shap = shap.PermutationExplainer(model.predict_proba, X_train[:50])
                    shap_values = explainer_shap(input_df)
                    expected_value = explainer_shap.expected_value
                    use_old_format = False
                    print(f"    ✓ PermutationExplainer succeeded as fallback")
                    fallback_used = True
                    
            elif model_name == "SVM":
                # Pipeline models - try different approaches
                print(f"    - Trying pipeline approach for {model_name}")
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
                        fallback_used = False
                        
                    except Exception as pipeline_error:
                        print(f"    ⚠️  Pipeline extraction failed: {str(pipeline_error)}")
                        print(f"    - Falling back to PermutationExplainer for pipeline")
                        
                        # Fallback to PermutationExplainer
                        explainer_shap = shap.PermutationExplainer(model.predict_proba, X_train[:50])
                        shap_values = explainer_shap(input_df)
                        expected_value = explainer_shap.expected_value
                        print(f"    ✓ PermutationExplainer succeeded for pipeline")
                        fallback_used = True
                        use_old_format = False
            
            # Test prediction and SHAP value extraction 
            pred_class_idx = model.predict(input_df)[0]
            print(f"    - Predicted class: {pred_class_idx}")
            
            # Handle different SHAP value formats
            if fallback_used:
                # New format from PermutationExplainer
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
                            single_expected_value = expected_value
                    else:
                        # Binary or regression case: shape is (n_samples, n_features)
                        single_shap_values = shap_values.values[0]
                        single_expected_value = shap_values.base_values[0] if hasattr(shap_values, 'base_values') else expected_value
                else:
                    # Fallback case
                    single_shap_values = shap_values[0]
                    single_expected_value = expected_value
            else:
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
            
            # Test waterfall plot creation with float conversion (the fix for format errors)
            try:
                waterfall_explanation = shap.Explanation(
                    values=single_shap_values.astype(float),
                    base_values=float(single_expected_value) if np.isscalar(single_expected_value) else single_expected_value.astype(float),
                    data=input_df.values[0].astype(float),
                    feature_names=list(feature_names)
                )
                print(f"    ✓ Waterfall plot data prepared successfully")
                
                # Show summary
                print(f"    - SHAP values shape: {single_shap_values.shape}")
                print(f"    - Expected value: {single_expected_value:.4f}")
                print(f"    - Sum of SHAP values: {np.sum(single_shap_values):.4f}")
                print(f"    - Fallback used: {fallback_used}")
                
            except Exception as waterfall_error:
                print(f"    ❌ Waterfall plot preparation failed: {str(waterfall_error)}")
                issues_found.append(f"{model_name} on {dataset_name}: Waterfall plot failed - {str(waterfall_error)}")
                continue
            
            print(f"  ✅ {model_name} on {dataset_name}: SHAP explanation successful!")
            
        except Exception as e:
            error_msg = f"{model_name} on {dataset_name}: {str(e)}"
            issues_found.append(error_msg)
            print(f"  ❌ {error_msg}")
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 SHAP FALLBACK TEST SUMMARY")
    print("=" * 50)
    
    if issues_found:
        print(f"⚠️  Found {len(issues_found)} issues:")
        for issue in issues_found:
            print(f"  - {issue}")
        return False
    else:
        print("✅ All SHAP fallback tests passed!")
        print("🎉 SHAP fixes are working correctly!")
        return True

if __name__ == "__main__":
    import sys
    success = test_shap_fixes()
    sys.exit(0 if success else 1)
