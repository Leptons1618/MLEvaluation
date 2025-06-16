#!/usr/bin/env python3
"""
SHAP Issues Fix for ML Evaluation Application

This script provides fixes for the identified SHAP issues:
1. Gradient Boosting multi-class support
2. SVM pipeline model handling
"""

import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split

def fix_gradient_boosting_shap():
    """
    Fix for Gradient Boosting SHAP issues
    
    Issues identified:
    1. GradientBoostingClassifier TreeExplainer only supports binary classification
    2. Format string issue with waterfall plot
    """
    print("=== Fixing Gradient Boosting SHAP Issues ===")
    
    # Load test data
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = GradientBoostingClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    sample_input = X_test.iloc[0:1]
    
    print("Attempting different SHAP approaches for Gradient Boosting...")
    
    # Approach 1: Try TreeExplainer (will fail for multi-class)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_input)
        print("✓ TreeExplainer worked (binary classification)")
    except Exception as e:
        print(f"❌ TreeExplainer failed: {str(e)}")
        
        # Approach 2: Use general Explainer with background data
        try:
            print("  Trying general Explainer...")
            explainer = shap.Explainer(model, X_train.sample(50, random_state=42))
            shap_values = explainer(sample_input)
            print("✓ General Explainer worked!")
            
            # Test waterfall plot
            pred_class = model.predict(sample_input)[0]
            if hasattr(shap_values, 'values'):
                if len(shap_values.values.shape) == 3:
                    single_shap_values = shap_values.values[0, :, pred_class]
                    single_expected_value = shap_values.base_values[0, pred_class]
                else:
                    single_shap_values = shap_values.values[0]
                    single_expected_value = shap_values.base_values[0]
                
                print(f"  SHAP values shape: {single_shap_values.shape}")
                print(f"  Expected value: {single_expected_value}")
                
                return 'general_explainer'
                
        except Exception as e2:
            print(f"❌ General Explainer also failed: {str(e2)}")
            
            # Approach 3: Use Permutation Explainer
            try:
                print("  Trying Permutation Explainer...")
                explainer = shap.PermutationExplainer(model.predict, X_train.sample(50, random_state=42))
                shap_values = explainer(sample_input)
                print("✓ Permutation Explainer worked!")
                return 'permutation_explainer'
                
            except Exception as e3:
                print(f"❌ Permutation Explainer failed: {str(e3)}")
                return None

def fix_svm_shap():
    """
    Fix for SVM SHAP issues with pipeline models
    
    Issue identified:
    - SVC model in pipeline is not callable for SHAP
    """
    print("\n=== Fixing SVM SHAP Issues ===")
    
    # Load test data
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(probability=True, random_state=42))
    ])
    model.fit(X_train, y_train)
    
    sample_input = X_test.iloc[0:1]
    
    print("Attempting different SHAP approaches for SVM Pipeline...")
    
    # Approach 1: Extract classifier and work with scaled data
    try:
        print("  Trying to extract classifier from pipeline...")
        classifier = model.named_steps['classifier']
        scaler = model.named_steps['scaler']
        
        # Scale the data
        X_train_scaled = scaler.transform(X_train.sample(50, random_state=42))
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=iris.feature_names)
        
        input_scaled = scaler.transform(sample_input)
        input_scaled_df = pd.DataFrame(input_scaled, columns=iris.feature_names)
        
        # Create SHAP explainer with scaled data
        explainer = shap.Explainer(classifier, X_train_scaled_df)
        shap_values = explainer(input_scaled_df)
        
        print("✓ Pipeline extraction + general Explainer worked!")
        print(f"  SHAP values shape: {shap_values.values.shape}")
        
        return 'pipeline_extraction'
        
    except Exception as e:
        print(f"❌ Pipeline extraction failed: {str(e)}")
        
        # Approach 2: Use Kernel Explainer
        try:
            print("  Trying Kernel Explainer...")
            explainer = shap.KernelExplainer(model.predict_proba, X_train.sample(20, random_state=42))
            shap_values = explainer.shap_values(sample_input, nsamples=50)
            
            print("✓ Kernel Explainer worked!")
            return 'kernel_explainer'
            
        except Exception as e2:
            print(f"❌ Kernel Explainer failed: {str(e2)}")
            
            # Approach 3: Use Permutation Explainer
            try:
                print("  Trying Permutation Explainer...")
                explainer = shap.PermutationExplainer(model.predict, X_train.sample(50, random_state=42))
                shap_values = explainer(sample_input)
                
                print("✓ Permutation Explainer worked!")
                return 'permutation_explainer'
                
            except Exception as e3:
                print(f"❌ Permutation Explainer failed: {str(e3)}")
                return None

def test_waterfall_plot_fix():
    """Test fix for waterfall plot format issues"""
    print("\n=== Testing Waterfall Plot Fix ===")
    
    # Load binary classification data
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = cancer.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train gradient boosting on binary data
    model = GradientBoostingClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    sample_input = X_test.iloc[0:1]
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_input)
        expected_value = explainer.expected_value
        
        # Handle format issue by ensuring proper data types
        single_shap_values = shap_values[0].astype(float)
        single_expected_value = float(expected_value)
        input_data = sample_input.values[0].astype(float)
        
        # Create explanation with proper data types
        waterfall_explanation = shap.Explanation(
            values=single_shap_values,
            base_values=single_expected_value,
            data=input_data,
            feature_names=list(cancer.feature_names)
        )
        
        print("✓ Waterfall explanation object created successfully")
        print(f"  Values dtype: {single_shap_values.dtype}")
        print(f"  Base value type: {type(single_expected_value)}")
        print(f"  Data dtype: {input_data.dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ Waterfall plot fix failed: {str(e)}")
        return False

def main():
    """Run all fixes and generate recommendations"""
    print("SHAP Issues Diagnosis and Fix")
    print("=" * 40)
    
    # Test fixes
    gb_fix = fix_gradient_boosting_shap()
    svm_fix = fix_svm_shap()
    waterfall_fix = test_waterfall_plot_fix()
    
    # Generate recommendations
    print("\n" + "=" * 40)
    print("RECOMMENDATIONS FOR MAIN.PY")
    print("=" * 40)
    
    print("\n1. GRADIENT BOOSTING FIXES:")
    if gb_fix == 'general_explainer':
        print("   ✓ Use general Explainer instead of TreeExplainer for multi-class")
        print("   ✓ Code: explainer = shap.Explainer(model, background_data)")
    elif gb_fix == 'permutation_explainer':
        print("   ✓ Use Permutation Explainer as fallback")
        print("   ✓ Code: explainer = shap.PermutationExplainer(model.predict, background_data)")
    else:
        print("   ❌ No working solution found - consider disabling for multi-class GB")
    
    print("\n2. SVM PIPELINE FIXES:")
    if svm_fix == 'pipeline_extraction':
        print("   ✓ Extract classifier from pipeline and use scaled data")
        print("   ✓ Current implementation should work with proper scaling")
    elif svm_fix == 'kernel_explainer':
        print("   ✓ Use Kernel Explainer as alternative")
        print("   ✓ Code: explainer = shap.KernelExplainer(model.predict_proba, background)")
    elif svm_fix == 'permutation_explainer':
        print("   ✓ Use Permutation Explainer as fallback")
    else:
        print("   ❌ No working solution found - consider disabling for SVM")
    
    print("\n3. WATERFALL PLOT FIXES:")
    if waterfall_fix:
        print("   ✓ Ensure all data types are float for waterfall plots")
        print("   ✓ Code: values.astype(float), float(base_values), data.astype(float)")
    else:
        print("   ❌ Waterfall plot issues persist")
    
    print("\n4. IMPLEMENTATION STRATEGY:")
    print("   1. Add try-catch blocks for different explainer types")
    print("   2. Fall back to alternative explainers when TreeExplainer fails")
    print("   3. Ensure proper data type conversion for plotting")
    print("   4. Add user-friendly error messages for unsupported combinations")

if __name__ == "__main__":
    main()
