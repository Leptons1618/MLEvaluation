#!/usr/bin/env python3
"""
Detailed SHAP value analysis for Gradient Boosting and SVM
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import shap
import warnings
warnings.filterwarnings('ignore')

def analyze_gradient_boosting_shap():
    """Analyze SHAP values for Gradient Boosting in detail"""
    print("🔍 Analyzing Gradient Boosting SHAP Values")
    print("=" * 50)
    
    # Test both binary and multi-class
    datasets = {
        'breast_cancer': load_breast_cancer(),
        'iris': load_iris()
    }
    
    for name, data in datasets.items():
        print(f"\n📊 Dataset: {name}")
        print("-" * 30)
        
        X, y = data.data, data.target
        feature_names = data.feature_names
        n_classes = len(np.unique(y))
        
        print(f"Classes: {n_classes}, Features: {len(feature_names)}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train model
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Get a test sample
        sample_input = X_test[0:1]
        input_df = pd.DataFrame(sample_input, columns=feature_names)
        
        # Get model prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        print(f"Model prediction: {prediction}")
        print(f"Prediction probabilities: {prediction_proba}")
        
        # Test different SHAP approaches
        print("\n🎯 Testing TreeExplainer:")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            expected_value = explainer.expected_value
            
            print(f"✅ TreeExplainer succeeded")
            print(f"Expected value type: {type(expected_value)}")
            print(f"Expected value: {expected_value}")
            print(f"SHAP values type: {type(shap_values)}")
            print(f"SHAP values shape: {np.array(shap_values).shape}")
            
            if isinstance(shap_values, list):
                print(f"SHAP values (list of {len(shap_values)} arrays):")
                for i, sv in enumerate(shap_values):
                    print(f"  Class {i}: shape={sv.shape}, sum={np.sum(sv[0]):.4f}")
                    
                # Check if our logic extracts the right values
                pred_class_idx = prediction
                single_shap_values = shap_values[pred_class_idx][0]
                single_expected_value = expected_value[pred_class_idx] if isinstance(expected_value, (list, np.ndarray)) else expected_value
                
                print(f"Extracted for class {pred_class_idx}:")
                print(f"  SHAP values sum: {np.sum(single_shap_values):.4f}")
                print(f"  Expected value: {single_expected_value:.4f}")
                print(f"  Total (SHAP + expected): {np.sum(single_shap_values) + single_expected_value:.4f}")
                print(f"  Model prediction prob: {prediction_proba[pred_class_idx]:.4f}")
                
            elif isinstance(shap_values, np.ndarray):
                if len(shap_values.shape) == 3:
                    print(f"SHAP values shape: {shap_values.shape}")
                    pred_class_idx = prediction
                    single_shap_values = shap_values[0, :, pred_class_idx]
                    single_expected_value = expected_value[pred_class_idx] if isinstance(expected_value, (list, np.ndarray)) else expected_value
                else:
                    print(f"SHAP values shape: {shap_values.shape}")
                    single_shap_values = shap_values[0]
                    single_expected_value = expected_value
                
                print(f"Extracted values:")
                print(f"  SHAP values sum: {np.sum(single_shap_values):.4f}")
                print(f"  Expected value: {single_expected_value:.4f}")
                print(f"  Total (SHAP + expected): {np.sum(single_shap_values) + single_expected_value:.4f}")
                print(f"  Model prediction prob: {prediction_proba[prediction]:.4f}")
            
        except Exception as e:
            print(f"❌ TreeExplainer failed: {str(e)}")
            
            print("\n🎯 Testing PermutationExplainer:")
            try:
                explainer = shap.PermutationExplainer(model.predict, X_train[:50])
                shap_values = explainer(input_df)
                
                print(f"✅ PermutationExplainer succeeded")
                print(f"SHAP values type: {type(shap_values)}")
                print(f"SHAP values shape: {shap_values.values.shape if hasattr(shap_values, 'values') else 'No values attr'}")
                
                if hasattr(shap_values, 'values'):
                    if len(shap_values.values.shape) == 3:
                        # Multi-class
                        pred_class_idx = prediction
                        single_shap_values = shap_values.values[0, :, pred_class_idx]
                        if hasattr(shap_values, 'base_values'):
                            if shap_values.base_values.ndim > 1:
                                single_expected_value = shap_values.base_values[0, pred_class_idx]
                            else:
                                single_expected_value = shap_values.base_values[0]
                        else:
                            single_expected_value = 0.0
                    else:
                        # Binary
                        single_shap_values = shap_values.values[0]
                        single_expected_value = shap_values.base_values[0] if hasattr(shap_values, 'base_values') else 0.0
                    
                    print(f"Extracted values:")
                    print(f"  SHAP values sum: {np.sum(single_shap_values):.4f}")
                    print(f"  Expected/base value: {single_expected_value:.4f}")
                    print(f"  Total (SHAP + expected): {np.sum(single_shap_values) + single_expected_value:.4f}")
                    print(f"  Model prediction: {prediction}")
                    print(f"  Model prediction prob: {prediction_proba[prediction]:.4f}")
                
            except Exception as e2:
                print(f"❌ PermutationExplainer failed: {str(e2)}")

def analyze_svm_shap():
    """Analyze SHAP values for SVM in detail"""
    print("\n\n🔍 Analyzing SVM SHAP Values")
    print("=" * 50)
    
    # Test with iris (multi-class)
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    print(f"Dataset: iris")
    print(f"Classes: {len(np.unique(y))}, Features: {len(feature_names)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(probability=True, random_state=42))
    ])
    model.fit(X_train, y_train)
    
    # Get a test sample
    sample_input = X_test[0:1]
    input_df = pd.DataFrame(sample_input, columns=feature_names)
    
    # Get model prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    
    print(f"Model prediction: {prediction}")
    print(f"Prediction probabilities: {prediction_proba}")
    
    # Test pipeline extraction approach
    print("\n🎯 Testing Pipeline Extraction:")
    try:
        classifier = model.named_steps['classifier']
        scaler = model.named_steps['scaler']
        
        X_train_scaled = scaler.transform(X_train)
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
        
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
        
        explainer = shap.Explainer(classifier, X_train_scaled_df[:100])
        shap_values = explainer(input_scaled_df)
        
        print(f"✅ Pipeline extraction succeeded")
        print(f"SHAP values type: {type(shap_values)}")
        print(f"SHAP values shape: {shap_values.values.shape if hasattr(shap_values, 'values') else 'No values attr'}")
        
        if hasattr(shap_values, 'values'):
            if len(shap_values.values.shape) == 3:
                # Multi-class
                pred_class_idx = prediction
                single_shap_values = shap_values.values[0, :, pred_class_idx]
                if hasattr(shap_values, 'base_values'):
                    if shap_values.base_values.ndim > 1:
                        single_expected_value = shap_values.base_values[0, pred_class_idx]
                    else:
                        single_expected_value = shap_values.base_values[0]
                else:
                    single_expected_value = 0.0
            else:
                # Binary
                single_shap_values = shap_values.values[0]
                single_expected_value = shap_values.base_values[0] if hasattr(shap_values, 'base_values') else 0.0
            
            print(f"Extracted values:")
            print(f"  SHAP values sum: {np.sum(single_shap_values):.4f}")
            print(f"  Expected/base value: {single_expected_value:.4f}")
            print(f"  Total (SHAP + expected): {np.sum(single_shap_values) + single_expected_value:.4f}")
            print(f"  Model prediction: {prediction}")
            print(f"  Model prediction prob: {prediction_proba[prediction]:.4f}")
        
    except Exception as e:
        print(f"❌ Pipeline extraction failed: {str(e)}")
        
        print("\n🎯 Testing PermutationExplainer:")
        try:
            explainer = shap.PermutationExplainer(model.predict, X_train[:50])
            shap_values = explainer(input_df)
            
            print(f"✅ PermutationExplainer succeeded")
            print(f"SHAP values type: {type(shap_values)}")
            print(f"SHAP values shape: {shap_values.values.shape if hasattr(shap_values, 'values') else 'No values attr'}")
            
            if hasattr(shap_values, 'values'):
                if len(shap_values.values.shape) == 3:
                    # Multi-class
                    pred_class_idx = prediction
                    single_shap_values = shap_values.values[0, :, pred_class_idx]
                    if hasattr(shap_values, 'base_values'):
                        if shap_values.base_values.ndim > 1:
                            single_expected_value = shap_values.base_values[0, pred_class_idx]
                        else:
                            single_expected_value = shap_values.base_values[0]
                    else:
                        single_expected_value = 0.0
                else:
                    # Binary
                    single_shap_values = shap_values.values[0]
                    single_expected_value = shap_values.base_values[0] if hasattr(shap_values, 'base_values') else 0.0
                
                print(f"Extracted values:")
                print(f"  SHAP values sum: {np.sum(single_shap_values):.4f}")
                print(f"  Expected/base value: {single_expected_value:.4f}")
                print(f"  Total (SHAP + expected): {np.sum(single_shap_values) + single_expected_value:.4f}")
                print(f"  Model prediction: {prediction}")
                print(f"  Model prediction prob: {prediction_proba[prediction]:.4f}")
            
        except Exception as e2:
            print(f"❌ PermutationExplainer failed: {str(e2)}")

if __name__ == "__main__":
    analyze_gradient_boosting_shap()
    analyze_svm_shap()
    
    print("\n" + "=" * 50)
    print("🎯 Key Points to Check:")
    print("1. Do SHAP values + expected value ≈ model prediction probability?")
    print("2. Are we extracting the right class-specific values?")
    print("3. Are base values being handled correctly?")
    print("4. Is the PermutationExplainer using the right function (predict vs predict_proba)?")
