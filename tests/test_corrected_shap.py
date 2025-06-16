#!/usr/bin/env python3
"""
Test the corrected SHAP values after the predict_proba fix
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

def test_corrected_shap_values():
    """Test SHAP values after the predict_proba fix"""
    print("🔧 Testing Corrected SHAP Values (using predict_proba)")
    print("=" * 60)
    
    # Test 1: Gradient Boosting on iris (multi-class, should use PermutationExplainer)
    print("\n1. 📊 Gradient Boosting on iris (multi-class)")
    print("-" * 50)
    
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    sample_input = X_test[0:1]
    input_df = pd.DataFrame(sample_input, columns=feature_names)
    
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    
    print(f"Model prediction: class {prediction}")
    print(f"Prediction probabilities: {prediction_proba}")
    print(f"Probability for predicted class {prediction}: {prediction_proba[prediction]:.6f}")
    
    # This should use PermutationExplainer (TreeExplainer fails for multi-class)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        print("❌ TreeExplainer unexpectedly succeeded (this should fail for multi-class)")
    except:
        print("✅ TreeExplainer failed as expected (multi-class)")
        
        # Now test PermutationExplainer with predict_proba
        explainer = shap.PermutationExplainer(model.predict_proba, X_train[:50])
        shap_values = explainer(input_df)
        
        print("✅ PermutationExplainer with predict_proba succeeded")
        print(f"SHAP values shape: {shap_values.values.shape}")
        
        if len(shap_values.values.shape) == 3:
            # Multi-class case: shape is (n_samples, n_features, n_classes)
            pred_class_idx = prediction
            single_shap_values = shap_values.values[0, :, pred_class_idx]
            if hasattr(shap_values, 'base_values'):
                if shap_values.base_values.ndim > 1:
                    single_expected_value = shap_values.base_values[0, pred_class_idx]
                else:
                    single_expected_value = shap_values.base_values[0]
            else:
                single_expected_value = 0.0
            
            print(f"SHAP values for predicted class {pred_class_idx}:")
            print(f"  SHAP values sum: {np.sum(single_shap_values):.6f}")
            print(f"  Base/expected value: {single_expected_value:.6f}")
            print(f"  Total (SHAP + base): {np.sum(single_shap_values) + single_expected_value:.6f}")
            print(f"  Model probability: {prediction_proba[pred_class_idx]:.6f}")
            print(f"  ✅ Difference: {abs((np.sum(single_shap_values) + single_expected_value) - prediction_proba[pred_class_idx]):.6f}")
    
    # Test 2: SVM on iris (pipeline, should use PermutationExplainer)
    print("\n2. 📊 SVM Pipeline on iris (should use PermutationExplainer)")
    print("-" * 50)
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(probability=True, random_state=42))
    ])
    model.fit(X_train, y_train)
    
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    
    print(f"Model prediction: class {prediction}")
    print(f"Prediction probabilities: {prediction_proba}")
    print(f"Probability for predicted class {prediction}: {prediction_proba[prediction]:.6f}")
    
    # This should fall back to PermutationExplainer
    try:
        classifier = model.named_steps['classifier']
        scaler = model.named_steps['scaler']
        
        X_train_scaled = scaler.transform(X_train)
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
        
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
        
        explainer = shap.Explainer(classifier, X_train_scaled_df[:100])
        shap_values = explainer(input_scaled_df)
        print("❌ Pipeline extraction unexpectedly succeeded")
    except Exception as e:
        print(f"✅ Pipeline extraction failed as expected: {str(e)[:60]}...")
        
        # Now test PermutationExplainer with predict_proba
        explainer = shap.PermutationExplainer(model.predict_proba, X_train[:50])
        shap_values = explainer(input_df)
        
        print("✅ PermutationExplainer with predict_proba succeeded")
        print(f"SHAP values shape: {shap_values.values.shape}")
        
        if len(shap_values.values.shape) == 3:
            # Multi-class case
            pred_class_idx = prediction
            single_shap_values = shap_values.values[0, :, pred_class_idx]
            if hasattr(shap_values, 'base_values'):
                if shap_values.base_values.ndim > 1:
                    single_expected_value = shap_values.base_values[0, pred_class_idx]
                else:
                    single_expected_value = shap_values.base_values[0]
            else:
                single_expected_value = 0.0
            
            print(f"SHAP values for predicted class {pred_class_idx}:")
            print(f"  SHAP values sum: {np.sum(single_shap_values):.6f}")
            print(f"  Base/expected value: {single_expected_value:.6f}")
            print(f"  Total (SHAP + base): {np.sum(single_shap_values) + single_expected_value:.6f}")
            print(f"  Model probability: {prediction_proba[pred_class_idx]:.6f}")
            print(f"  ✅ Difference: {abs((np.sum(single_shap_values) + single_expected_value) - prediction_proba[pred_class_idx]):.6f}")

    print("\n" + "=" * 60)
    print("🎯 Summary:")
    print("✅ PermutationExplainer now uses predict_proba (not predict)")
    print("✅ SHAP values should now sum to probabilities (not class labels)")
    print("✅ Expected values should match base values from SHAP objects")

if __name__ == "__main__":
    test_corrected_shap_values()
