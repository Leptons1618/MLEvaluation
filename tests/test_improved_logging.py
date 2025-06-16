#!/usr/bin/env python3
"""
Test the improved logging by running SHAP scenarios that trigger fallbacks
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import shap
import logging
import warnings
warnings.filterwarnings('ignore')

# Setup logging to see our improvements
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create the same logger structure as main.py
explanation_logger = logging.getLogger('ml_evaluation.explanation')

def test_improved_logging():
    """Test improved logging for SHAP explanations"""
    print("🔍 Testing Improved SHAP Logging")
    print("=" * 50)
    
    # Test 1: Gradient Boosting multi-class (should trigger fallback)
    print("\n1. Testing Gradient Boosting on iris (should use fallback)")
    print("-" * 50)
    
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    input_df = pd.DataFrame(X_test[0:1], columns=feature_names)
    model_choice = "Gradient Boosting"
    
    # Simulate the exact logging from main.py
    explanation_logger.info("Starting SHAP explanation generation")
    explanation_logger.debug(f"Using TreeExplainer for {model_choice}")
    
    try:
        explainer_shap = shap.TreeExplainer(model)
        shap_values = explainer_shap.shap_values(input_df)
        expected_value = explainer_shap.expected_value
        use_old_format = True
        explanation_logger.debug("TreeExplainer succeeded")
    except Exception as tree_error:
        explanation_logger.warning(f"TreeExplainer failed for {model_choice}: {str(tree_error)}")
        if "only supported for binary classification" in str(tree_error):
            explanation_logger.info(f"Multi-class classification detected for {model_choice}, using PermutationExplainer instead")
            print("ℹ️ Using PermutationExplainer for multi-class classification (explains probabilities)")
        else:
            explanation_logger.info(f"TreeExplainer incompatible with current {model_choice} configuration, falling back to PermutationExplainer")
            print("ℹ️ Using PermutationExplainer as fallback method")
        
        explainer_shap = shap.PermutationExplainer(model.predict_proba, X_train[:50])
        shap_values = explainer_shap(input_df)
        expected_value = None
        use_old_format = False
        explanation_logger.info("PermutationExplainer successfully configured for probability explanation")
    
    # Test prediction logging
    pred_class_idx = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0]
    explanation_logger.debug(f"Model prediction: class {pred_class_idx}, probability: {pred_proba[pred_class_idx]:.4f}")
    
    # Test SHAP validation logging
    if hasattr(shap_values, 'values'):
        single_shap_values = shap_values.values[0, :, pred_class_idx]
        single_expected_value = shap_values.base_values[0, pred_class_idx] if shap_values.base_values.ndim > 1 else shap_values.base_values[0]
    else:
        single_shap_values = shap_values[pred_class_idx][0]
        single_expected_value = expected_value[pred_class_idx] if isinstance(expected_value, (list, np.ndarray)) else expected_value
    
    safe_shap_values = np.array(single_shap_values, dtype=float)
    safe_expected_value = float(single_expected_value) if single_expected_value is not None else 0.0
    
    shap_sum = np.sum(safe_shap_values) + safe_expected_value
    explanation_logger.debug(f"SHAP validation - Values sum: {np.sum(safe_shap_values):.4f}, Base: {safe_expected_value:.4f}, Total: {shap_sum:.4f}")
    explanation_logger.debug(f"Model probability for class {pred_class_idx}: {pred_proba[pred_class_idx]:.4f}")
    explanation_logger.info(f"SHAP waterfall plot generated successfully for {model_choice}")
    explanation_logger.debug(f"Waterfall plot shows {len(safe_shap_values)} feature contributions")
    explanation_logger.info(f"Feature contributions table generated with {len(feature_names)} features")
    
    print(f"✅ Completed logging test for {model_choice}")
    
    # Test 2: SVM pipeline (should trigger fallback)
    print("\n2. Testing SVM Pipeline on iris (should use fallback)")
    print("-" * 50)
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(probability=True, random_state=42))
    ])
    model.fit(X_train, y_train)
    model_choice = "SVM"
    
    explanation_logger.debug(f"Using general Explainer for {model_choice}")
    
    if hasattr(model, 'named_steps'):
        classifier = model.named_steps['classifier']
        scaler = model.named_steps['scaler']
        
        try:
            X_train_scaled = scaler.transform(X_train)
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
            
            input_scaled = scaler.transform(input_df)
            input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
            
            explainer_shap = shap.Explainer(classifier, X_train_scaled_df[:100])
            shap_values = explainer_shap(input_scaled_df)
            expected_value = explainer_shap.expected_value
            explanation_logger.debug("Pipeline model SHAP values calculated with extracted classifier")
        except Exception as pipeline_error:
            explanation_logger.warning(f"Pipeline extraction failed for {model_choice}: {str(pipeline_error)}")
            if "not callable" in str(pipeline_error):
                explanation_logger.info(f"Pipeline model {model_choice} requires PermutationExplainer for SHAP analysis")
                print(f"ℹ️ Using PermutationExplainer for {model_choice} pipeline (explains probabilities)")
            else:
                explanation_logger.info(f"Pipeline approach incompatible, falling back to PermutationExplainer")
                print("ℹ️ Using PermutationExplainer as fallback method")
            
            explainer_shap = shap.PermutationExplainer(model.predict_proba, X_train[:50])
            shap_values = explainer_shap(input_df)
            expected_value = None
            explanation_logger.info("PermutationExplainer successfully configured for pipeline probability explanation")
    
    print(f"✅ Completed logging test for {model_choice}")
    
    print("\n" + "=" * 50)
    print("🎯 Improved Logging Features:")
    print("✅ Specific error type detection and messaging")
    print("✅ User-friendly explanations of fallback methods")
    print("✅ SHAP value validation logging")
    print("✅ Model prediction and probability logging")
    print("✅ Detailed success and error context")

if __name__ == "__main__":
    test_improved_logging()
