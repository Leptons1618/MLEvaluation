#!/usr/bin/env python3
"""
Debug the remaining SHAP format string error for gradient_boosting on breast_cancer
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def debug_format_error():
    """Debug the format string error"""
    print("🔍 Debugging SHAP format string error")
    print("=" * 50)
    
    # Load data
    dataset = load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Get sample input
    input_df = X_test.iloc[0:1]
    
    print(f"Dataset: {dataset.target_names}")
    print(f"Classes: {len(dataset.target_names)} (binary classification)")
    print(f"Sample input shape: {input_df.shape}")
    
    # Test TreeExplainer (should work for binary classification)
    try:
        print("\nTesting TreeExplainer...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        expected_value = explainer.expected_value
        
        print(f"SHAP values type: {type(shap_values)}")
        print(f"SHAP values shape: {np.array(shap_values).shape}")
        print(f"Expected value type: {type(expected_value)}")
        print(f"Expected value: {expected_value}")
        
        # Get prediction
        pred_class = model.predict(input_df)[0]
        pred_proba = model.predict_proba(input_df)[0]
        print(f"Predicted class: {pred_class}")
        print(f"Predicted probabilities: {pred_proba}")
        
        # Handle binary classification SHAP values
        if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 2:
            # Binary case - use values as-is
            single_shap_values = shap_values[0]
            single_expected_value = expected_value
        else:
            # Multi-class case (shouldn't happen for binary)
            single_shap_values = shap_values[pred_class][0]
            single_expected_value = expected_value[pred_class] if isinstance(expected_value, (list, np.ndarray)) else expected_value
        
        print(f"Single SHAP values shape: {single_shap_values.shape}")
        print(f"Single expected value: {single_expected_value}")
        print(f"SHAP values sum: {np.sum(single_shap_values):.4f}")
        print(f"Data types - SHAP: {single_shap_values.dtype}, Expected: {type(single_expected_value)}")
        
        # Test waterfall plot creation with safe conversions
        try:
            print("\nTesting waterfall plot...")
            
            # Ensure all values are proper float types
            safe_shap_values = np.array(single_shap_values, dtype=np.float64)
            safe_expected_value = np.float64(single_expected_value)
            safe_input_data = np.array(input_df.values[0], dtype=np.float64)
            
            print(f"Safe types - SHAP: {safe_shap_values.dtype}, Expected: {type(safe_expected_value)}")
            print(f"Safe SHAP range: [{safe_shap_values.min():.6f}, {safe_shap_values.max():.6f}]")
            
            waterfall_explanation = shap.Explanation(
                values=safe_shap_values,
                base_values=safe_expected_value,
                data=safe_input_data,
                feature_names=list(dataset.feature_names)
            )
            
            fig, ax = plt.subplots(figsize=(8, 6))
            shap.plots.waterfall(waterfall_explanation, show=False)
            plt.close(fig)
            
            print("✅ Waterfall plot created successfully!")
            return True
            
        except Exception as plot_error:
            print(f"❌ Waterfall plot failed: {str(plot_error)}")
            print(f"Error type: {type(plot_error)}")
            
            # Try alternative approach
            print("\nTrying alternative waterfall approach...")
            try:
                # Use different data types
                alt_shap_values = single_shap_values.astype('float32')
                alt_expected_value = float(single_expected_value)
                alt_input_data = input_df.values[0].astype('float32')
                
                alt_explanation = shap.Explanation(
                    values=alt_shap_values,
                    base_values=alt_expected_value,
                    data=alt_input_data,
                    feature_names=[str(f) for f in dataset.feature_names]  # Convert to strings
                )
                
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.plots.waterfall(alt_explanation, show=False)
                plt.close(fig)
                
                print("✅ Alternative waterfall plot created successfully!")
                return True
                
            except Exception as alt_error:
                print(f"❌ Alternative approach also failed: {str(alt_error)}")
                return False
    
    except Exception as tree_error:
        print(f"❌ TreeExplainer failed: {str(tree_error)}")
        return False

if __name__ == "__main__":
    success = debug_format_error()
    if success:
        print("\n🎉 Issue resolved!")
    else:
        print("\n🔧 Issue requires further investigation")
