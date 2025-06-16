"""
Explanation utilities for ML Evaluation App
Handles SHAP, LIME, and feature importance explanations
"""

import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional
from .config import SHAP_SAMPLE_SIZE, SHAP_BACKGROUND_SIZE
from .model_handler import is_pipeline_model, is_tree_based_model, get_model_name
from .data_handler import ensure_dataframe
from .logging_config import get_logger

logger = get_logger('explanation')


def generate_shap_explanation(
    model, 
    X_train: pd.DataFrame, 
    input_df: pd.DataFrame, 
    feature_names: list
) -> Tuple[Optional[np.ndarray], Optional[float], bool, Optional[str]]:
    """
    Generate SHAP explanation using improved logic with fallbacks
    
    Args:
        model: Trained model
        X_train: Training data for background
        input_df: Input instance to explain
        feature_names: List of feature names
        
    Returns:
        Tuple of (shap_values, expected_value, success, error_message)
    """
    model_name = get_model_name(model)
    logger.info(f"Generating SHAP explanation for {model_name}")
    
    try:
        if is_tree_based_model(model) and not is_pipeline_model(model):
            # Tree-based models - try TreeExplainer first, fallback to others
            logger.debug(f"Using TreeExplainer for {model_name}")
            try:
                explainer_shap = shap.TreeExplainer(model)
                shap_values = explainer_shap.shap_values(input_df)
                expected_value = explainer_shap.expected_value
                use_old_format = True
                logger.debug("TreeExplainer succeeded")
            except Exception as tree_error:
                logger.warning(f"TreeExplainer failed for {model_name}: {str(tree_error)}")
                if "only supported for binary classification" in str(tree_error):
                    logger.info(f"Multi-class classification detected for {model_name}, using PermutationExplainer instead")
                else:
                    logger.info(f"TreeExplainer incompatible with current {model_name} configuration, falling back to PermutationExplainer")
                
                # Fallback to PermutationExplainer - KEY FIX: use predict_proba
                explainer_shap = shap.PermutationExplainer(
                    model.predict_proba, 
                    X_train.sample(SHAP_SAMPLE_SIZE, random_state=42)
                )
                shap_values = explainer_shap(input_df)
                expected_value = None  # PermutationExplainer doesn't have expected_value
                use_old_format = False
                logger.info("PermutationExplainer successfully configured for probability explanation")
        else:
            # Pipeline models or other models - need to handle differently
            logger.debug(f"Using general Explainer for {model_name}")
            if is_pipeline_model(model):
                # This is a pipeline - try different approaches
                classifier = model.named_steps['classifier']
                scaler = model.named_steps['scaler']
                
                try:
                    # Approach 1: Extract classifier and use transformed data
                    X_train_scaled = scaler.transform(X_train)
                    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
                    
                    input_scaled = scaler.transform(input_df)
                    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
                    
                    explainer_shap = shap.Explainer(classifier, X_train_scaled_df.sample(SHAP_BACKGROUND_SIZE))
                    shap_values = explainer_shap(input_scaled_df)
                    expected_value = explainer_shap.expected_value
                    logger.debug("Pipeline model SHAP values calculated with extracted classifier")
                    
                except Exception as pipeline_error:
                    logger.warning(f"Pipeline extraction failed for {model_name}: {str(pipeline_error)}")
                    if "not callable" in str(pipeline_error):
                        logger.info(f"Pipeline model {model_name} requires PermutationExplainer for SHAP analysis")
                    else:
                        logger.info(f"Pipeline approach incompatible, falling back to PermutationExplainer")
                    
                    # Fallback to PermutationExplainer - KEY FIX: use predict_proba
                    explainer_shap = shap.PermutationExplainer(
                        model.predict_proba, 
                        X_train.sample(SHAP_SAMPLE_SIZE, random_state=42)
                    )
                    shap_values = explainer_shap(input_df)
                    expected_value = None  # PermutationExplainer doesn't have expected_value
                    logger.info("PermutationExplainer successfully configured for pipeline probability explanation")
            else:
                # Not a pipeline - use original approach
                explainer_shap = shap.Explainer(model, X_train.sample(SHAP_BACKGROUND_SIZE))
                shap_values = explainer_shap(input_df)
                expected_value = explainer_shap.expected_value
                logger.debug("Non-pipeline model SHAP values calculated")
            use_old_format = False
        
        # Get prediction for the current input
        pred_class_idx = model.predict(input_df)[0]
        pred_proba = model.predict_proba(input_df)[0]
        logger.debug(f"Model prediction: class {pred_class_idx}, probability: {pred_proba[pred_class_idx]:.4f}")
        
        # Handle different SHAP value formats with more robust logic
        single_shap_values, single_expected_value = _extract_single_shap_values(
            shap_values, expected_value, pred_class_idx, use_old_format
        )
        
        # Validate SHAP values
        assert len(single_shap_values) == len(feature_names), f"SHAP values length mismatch: {len(single_shap_values)} vs {len(feature_names)}"
        assert np.isfinite(single_shap_values).all(), "SHAP values contain NaN or inf"
        assert np.isfinite(single_expected_value), f"Expected value is not finite: {single_expected_value}"
        
        # Log SHAP value validation
        shap_sum = np.sum(single_shap_values) + single_expected_value
        logger.debug(f"SHAP validation - Values sum: {np.sum(single_shap_values):.4f}, Base: {single_expected_value:.4f}, Total: {shap_sum:.4f}")
        logger.debug(f"Model probability for class {pred_class_idx}: {pred_proba[pred_class_idx]:.4f}")
        
        return single_shap_values, single_expected_value, True, None
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"SHAP explanation failed for {model_name}: {error_msg}")
        return None, None, False, error_msg


def _extract_single_shap_values(shap_values, expected_value, pred_class_idx, use_old_format):
    """Extract single SHAP values from different formats"""
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
            # Handle expected_value being a numpy array
            if isinstance(expected_value, np.ndarray):
                single_expected_value = expected_value[0] if expected_value.size == 1 else expected_value
            else:
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
    
    return single_shap_values, single_expected_value


def create_shap_waterfall_plot(shap_values, expected_value, input_data, feature_names):
    """Create a SHAP waterfall plot with proper error handling"""
    try:
        # Ensure all values are properly converted to scalars/arrays
        safe_shap_values = np.array(shap_values, dtype=np.float64)
        
        # Handle expected_value being a numpy array or scalar
        if isinstance(expected_value, np.ndarray):
            safe_expected_value = np.float64(expected_value.item() if expected_value.size == 1 else expected_value[0])
        else:
            safe_expected_value = np.float64(expected_value if expected_value is not None else 0.0)
        
        safe_input_data = np.array(input_data, dtype=np.float64)
        
        waterfall_explanation = shap.Explanation(
            values=safe_shap_values,
            base_values=safe_expected_value,
            data=safe_input_data,
            feature_names=[str(f) for f in feature_names]  # Ensure strings
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(waterfall_explanation, show=False)
        
        return fig, True, None
        
    except Exception as e:
        logger.error(f"Error creating SHAP waterfall plot: {str(e)}")
        return None, False, str(e)


def generate_lime_explanation(model, X_train, input_instance, feature_names, target_names):
    """Generate LIME explanation"""
    logger.info("Generating LIME explanation")
    
    try:
        # Create LIME explainer
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=feature_names,
            class_names=target_names,
            mode='classification'
        )
        
        # Create a wrapper function to ensure proper DataFrame format for LIME
        def model_predict_proba_wrapper(X):
            """Wrapper to ensure X is DataFrame with proper feature names"""
            X_df = ensure_dataframe(X, feature_names)
            return model.predict_proba(X_df)
        
        # Explain single instance
        exp = explainer_lime.explain_instance(
            input_instance.values,
            model_predict_proba_wrapper,
            num_features=len(feature_names)
        )
        
        logger.info("LIME explanation generated successfully")
        return exp, True, None
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"LIME explanation failed: {error_msg}")
        return None, False, error_msg


def get_feature_importance(model):
    """Get feature importance if available"""
    try:
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_, True, None
        else:
            return None, False, "Model does not support feature importance"
    except Exception as e:
        return None, False, str(e)
