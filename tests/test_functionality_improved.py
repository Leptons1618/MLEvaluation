#!/usr/bin/env python3
"""
Improved Comprehensive Test Suite for ML Evaluation Application

This test suite uses the same improved SHAP logic as main.py, including:
- Proper fallback mechanisms for multi-class Gradient Boosting
- PermutationExplainer fallback for SVM pipelines
- Robust error handling and logging
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
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import warnings
import logging

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Configure logging for testing
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class TestMLEvaluationAppImproved:
    """Improved test suite using main.py SHAP logic"""
    
    @classmethod
    def setup_class(cls):
        """Set up test fixtures"""
        print("Setting up test fixtures...")
        
        # Load all datasets
        cls.datasets = {
            'iris': load_iris(),
            'wine': load_wine(),
            'breast_cancer': load_breast_cancer()
        }
        
        # Prepare data
        cls.test_data = {}
        for name, dataset in cls.datasets.items():
            X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
            y = dataset.target
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            cls.test_data[name] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': dataset.feature_names,
                'target_names': dataset.target_names
            }
        
        # Define models with same configuration as main.py
        cls.models = {
            'random_forest': lambda: RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': lambda: Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ]),
            'svm': lambda: Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(probability=True, random_state=42))
            ])
        }
        
        # Train all models
        cls.trained_models = {}
        for model_name, model_factory in cls.models.items():
            cls.trained_models[model_name] = {}
            for dataset_name, data in cls.test_data.items():
                model = model_factory()
                model.fit(data['X_train'], data['y_train'])
                
                cls.trained_models[model_name][dataset_name] = {
                    'model': model,
                    'data': data
                }
        
        print("Test fixtures ready!")
    
    @staticmethod
    def ensure_dataframe(X, feature_names):
        """Ensure input is a DataFrame with proper feature names"""
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=feature_names)
    
    def generate_shap_explanation_improved(self, model, model_name, X_train, input_df, feature_names):
        """
        Generate SHAP explanation using improved logic from main.py
        
        Returns:
            tuple: (shap_values, expected_value, success, error_message)
        """
        try:
            logger.info(f"Generating SHAP explanation for {model_name}")
            
            if model_name in ["random_forest", "gradient_boosting"]:
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
                        print(f"  Using PermutationExplainer for multi-class {model_name}")
                    else:
                        logger.info(f"TreeExplainer incompatible with current {model_name} configuration, falling back to PermutationExplainer")
                        print(f"  Using PermutationExplainer as fallback for {model_name}")
                    
                    # Fallback to PermutationExplainer - KEY FIX: use predict_proba
                    explainer_shap = shap.PermutationExplainer(model.predict_proba, X_train.sample(50, random_state=42))
                    shap_values = explainer_shap(input_df)
                    expected_value = None  # PermutationExplainer doesn't have expected_value
                    use_old_format = False
                    logger.info("PermutationExplainer successfully configured for probability explanation")
            else:
                # Pipeline models (Logistic Regression, SVM) - need to handle differently
                logger.debug(f"Using general Explainer for {model_name}")
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
                        
                        explainer_shap = shap.Explainer(classifier, X_train_scaled_df.sample(100))
                        shap_values = explainer_shap(input_scaled_df)
                        expected_value = explainer_shap.expected_value
                        logger.debug("Pipeline model SHAP values calculated with extracted classifier")
                        
                    except Exception as pipeline_error:
                        logger.warning(f"Pipeline extraction failed for {model_name}: {str(pipeline_error)}")
                        if "not callable" in str(pipeline_error):
                            logger.info(f"Pipeline model {model_name} requires PermutationExplainer for SHAP analysis")
                            print(f"  Using PermutationExplainer for {model_name} pipeline")
                        else:
                            logger.info(f"Pipeline approach incompatible, falling back to PermutationExplainer")
                            print(f"  Using PermutationExplainer as fallback for {model_name}")
                        
                        # Fallback to PermutationExplainer - KEY FIX: use predict_proba
                        explainer_shap = shap.PermutationExplainer(model.predict_proba, X_train.sample(50, random_state=42))
                        shap_values = explainer_shap(input_df)
                        expected_value = None  # PermutationExplainer doesn't have expected_value
                        logger.info("PermutationExplainer successfully configured for pipeline probability explanation")
                else:
                    # Not a pipeline - use original approach
                    explainer_shap = shap.Explainer(model, X_train.sample(100))
                    shap_values = explainer_shap(input_df)
                    expected_value = explainer_shap.expected_value
                    logger.debug("Non-pipeline model SHAP values calculated")
                use_old_format = False
            
            # Get prediction for the current input
            pred_class_idx = model.predict(input_df)[0]
            pred_proba = model.predict_proba(input_df)[0]
            logger.debug(f"Model prediction: class {pred_class_idx}, probability: {pred_proba[pred_class_idx]:.4f}")
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
    
    def test_datasets(self):
        """Test dataset loading"""
        print("\n=== Testing Dataset Loading ===")
        for name, dataset in self.datasets.items():
            print(f"Testing {name} dataset...")
            assert len(dataset.data) > 0, f"{name} dataset is empty"
            assert len(dataset.target) > 0, f"{name} target is empty"
            print(f"  ✓ {name}: {len(dataset.data)} samples, {len(dataset.data[0])} features")
        print("✅ All datasets loaded successfully!")
    
    def test_model_training(self):
        """Test model training"""
        print("\n=== Testing Model Training ===")
        for model_name, model_data in self.trained_models.items():
            print(f"Testing {model_name}...")
            for dataset_name, data in model_data.items():
                model = data['model']
                test_data = data['data']
                
                # Test predictions
                train_score = model.score(test_data['X_train'], test_data['y_train'])
                test_score = model.score(test_data['X_test'], test_data['y_test'])
                
                assert train_score > 0.5, f"{model_name} on {dataset_name} train score too low: {train_score}"
                assert test_score > 0.5, f"{model_name} on {dataset_name} test score too low: {test_score}"
                
                print(f"  ✓ {model_name} on {dataset_name}: Train={train_score:.3f}, Test={test_score:.3f}")
        print("✅ All models trained successfully!")
    
    def test_predictions(self):
        """Test model predictions"""
        print("\n=== Testing Predictions ===")
        for model_name, model_data in self.trained_models.items():
            for dataset_name, data in model_data.items():
                model = data['model']
                test_data = data['data']
                
                # Test single prediction
                sample_input = test_data['X_test'].iloc[0:1]
                pred_class = model.predict(sample_input)[0]
                pred_proba = model.predict_proba(sample_input)[0]
                
                assert pred_class in range(len(test_data['target_names'])), f"Invalid prediction class: {pred_class}"
                assert len(pred_proba) == len(test_data['target_names']), f"Probability shape mismatch"
                assert abs(sum(pred_proba) - 1.0) < 1e-6, f"Probabilities don't sum to 1: {sum(pred_proba)}"
                
                print(f"  ✓ {model_name} on {dataset_name}: Pred={pred_class}, Confidence={pred_proba[pred_class]:.3f}")
        print("✅ All predictions working correctly!")
    
    def test_shap_explanations_improved(self):
        """Test SHAP explanations using improved logic from main.py"""
        print("\n=== Testing SHAP Explanations (Improved) ===")
        
        success_count = 0
        total_count = 0
        issues = []
        
        for model_name, model_data in self.trained_models.items():
            for dataset_name, data in model_data.items():
                total_count += 1
                print(f"\nTesting SHAP for {model_name} on {dataset_name}...")
                
                model = data['model']
                test_data = data['data']
                sample_input = test_data['X_test'].iloc[0:1]
                input_df = self.ensure_dataframe(sample_input, test_data['feature_names'])
                
                # Use improved SHAP logic
                shap_values, expected_value, success, error_msg = self.generate_shap_explanation_improved(
                    model, model_name, test_data['X_train'], input_df, test_data['feature_names']
                )
                
                if success:
                    success_count += 1
                    pred_class = model.predict(input_df)[0]
                    
                    print(f"  ✅ SHAP succeeded for {model_name} on {dataset_name}")
                    print(f"    Predicted class: {pred_class}")
                    print(f"    SHAP values shape: {np.array(shap_values).shape}")
                    print(f"    Expected value: {expected_value}")
                    print(f"    SHAP sum: {np.sum(shap_values):.4f}")
                      # Test waterfall plot creation
                    try:
                        # Ensure all values are properly converted to scalars/arrays
                        safe_shap_values = np.array(shap_values, dtype=np.float64)
                        
                        # Handle expected_value being a numpy array or scalar
                        if isinstance(expected_value, np.ndarray):
                            safe_expected_value = np.float64(expected_value.item() if expected_value.size == 1 else expected_value[0])
                        else:
                            safe_expected_value = np.float64(expected_value if expected_value is not None else 0.0)
                        
                        safe_input_data = np.array(input_df.values[0], dtype=np.float64)
                        
                        waterfall_explanation = shap.Explanation(
                            values=safe_shap_values,
                            base_values=safe_expected_value,
                            data=safe_input_data,
                            feature_names=[str(f) for f in test_data['feature_names']]  # Ensure strings
                        )
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        shap.plots.waterfall(waterfall_explanation, show=False)
                        plt.close(fig)
                        
                        print(f"    ✓ Waterfall plot created successfully")
                        
                    except Exception as plot_error:
                        print(f"    ⚠️  Waterfall plot failed: {str(plot_error)}")
                        issues.append(f"{model_name}-{dataset_name}: Waterfall plot - {str(plot_error)}")
                else:
                    print(f"  ❌ SHAP failed for {model_name} on {dataset_name}: {error_msg}")
                    issues.append(f"{model_name}-{dataset_name}: {error_msg}")
        
        print(f"\n📊 SHAP Results Summary:")
        print(f"  Success: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        
        if issues:
            print(f"\n⚠️  Issues found ({len(issues)}):")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\n✅ All SHAP explanations working correctly!")
        
        return success_count, total_count, issues
    
    def test_lime_explanations(self):
        """Test LIME explanations"""
        print("\n=== Testing LIME Explanations ===")
        
        for model_name, model_data in self.trained_models.items():
            for dataset_name, data in model_data.items():
                print(f"Testing LIME for {model_name} on {dataset_name}...")
                
                model = data['model']
                test_data = data['data']
                
                try:
                    # LIME explainer
                    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
                        test_data['X_train'].values,
                        feature_names=test_data['feature_names'],
                        class_names=test_data['target_names'],
                        mode='classification'
                    )
                    
                    # Explain single instance
                    sample_input = test_data['X_test'].iloc[0]
                    
                    def model_predict_proba_wrapper(X):
                        X_df = self.ensure_dataframe(X, test_data['feature_names'])
                        return model.predict_proba(X_df)
                    
                    exp = explainer_lime.explain_instance(
                        sample_input.values,
                        model_predict_proba_wrapper,
                        num_features=len(test_data['feature_names'])
                    )
                    
                    explanations = exp.as_list()
                    assert len(explanations) > 0, "No LIME explanations generated"
                    
                    print(f"  ✓ {model_name} on {dataset_name}: {len(explanations)} feature explanations")
                    
                except Exception as e:
                    print(f"  ❌ LIME failed for {model_name} on {dataset_name}: {str(e)}")
                    raise
        
        print("✅ All LIME explanations working correctly!")
    
    def test_feature_importance(self):
        """Test feature importance"""
        print("\n=== Testing Feature Importance ===")
        
        for model_name, model_data in self.trained_models.items():
            for dataset_name, data in model_data.items():
                model = data['model']
                
                try:
                    # Check if model has feature importance
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        assert len(importances) == len(data['data']['feature_names']), "Feature importance length mismatch"
                        assert np.isfinite(importances).all(), "Feature importance contains NaN or inf"
                        print(f"  ✓ {model_name} on {dataset_name}: Feature importance available")
                    else:
                        print(f"  ℹ️  {model_name} on {dataset_name}: No feature importance (expected for pipelines)")
                        
                except Exception as e:
                    print(f"  ❌ Feature importance failed for {model_name} on {dataset_name}: {str(e)}")
                    raise
        
        print("✅ Feature importance tests completed!")
    
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting Improved ML Evaluation App Tests")
        print("=" * 60)
        
        try:
            self.test_datasets()
            self.test_model_training()
            self.test_predictions()
            success_count, total_count, shap_issues = self.test_shap_explanations_improved()
            self.test_lime_explanations()
            self.test_feature_importance()
            
            print("\n" + "=" * 60)
            print("🏁 TEST SUMMARY")
            print("=" * 60)
            
            if shap_issues:
                print(f"⚠️  FOUND {len(shap_issues)} SHAP ISSUES")
                print("SHAP Issues:")
                for issue in shap_issues:
                    print(f"  - {issue}")
                print("\n📋 RECOMMENDATIONS:")
                print("  - Check that fallback logic is working correctly")
                print("  - Verify PermutationExplainer is using predict_proba")
                print("  - Review error logs for detailed debugging information")
                print("🔧 Some issues found, but fallbacks should be working.")
                return 1
            else:
                print("✅ ALL TESTS PASSED!")
                print(f"🎉 SHAP Success Rate: {success_count}/{total_count} (100%)")
                print("🎯 All models and explanation methods working correctly!")
                return 0
        
        except Exception as e:
            print(f"\n❌ Test suite failed with error: {str(e)}")
            return 1

def main():
    """Main test function"""
    print("ML Evaluation App - Improved Comprehensive Test Suite")
    print("=" * 60)
    
    # Create test instance
    test_suite = TestMLEvaluationAppImproved()
    test_suite.setup_class()
    
    # Run tests
    exit_code = test_suite.run_all_tests()
    exit(exit_code)

if __name__ == "__main__":
    main()
