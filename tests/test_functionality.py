#!/usr/bin/env python3
"""
Comprehensive Test Suite for ML Evaluation Application

This test suite validates all major functionality including:
- Dataset loading
- Model training
- Predictions
- Explainability methods (SHAP, LIME, Feature Importance)
- Special focus on SHAP issues with Gradient Boosting and SVM
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

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

class TestMLEvaluationApp:
    """Test suite for ML Evaluation Application"""
    
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
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            cls.test_data[name] = {
                'X': X,
                'y': y,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': dataset.feature_names,
                'target_names': dataset.target_names
            }
        
        # Model configurations
        cls.models = {
            'random_forest': RandomForestClassifier(n_estimators=10, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=10, random_state=42),
            'logistic_regression': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ]),
            'svm': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(probability=True, random_state=42, kernel='rbf'))
            ])
        }
        
        print("Test fixtures ready!")
    
    def ensure_dataframe(self, data, feature_names):
        """Helper function to ensure proper DataFrame format"""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = data.reshape(1, -1)
            return pd.DataFrame(data, columns=feature_names)
        else:
            return pd.DataFrame(data, columns=feature_names)
    
    def test_dataset_loading(self):
        """Test dataset loading functionality"""
        print("\n=== Testing Dataset Loading ===")
        
        for dataset_name, data in self.test_data.items():
            print(f"Testing {dataset_name} dataset...")
            
            # Check data integrity
            assert data['X'].shape[0] > 0, f"{dataset_name}: No samples in X"
            assert len(data['y']) > 0, f"{dataset_name}: No samples in y"
            assert data['X'].shape[0] == len(data['y']), f"{dataset_name}: X and y size mismatch"
            assert len(data['feature_names']) == data['X'].shape[1], f"{dataset_name}: Feature names mismatch"
              # Check train/test split
            assert data['X_train'].shape[0] > 0, f"{dataset_name}: No training samples"
            assert data['X_test'].shape[0] > 0, f"{dataset_name}: No test samples"
            print(f"  ✓ {dataset_name}: {data['X'].shape[0]} samples, {data['X'].shape[1]} features")
        
        print("✅ All datasets loaded successfully!")
    
    def test_model_training(self):
        """Test model training for all model types"""
        print("\n=== Testing Model Training ===")
        
        self.trained_models = {}
        
        for model_name, model in self.models.items():
            print(f"Testing {model_name}...")
            self.trained_models[model_name] = {}
            
            for dataset_name, data in self.test_data.items():
                try:
                    # Create a fresh copy of the model
                    if model_name == 'random_forest':
                        model_copy = RandomForestClassifier(n_estimators=10, random_state=42)
                    elif model_name == 'gradient_boosting':
                        model_copy = GradientBoostingClassifier(n_estimators=10, random_state=42)
                    elif model_name == 'logistic_regression':
                        model_copy = Pipeline([
                            ('scaler', StandardScaler()),
                            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
                        ])
                    elif model_name == 'svm':
                        model_copy = Pipeline([
                            ('scaler', StandardScaler()),
                            ('classifier', SVC(probability=True, random_state=42, kernel='rbf'))
                        ])
                    
                    # Train model
                    model_copy.fit(data['X_train'], data['y_train'])
                    
                    # Test prediction
                    train_score = model_copy.score(data['X_train'], data['y_train'])
                    test_score = model_copy.score(data['X_test'], data['y_test'])
                    
                    self.trained_models[model_name][dataset_name] = {
                        'model': model_copy,
                        'train_score': train_score,
                        'test_score': test_score,
                        'data': data
                    }
                    
                    print(f"  ✓ {model_name} on {dataset_name}: Train={train_score:.3f}, Test={test_score:.3f}")
                    
                except Exception as e:
                    print(f"  ❌ {model_name} on {dataset_name}: {str(e)}")
                    raise
        
        print("✅ All models trained successfully!")
    
    def test_predictions(self):
        """Test prediction functionality"""
        print("\n=== Testing Predictions ===")
        
        for model_name, model_data in self.trained_models.items():
            for dataset_name, data in model_data.items():
                model = data['model']
                test_data = data['data']
                
                # Test single prediction
                sample_input = test_data['X_test'].iloc[0:1]
                
                try:
                    # Ensure DataFrame format
                    input_df = self.ensure_dataframe(sample_input, test_data['feature_names'])
                    
                    # Test prediction
                    pred_class = model.predict(input_df)[0]
                    pred_proba = model.predict_proba(input_df)[0]
                    
                    assert pred_class in range(len(test_data['target_names'])), f"Invalid prediction class: {pred_class}"
                    assert len(pred_proba) == len(test_data['target_names']), f"Probability shape mismatch"
                    assert abs(sum(pred_proba) - 1.0) < 1e-6, f"Probabilities don't sum to 1: {sum(pred_proba)}"
                    
                    print(f"  ✓ {model_name} on {dataset_name}: Pred={pred_class}, Confidence={pred_proba[pred_class]:.3f}")
                    
                except Exception as e:
                    print(f"  ❌ {model_name} on {dataset_name}: {str(e)}")
                    raise
        
        print("✅ All predictions working correctly!")
    
    def test_shap_explanations(self):
        """Test SHAP explanations for all models with detailed error handling"""
        print("\n=== Testing SHAP Explanations ===")
        
        problematic_combinations = []
        
        for model_name, model_data in self.trained_models.items():
            for dataset_name, data in model_data.items():
                print(f"\nTesting SHAP for {model_name} on {dataset_name}...")
                
                model = data['model']
                test_data = data['data']
                sample_input = test_data['X_test'].iloc[0:1]
                input_df = self.ensure_dataframe(sample_input, test_data['feature_names'])
                
                try:
                    # Determine SHAP explainer type
                    if model_name in ['random_forest', 'gradient_boosting']:
                        print(f"  Using TreeExplainer for {model_name}")
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(input_df)
                        expected_value = explainer.expected_value
                        use_old_format = True
                        
                    else:  # Pipeline models
                        print(f"  Using general Explainer for {model_name}")
                        if hasattr(model, 'named_steps'):
                            # Pipeline model
                            classifier = model.named_steps['classifier']
                            scaler = model.named_steps['scaler']
                            
                            # Transform data
                            X_train_scaled = scaler.transform(test_data['X_train'].sample(50, random_state=42))
                            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=test_data['feature_names'])
                            
                            input_scaled = scaler.transform(input_df)
                            input_scaled_df = pd.DataFrame(input_scaled, columns=test_data['feature_names'])
                            
                            explainer = shap.Explainer(classifier, X_train_scaled_df)
                            shap_values = explainer(input_scaled_df)
                            expected_value = explainer.expected_value
                        else:
                            # Non-pipeline model
                            explainer = shap.Explainer(model, test_data['X_train'].sample(50, random_state=42))
                            shap_values = explainer(input_df)
                            expected_value = explainer.expected_value
                        
                        use_old_format = False
                    
                    # Analyze SHAP values structure
                    print(f"    SHAP values type: {type(shap_values)}")
                    
                    if use_old_format:
                        if isinstance(shap_values, list):
                            print(f"    SHAP values: list with {len(shap_values)} elements")
                            print(f"    First element shape: {np.array(shap_values[0]).shape}")
                        else:
                            print(f"    SHAP values shape: {np.array(shap_values).shape}")
                    else:
                        if hasattr(shap_values, 'values'):
                            print(f"    SHAP values shape: {shap_values.values.shape}")
                        else:
                            print(f"    SHAP values shape: {np.array(shap_values).shape}")
                    
                    # Test SHAP value extraction
                    pred_class_idx = model.predict(input_df)[0]
                    print(f"    Predicted class: {pred_class_idx}")
                    
                    # Extract single SHAP values
                    if use_old_format:
                        if isinstance(shap_values, list):
                            single_shap_values = shap_values[pred_class_idx][0]
                            single_expected_value = expected_value[pred_class_idx] if isinstance(expected_value, (list, np.ndarray)) else expected_value
                        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                            single_shap_values = shap_values[0, :, pred_class_idx]
                            single_expected_value = expected_value[pred_class_idx] if isinstance(expected_value, (list, np.ndarray)) else expected_value
                        else:
                            single_shap_values = shap_values[0] if len(shap_values.shape) > 1 else shap_values
                            single_expected_value = expected_value
                    else:
                        if hasattr(shap_values, 'values'):
                            if len(shap_values.values.shape) == 3:
                                single_shap_values = shap_values.values[0, :, pred_class_idx]
                                if hasattr(shap_values, 'base_values'):
                                    if shap_values.base_values.ndim > 1:
                                        single_expected_value = shap_values.base_values[0, pred_class_idx]
                                    else:
                                        single_expected_value = shap_values.base_values[0]
                                else:
                                    single_expected_value = expected_value
                            else:
                                single_shap_values = shap_values.values[0]
                                single_expected_value = shap_values.base_values[0] if hasattr(shap_values, 'base_values') else expected_value
                        else:
                            single_shap_values = shap_values[0]
                            single_expected_value = expected_value
                    
                    # Validate SHAP values
                    assert len(single_shap_values) == len(test_data['feature_names']), f"SHAP values length mismatch: {len(single_shap_values)} vs {len(test_data['feature_names'])}"
                    assert np.isfinite(single_shap_values).all(), "SHAP values contain NaN or inf"
                    assert np.isfinite(single_expected_value), f"Expected value is not finite: {single_expected_value}"
                    
                    print(f"    ✓ SHAP values extracted: {len(single_shap_values)} features")
                    print(f"    ✓ Expected value: {single_expected_value:.4f}")
                    print(f"    ✓ SHAP sum: {single_shap_values.sum():.4f}")
                    
                    # Test waterfall plot creation
                    try:
                        waterfall_explanation = shap.Explanation(
                            values=single_shap_values,
                            base_values=single_expected_value,
                            data=input_df.values[0],
                            feature_names=list(test_data['feature_names'])
                        )
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        shap.plots.waterfall(waterfall_explanation, show=False)
                        plt.close(fig)
                        
                        print(f"    ✓ Waterfall plot created successfully")
                        
                    except Exception as plot_error:
                        print(f"    ⚠️  Waterfall plot failed: {str(plot_error)}")
                        problematic_combinations.append(f"{model_name}-{dataset_name}: Waterfall plot - {str(plot_error)}")
                    
                except Exception as e:
                    print(f"  ❌ SHAP failed for {model_name} on {dataset_name}: {str(e)}")
                    problematic_combinations.append(f"{model_name}-{dataset_name}: {str(e)}")
                    
                    # Additional debugging information
                    print(f"    Debug info:")
                    print(f"      Model type: {type(model)}")
                    print(f"      Has named_steps: {hasattr(model, 'named_steps')}")
                    if hasattr(model, 'named_steps'):
                        print(f"      Steps: {list(model.named_steps.keys())}")
                    print(f"      Input shape: {input_df.shape}")
                    print(f"      Feature names: {len(test_data['feature_names'])}")
        
        if problematic_combinations:
            print(f"\n⚠️  Found {len(problematic_combinations)} issues:")
            for issue in problematic_combinations:
                print(f"  - {issue}")
        else:
            print("\n✅ All SHAP explanations working correctly!")
        
        return problematic_combinations
    
    def test_lime_explanations(self):
        """Test LIME explanations for all models"""
        print("\n=== Testing LIME Explanations ===")
        
        problematic_combinations = []
        
        for model_name, model_data in self.trained_models.items():
            for dataset_name, data in model_data.items():
                print(f"Testing LIME for {model_name} on {dataset_name}...")
                
                model = data['model']
                test_data = data['data']
                sample_input = test_data['X_test'].iloc[0:1]
                
                try:
                    # Wrapper function for LIME
                    def model_predict_proba_wrapper(X):
                        X_df = self.ensure_dataframe(X, test_data['feature_names'])
                        return model.predict_proba(X_df)
                    
                    # Create LIME explainer
                    explainer = lime.lime_tabular.LimeTabularExplainer(
                        test_data['X_train'].values,
                        feature_names=test_data['feature_names'],
                        class_names=test_data['target_names'],
                        discretize_continuous=True
                    )
                    
                    # Generate explanation
                    explanation = explainer.explain_instance(
                        sample_input.values[0],
                        model_predict_proba_wrapper,
                        num_features=len(test_data['feature_names'])
                    )
                    
                    # Validate explanation
                    exp_list = explanation.as_list()
                    assert len(exp_list) > 0, "No explanation generated"
                    
                    print(f"  ✓ {model_name} on {dataset_name}: {len(exp_list)} feature explanations")
                    
                except Exception as e:
                    print(f"  ❌ LIME failed for {model_name} on {dataset_name}: {str(e)}")
                    problematic_combinations.append(f"{model_name}-{dataset_name}: {str(e)}")
        
        if problematic_combinations:
            print(f"\n⚠️  Found {len(problematic_combinations)} LIME issues:")
            for issue in problematic_combinations:
                print(f"  - {issue}")
        else:
            print("\n✅ All LIME explanations working correctly!")
        
        return problematic_combinations
    
    def test_feature_importance(self):
        """Test feature importance for tree-based models"""
        print("\n=== Testing Feature Importance ===")
        
        for model_name, model_data in self.trained_models.items():
            for dataset_name, data in model_data.items():
                model = data['model']
                test_data = data['data']
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    assert len(importances) == len(test_data['feature_names']), "Feature importance length mismatch"
                    assert (importances >= 0).all(), "Negative feature importances found"
                    print(f"  ✓ {model_name} on {dataset_name}: Feature importance available")
                else:
                    print(f"  ℹ️  {model_name} on {dataset_name}: No feature importance (expected for pipelines)")
        
        print("✅ Feature importance tests completed!")
    
    def run_comprehensive_test(self):
        """Run all tests in sequence"""
        print("🚀 Starting Comprehensive ML Evaluation App Tests")
        print("=" * 60)
        
        try:
            # Run all tests
            self.test_dataset_loading()
            self.test_model_training()
            self.test_predictions()
            
            shap_issues = self.test_shap_explanations()
            lime_issues = self.test_lime_explanations()
            self.test_feature_importance()
            
            # Summary
            print("\n" + "=" * 60)
            print("🏁 TEST SUMMARY")
            print("=" * 60)
            
            total_issues = len(shap_issues) + len(lime_issues)
            
            if total_issues == 0:
                print("✅ ALL TESTS PASSED! 🎉")
                print("   The ML Evaluation App is fully functional.")
            else:
                print(f"⚠️  FOUND {total_issues} ISSUES")
                print("\nSHAP Issues:")
                for issue in shap_issues:
                    print(f"  - {issue}")
                print("\nLIME Issues:")
                for issue in lime_issues:
                    print(f"  - {issue}")
                
                print("\n📋 RECOMMENDATIONS:")
                if any("gradient_boosting" in issue for issue in shap_issues):
                    print("  - Check Gradient Boosting SHAP implementation")
                if any("svm" in issue for issue in shap_issues):
                    print("  - Verify SVM pipeline SHAP handling")
                print("  - Review error logs for detailed debugging information")
            
            return total_issues == 0
            
        except Exception as e:
            print(f"\n❌ CRITICAL TEST FAILURE: {str(e)}")
            raise


def main():
    """Main test runner"""
    print("ML Evaluation App - Comprehensive Test Suite")
    print("=" * 50)
    
    # Initialize test suite
    test_suite = TestMLEvaluationApp()
    test_suite.setup_class()
    
    # Run comprehensive tests
    success = test_suite.run_comprehensive_test()
    
    if success:
        print("\n🎯 All tests passed! The application is ready for use.")
        exit(0)
    else:
        print(f"\n🔧 Some issues found. Check the output above for details.")
        exit(1)


if __name__ == "__main__":
    main()
