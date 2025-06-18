"""
Comprehensive tests for enhanced data preparation functionality
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.data_preparation_enhanced import (
    DataPreparationTools,
    create_enhanced_sample_datasets,
    apply_auto_preparation,
    get_preparation_recommendations
)

class TestDataPreparationTools:
    """Test enhanced data preparation tools"""
    
    def setup_method(self):
        """Setup test data and preparation tools"""
        self.prep_tools = DataPreparationTools()
        
        # Create test data with various issues
        np.random.seed(42)
        self.test_df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(10, 5, 100),
            'categorical': np.random.choice(['A', 'B', 'C'], 100),
            'high_cardinality': [f'cat_{i}' for i in np.random.choice(range(60), 100)],
            'target': np.random.choice([0, 1, 2], 100, p=[0.7, 0.2, 0.1])
        })
        
        # Introduce data quality issues
        # Missing values
        self.test_df.loc[10:15, 'feature1'] = np.nan
        self.test_df.loc[50:75, 'feature2'] = np.nan
        
        # Duplicate rows
        duplicate_rows = self.test_df.iloc[:3].copy()
        self.test_df = pd.concat([self.test_df, duplicate_rows], ignore_index=True)
        
        # Missing target values
        self.test_df.loc[90:92, 'target'] = np.nan
    
    def test_analyze_preparation_needs(self):
        """Test preparation needs analysis"""
        analysis = self.prep_tools.analyze_preparation_needs(self.test_df, 'target')
        
        # Should identify various issues
        assert len(analysis['issues']) > 0
        assert len(analysis['suggestions']) > 0
        assert len(analysis['auto_fixes']) > 0
        
        # Should detect missing target values
        assert any('Target column has' in issue for issue in analysis['issues'])
        
        # Should detect missing feature values
        assert any('missing values' in issue for issue in analysis['issues'])
        
        # Should detect duplicates
        assert any('duplicate' in issue for issue in analysis['issues'])
        
        # Should detect class imbalance
        assert any('imbalance' in issue for issue in analysis['issues'])
    
    def test_auto_preparation_missing_values(self):
        """Test automatic preparation for missing values"""
        # Create data with missing values
        df_missing = pd.DataFrame({
            'numeric_col': [1, 2, np.nan, 4, 5],
            'categorical_col': ['A', 'B', np.nan, 'C', 'D'],
            'target': [0, 1, 0, 1, 0]
        })
        
        result = self.prep_tools.auto_prepare_data(df_missing, 'target', ['impute_missing_values'])
        
        # Should handle missing values
        assert result['X_processed'].isnull().sum().sum() == 0
        assert len(result['preparation_log']) > 0
        assert 'imputation' in str(result['preparation_log']).lower()
    
    def test_auto_preparation_duplicates(self):
        """Test automatic preparation for duplicate removal"""
        # Create data with duplicates
        df_duplicates = pd.DataFrame({
            'feature1': [1, 2, 3, 1, 2],
            'feature2': [10, 20, 30, 10, 20],
            'target': [0, 1, 0, 0, 1]
        })
        
        result = self.prep_tools.auto_prepare_data(df_duplicates, 'target', ['remove_duplicates'])
        
        # Should remove duplicates
        original_rows = len(df_duplicates)
        processed_rows = len(result['X_processed']) + len(result['y_processed'])
        assert processed_rows <= original_rows
        assert 'duplicate' in str(result['preparation_log']).lower()
    
    def test_auto_preparation_small_classes(self):
        """Test handling of small classes that cause stratification errors"""
        # Create data with very small classes
        df_small_classes = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 20),
            'feature2': np.random.normal(0, 1, 20),
            'target': [0] * 18 + [1] * 1 + [2] * 1  # Classes with only 1 sample each
        })
        
        result = self.prep_tools.auto_prepare_data(
            df_small_classes, 'target', ['handle_small_classes']
        )
        
        # Should handle small classes without errors
        assert result is not None
        assert len(result['preparation_log']) > 0
    
    def test_feature_scaling(self):
        """Test feature scaling functionality"""
        # Create data with different scales
        df_scaling = pd.DataFrame({
            'small_scale': np.random.normal(0, 1, 100),
            'large_scale': np.random.normal(1000, 500, 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        result = self.prep_tools.auto_prepare_data(df_scaling, 'target', ['scale_features'])
        
        # Features should be scaled to similar ranges
        X_scaled = result['X_processed']
        small_std = X_scaled['small_scale'].std()
        large_std = X_scaled['large_scale'].std()
        
        # After scaling, standard deviations should be similar
        assert abs(small_std - large_std) < 1.0
        assert 'scaling' in str(result['preparation_log']).lower()
    
    def test_class_balancing(self):
        """Test class balancing functionality"""
        # Create imbalanced data
        df_imbalanced = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'target': [0] * 90 + [1] * 10  # 9:1 imbalance
        })
        
        # Test with balancing (if available)
        result = self.prep_tools.auto_prepare_data(
            df_imbalanced, 'target', ['balance_classes']
        )
        
        # Should attempt balancing
        assert result is not None
        assert len(result['preparation_log']) > 0
    
    def test_feature_selection(self):
        """Test feature selection functionality"""
        # Create data with many features
        np.random.seed(42)
        feature_data = {f'feature_{i}': np.random.normal(0, 1, 100) for i in range(25)}
        feature_data['target'] = np.random.choice([0, 1], 100)
        df_many_features = pd.DataFrame(feature_data)
        
        result = self.prep_tools.auto_prepare_data(
            df_many_features, 'target', ['select_features']
        )
        
        # Should reduce number of features
        original_features = len(df_many_features.columns) - 1  # Exclude target
        selected_features = len(result['X_processed'].columns)
        
        assert selected_features <= original_features
        assert 'feature selection' in str(result['preparation_log']).lower()
    
    def test_preparation_recommendations(self):
        """Test preparation recommendations"""
        # Test with different types of data issues
        
        # High missing values
        df_high_missing = pd.DataFrame({
            'feature1': [np.nan] * 80 + [1] * 20,
            'feature2': np.random.normal(0, 1, 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        recommendations = get_preparation_recommendations(df_high_missing, 'target')
        
        assert len(recommendations) > 0
        assert any('missing' in rec.lower() for rec in recommendations)
    
    def test_preparation_logging(self):
        """Test preparation operation logging"""
        original_history_length = len(self.prep_tools.preparation_history)
        
        # Perform some preparation
        self.prep_tools.auto_prepare_data(
            self.test_df, 'target', ['remove_duplicates', 'impute_missing_values']
        )
        
        # History should be updated
        assert len(self.prep_tools.preparation_history) > original_history_length
    
    def test_error_handling(self):
        """Test error handling in preparation tools"""
        # Test with invalid target column
        with pytest.raises(Exception):
            self.prep_tools.analyze_preparation_needs(self.test_df, 'nonexistent_target')
        
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        with pytest.raises(Exception):
            self.prep_tools.analyze_preparation_needs(empty_df, 'target')
        
        # Test with invalid auto_fixes
        result = self.prep_tools.auto_prepare_data(
            self.test_df, 'target', ['invalid_fix']
        )
        # Should handle gracefully without crashing
        assert result is not None


class TestEnhancedSampleDatasets:
    """Test enhanced sample dataset creation"""
    
    def test_create_enhanced_sample_datasets(self):
        """Test creation of enhanced sample datasets"""
        datasets = create_enhanced_sample_datasets()
        
        # Should create multiple datasets
        assert len(datasets) > 0
        
        # Each dataset should have required structure
        for name, dataset in datasets.items():
            assert 'data' in dataset
            assert 'description' in dataset
            assert 'target_column' in dataset
            assert 'issues' in dataset
            
            # Data should be a DataFrame
            assert isinstance(dataset['data'], pd.DataFrame)
            
            # Should have target column
            assert dataset['target_column'] in dataset['data'].columns
            
            # Should have some issues identified
            assert len(dataset['issues']) > 0
    
    def test_dataset_quality_issues(self):
        """Test that created datasets have intended quality issues"""
        datasets = create_enhanced_sample_datasets()
        
        for name, dataset in datasets.items():
            df = dataset['data']
            
            # Check for missing values (some datasets should have them)
            if 'missing' in dataset['issues']:
                assert df.isnull().sum().sum() > 0
            
            # Check for duplicates (some datasets should have them)
            if 'duplicates' in dataset['issues']:
                assert df.duplicated().sum() > 0
            
            # Check for class imbalance (some datasets should have it)
            if 'imbalance' in dataset['issues']:
                target_col = dataset['target_column']
                class_counts = df[target_col].value_counts()
                max_count = class_counts.max()
                min_count = class_counts.min()
                assert max_count / min_count > 2  # At least 2:1 imbalance


class TestAutoPreparationFunction:
    """Test the standalone auto preparation function"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.test_df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'feature2': np.random.normal(10, 2, 50),
            'categorical': np.random.choice(['A', 'B', 'C'], 50),
            'target': np.random.choice([0, 1], 50)
        })
        
        # Add missing values
        self.test_df.loc[5:8, 'feature1'] = np.nan
        self.test_df.loc[15:18, 'categorical'] = np.nan
    
    def test_apply_auto_preparation(self):
        """Test apply_auto_preparation function"""
        result = apply_auto_preparation(self.test_df, 'target')
        
        # Should return processed data
        assert 'X_train' in result
        assert 'X_test' in result
        assert 'y_train' in result
        assert 'y_test' in result
        assert 'preparation_log' in result
        assert 'scaler' in result
        
        # Data shapes should be consistent
        assert len(result['X_train']) == len(result['y_train'])
        assert len(result['X_test']) == len(result['y_test'])
        
        # Should handle missing values
        assert result['X_train'].isnull().sum().sum() == 0
        assert result['X_test'].isnull().sum().sum() == 0
    
    def test_auto_preparation_with_options(self):
        """Test auto preparation with specific options"""
        result = apply_auto_preparation(
            self.test_df, 
            'target', 
            test_size=0.3,
            random_state=42,
            apply_scaling=True,
            handle_missing=True
        )
        
        # Should apply specified options
        total_samples = len(self.test_df)
        test_samples = len(result['X_test'])
        test_ratio = test_samples / total_samples
        
        # Test size should be approximately 0.3
        assert abs(test_ratio - 0.3) < 0.1
        
        # Should have scaler if scaling was applied
        if result.get('scaler') is not None:
            assert hasattr(result['scaler'], 'transform')


class TestIntegrationScenarios:
    """Integration tests for realistic data preparation scenarios"""
    
    def test_typical_ml_workflow(self):
        """Test a typical ML workflow with data preparation"""
        # Create realistic dataset
        np.random.seed(42)
        n_samples = 200
        
        # Simulate customer data with various issues
        df = pd.DataFrame({
            'customer_id': range(n_samples),
            'age': np.random.normal(35, 10, n_samples),
            'income': np.random.exponential(50000, n_samples),
            'credit_score': np.random.normal(650, 100, n_samples),
            'product_category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'purchased': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        })
        
        # Introduce realistic data issues
        # Missing income for some customers
        missing_income = np.random.choice(n_samples, 20, replace=False)
        df.loc[missing_income, 'income'] = np.nan
        
        # Missing credit score for some customers
        missing_credit = np.random.choice(n_samples, 15, replace=False)
        df.loc[missing_credit, 'credit_score'] = np.nan
        
        # Some duplicate customers
        duplicates = df.iloc[:5].copy()
        df = pd.concat([df, duplicates], ignore_index=True)
        
        # Test full preparation workflow
        prep_tools = DataPreparationTools()
        
        # Step 1: Analyze issues
        analysis = prep_tools.analyze_preparation_needs(df, 'purchased')
        
        assert len(analysis['issues']) > 0
        assert 'missing values' in str(analysis['issues']).lower()
        assert 'duplicate' in str(analysis['issues']).lower()
        
        # Step 2: Apply auto preparation
        result = prep_tools.auto_prepare_data(
            df, 'purchased', analysis['auto_fixes']
        )
        
        # Should successfully prepare data
        assert result is not None
        assert len(result['X_processed']) > 0
        assert len(result['y_processed']) > 0
        assert len(result['preparation_log']) > 0
        
        # Data should be clean
        assert result['X_processed'].isnull().sum().sum() == 0
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        prep_tools = DataPreparationTools()
        
        # Test with single row
        single_row_df = pd.DataFrame({
            'feature': [1],
            'target': [0]
        })
        
        try:
            analysis = prep_tools.analyze_preparation_needs(single_row_df, 'target')
            # Should handle gracefully
            assert isinstance(analysis, dict)
        except Exception as e:
            # Should not crash catastrophically
            assert "target" in str(e).lower() or "sample" in str(e).lower()
        
        # Test with all missing target
        all_missing_target = pd.DataFrame({
            'feature': [1, 2, 3],
            'target': [np.nan, np.nan, np.nan]
        })
        
        analysis = prep_tools.analyze_preparation_needs(all_missing_target, 'target')
        assert 'Target column has' in str(analysis['issues'])
        
        # Test with constant features
        constant_features = pd.DataFrame({
            'constant_feature': [1, 1, 1, 1, 1],
            'target': [0, 1, 0, 1, 0]
        })
        
        analysis = prep_tools.analyze_preparation_needs(constant_features, 'target')
        # Should handle without errors
        assert isinstance(analysis, dict)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
