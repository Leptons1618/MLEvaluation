"""
Test suite for enhanced UI components
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from components.enhanced_ui_components import (
    apply_column_type_changes,
    apply_imputation,
    create_combined_features
)
from utils.data_preparation import analyze_dataset


class TestColumnTypeChanges:
    """Test column type conversion functionality"""
    
    def setup_method(self):
        """Setup test data"""
        self.df = pd.DataFrame({
            'numeric_string': ['1', '2', '3', '4', '5'],
            'boolean_string': ['True', 'False', 'True', 'False', 'True'],
            'date_string': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
            'category_data': ['A', 'B', 'A', 'C', 'B'],
            'float_data': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
    
    def test_numeric_conversion(self):
        """Test conversion to numeric types"""
        type_changes = {'numeric_string': 'float64'}
        df_result = apply_column_type_changes(self.df, type_changes)
        
        assert df_result['numeric_string'].dtype == 'float64'
        assert df_result['numeric_string'].iloc[0] == 1.0
    
    def test_integer_conversion(self):
        """Test conversion to integer type"""
        type_changes = {'numeric_string': 'int64'}
        df_result = apply_column_type_changes(self.df, type_changes)
        
        assert df_result['numeric_string'].dtype == 'Int64'  # Nullable integer
        assert df_result['numeric_string'].iloc[0] == 1
    
    def test_boolean_conversion(self):
        """Test conversion to boolean type"""
        type_changes = {'boolean_string': 'boolean'}
        df_result = apply_column_type_changes(self.df, type_changes)
        
        assert df_result['boolean_string'].dtype == 'boolean'
        assert df_result['boolean_string'].iloc[0] == True
    
    def test_datetime_conversion(self):
        """Test conversion to datetime type"""
        type_changes = {'date_string': 'datetime64'}
        df_result = apply_column_type_changes(self.df, type_changes)
        
        assert pd.api.types.is_datetime64_any_dtype(df_result['date_string'])
        assert df_result['date_string'].iloc[0] == pd.Timestamp('2023-01-01')
    
    def test_category_conversion(self):
        """Test conversion to category type"""
        type_changes = {'category_data': 'category'}
        df_result = apply_column_type_changes(self.df, type_changes)
        
        assert df_result['category_data'].dtype.name == 'category'
        assert 'A' in df_result['category_data'].cat.categories
    
    def test_multiple_conversions(self):
        """Test multiple type conversions at once"""
        type_changes = {
            'numeric_string': 'int64',
            'boolean_string': 'boolean',
            'category_data': 'category'
        }
        df_result = apply_column_type_changes(self.df, type_changes)
        
        assert df_result['numeric_string'].dtype == 'Int64'
        assert df_result['boolean_string'].dtype == 'boolean'
        assert df_result['category_data'].dtype.name == 'category'
    
    def test_invalid_conversion_handling(self):
        """Test handling of invalid conversions"""
        df_invalid = pd.DataFrame({
            'text_data': ['hello', 'world', 'test', 'data', 'invalid']
        })
        
        type_changes = {'text_data': 'int64'}
        # Should not raise error, but conversion should fail gracefully
        df_result = apply_column_type_changes(df_invalid, type_changes)
        
        # Original data should remain if conversion fails
        assert len(df_result) == len(df_invalid)


class TestImputation:
    """Test missing value imputation functionality"""
    
    def setup_method(self):
        """Setup test data with missing values"""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'numeric_col': [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0],
            'categorical_col': ['A', 'B', None, 'A', 'B', None, 'C'],
            'integer_col': [1, 2, np.nan, 4, 5, np.nan, 7],
            'target': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X']
        })
    
    def test_mean_imputation(self):
        """Test mean imputation for numeric data"""
        config = {'numeric_col': {'method': 'Mean', 'custom_value': None}}
        df_result = apply_imputation(self.df, config)
        
        # Check that no missing values remain
        assert df_result['numeric_col'].isnull().sum() == 0
        
        # Check that mean was used for imputation
        original_mean = self.df['numeric_col'].mean()
        imputed_values = df_result.loc[self.df['numeric_col'].isnull(), 'numeric_col']
        assert all(val == original_mean for val in imputed_values)
    
    def test_median_imputation(self):
        """Test median imputation for numeric data"""
        config = {'numeric_col': {'method': 'Median', 'custom_value': None}}
        df_result = apply_imputation(self.df, config)
        
        assert df_result['numeric_col'].isnull().sum() == 0
        
        original_median = self.df['numeric_col'].median()
        imputed_values = df_result.loc[self.df['numeric_col'].isnull(), 'numeric_col']
        assert all(val == original_median for val in imputed_values)
    
    def test_mode_imputation(self):
        """Test mode imputation for categorical data"""
        config = {'categorical_col': {'method': 'Mode', 'custom_value': None}}
        df_result = apply_imputation(self.df, config)
        
        assert df_result['categorical_col'].isnull().sum() == 0
        
        original_mode = self.df['categorical_col'].mode().iloc[0]
        imputed_values = df_result.loc[self.df['categorical_col'].isnull(), 'categorical_col']
        assert all(val == original_mode for val in imputed_values)
    
    def test_custom_value_imputation(self):
        """Test custom value imputation"""
        custom_val = 'MISSING'
        config = {'categorical_col': {'method': 'Custom Value', 'custom_value': custom_val}}
        df_result = apply_imputation(self.df, config)
        
        assert df_result['categorical_col'].isnull().sum() == 0
        
        imputed_values = df_result.loc[self.df['categorical_col'].isnull(), 'categorical_col']
        assert all(val == custom_val for val in imputed_values)
    
    def test_forward_fill_imputation(self):
        """Test forward fill imputation"""
        config = {'numeric_col': {'method': 'Forward Fill', 'custom_value': None}}
        df_result = apply_imputation(self.df, config)
        
        # Should have fewer missing values (forward fill may not fill all)
        assert df_result['numeric_col'].isnull().sum() <= self.df['numeric_col'].isnull().sum()
    
    def test_drop_rows_imputation(self):
        """Test drop rows imputation"""
        config = {'categorical_col': {'method': 'Drop Rows', 'custom_value': None}}
        df_result = apply_imputation(self.df, config)
        
        # Should have fewer rows
        assert len(df_result) < len(self.df)
        
        # Should have no missing values in the target column
        assert df_result['categorical_col'].isnull().sum() == 0
    
    def test_multiple_column_imputation(self):
        """Test imputation on multiple columns"""
        config = {
            'numeric_col': {'method': 'Mean', 'custom_value': None},
            'categorical_col': {'method': 'Mode', 'custom_value': None}
        }
        df_result = apply_imputation(self.df, config)
        
        assert df_result['numeric_col'].isnull().sum() == 0
        assert df_result['categorical_col'].isnull().sum() == 0


class TestFeatureCombination:
    """Test feature combination functionality"""
    
    def setup_method(self):
        """Setup test data"""
        self.df = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [2, 4, 6, 8, 10],
            'feature_3': [0.5, 1.0, 1.5, 2.0, 2.5],
            'target': ['A', 'B', 'A', 'B', 'A']
        })
        
        self.feature_pairs = [('feature_1', 'feature_2'), ('feature_1', 'feature_3')]
    
    def test_sum_combination(self):
        """Test sum feature combination"""
        df_result = create_combined_features(self.df, self.feature_pairs, 'Sum')
        
        # Check new columns exist
        assert 'feature_1_plus_feature_2' in df_result.columns
        assert 'feature_1_plus_feature_3' in df_result.columns
        
        # Check values are correct
        assert df_result['feature_1_plus_feature_2'].iloc[0] == 3  # 1 + 2
        assert df_result['feature_1_plus_feature_3'].iloc[0] == 1.5  # 1 + 0.5
    
    def test_difference_combination(self):
        """Test difference feature combination"""
        df_result = create_combined_features(self.df, self.feature_pairs, 'Difference')
        
        assert 'feature_1_minus_feature_2' in df_result.columns
        assert df_result['feature_1_minus_feature_2'].iloc[0] == -1  # 1 - 2
    
    def test_product_combination(self):
        """Test product feature combination"""
        df_result = create_combined_features(self.df, self.feature_pairs, 'Product')
        
        assert 'feature_1_times_feature_2' in df_result.columns
        assert df_result['feature_1_times_feature_2'].iloc[0] == 2  # 1 * 2
    
    def test_ratio_combination(self):
        """Test ratio feature combination"""
        df_result = create_combined_features(self.df, self.feature_pairs, 'Ratio')
        
        assert 'feature_1_div_feature_2' in df_result.columns
        # Check division with small epsilon to avoid division by zero
        expected_ratio = 1 / (2 + 1e-8)
        assert abs(df_result['feature_1_div_feature_2'].iloc[0] - expected_ratio) < 1e-7
    
    def test_all_combinations(self):
        """Test all feature combinations at once"""
        df_result = create_combined_features(self.df, self.feature_pairs, 'All')
        
        # Should have all types of combinations
        expected_new_cols = [
            'feature_1_plus_feature_2', 'feature_1_minus_feature_2',
            'feature_1_times_feature_2', 'feature_1_div_feature_2',
            'feature_1_plus_feature_3', 'feature_1_minus_feature_3',
            'feature_1_times_feature_3', 'feature_1_div_feature_3'
        ]
        
        for col in expected_new_cols:
            assert col in df_result.columns
        
        # Original columns should still exist
        for col in self.df.columns:
            assert col in df_result.columns


class TestDatasetAnalysis:
    """Test dataset analysis functionality"""
    
    def setup_method(self):
        """Setup test data"""
        self.df = pd.DataFrame({
            'numeric_int': [1, 2, 3, 4, 5],
            'numeric_float': [1.1, 2.2, 3.3, 4.4, 5.5],
            'categorical': ['A', 'B', 'C', 'A', 'B'],
            'boolean': [True, False, True, False, True],
            'datetime': pd.date_range('2023-01-01', periods=5),
            'mixed_with_missing': [1, 'text', 3, np.nan, 5]
        })
    
    def test_basic_analysis(self):
        """Test basic dataset analysis"""
        analysis = analyze_dataset(self.df)
        
        # Check basic properties
        assert analysis['shape'] == (5, 6)
        assert len(analysis['columns']) == 6
        assert 'numeric_int' in analysis['numeric_columns']
        assert 'categorical' in analysis['categorical_columns']
        assert 'datetime' in analysis['datetime_columns']
    
    def test_missing_values_detection(self):
        """Test missing values detection"""
        analysis = analyze_dataset(self.df)
        
        # Should detect missing value in mixed column
        assert analysis['missing_values']['mixed_with_missing'] > 0
        assert analysis['missing_values']['numeric_int'] == 0
    
    def test_duplicates_detection(self):
        """Test duplicate detection"""
        df_with_dups = pd.concat([self.df, self.df.iloc[[0, 1]]], ignore_index=True)
        analysis = analyze_dataset(df_with_dups)
        
        assert analysis['duplicates'] == 2
    
    def test_memory_usage(self):
        """Test memory usage calculation"""
        analysis = analyze_dataset(self.df)
        
        assert 'memory_usage' in analysis
        assert analysis['memory_usage'] > 0
    
    def test_categorical_info(self):
        """Test categorical information extraction"""
        analysis = analyze_dataset(self.df)
        
        assert 'categorical_info' in analysis
        assert 'categorical' in analysis['categorical_info']
        
        cat_info = analysis['categorical_info']['categorical']
        assert cat_info['unique_count'] == 3
        assert 'A' in cat_info['top_values']


class TestIntegration:
    """Integration tests for enhanced features"""
    
    def test_full_pipeline(self):
        """Test full data preparation pipeline"""
        # Create test data
        df = pd.DataFrame({
            'numeric_str': ['1', '2', '3', np.nan, '5'],
            'categorical': ['A', 'B', None, 'A', 'B'],
            'target': ['X', 'Y', 'X', 'Y', 'X']
        })
        
        # Step 1: Type conversion
        type_changes = {'numeric_str': 'float64'}
        df = apply_column_type_changes(df, type_changes)
        assert df['numeric_str'].dtype == 'float64'
        
        # Step 2: Imputation
        imputation_config = {
            'numeric_str': {'method': 'Mean', 'custom_value': None},
            'categorical': {'method': 'Mode', 'custom_value': None}
        }
        df = apply_imputation(df, imputation_config)
        assert df.isnull().sum().sum() == 0
        
        # Step 3: Feature creation
        feature_pairs = [('numeric_str', 'numeric_str')]  # Create squared feature
        df = create_combined_features(df, feature_pairs, 'Product')
        assert 'numeric_str_times_numeric_str' in df.columns
        
        # Final validation
        assert len(df) > 0
        assert df.isnull().sum().sum() == 0
    
    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        
        # Should handle empty dataframe gracefully
        try:
            apply_column_type_changes(empty_df, {})
            apply_imputation(empty_df, {})
            create_combined_features(empty_df, [], 'Sum')
        except Exception as e:
            pytest.fail(f"Functions should handle empty dataframe gracefully: {e}")
        
        # Test with invalid configurations
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        # Invalid type change
        result = apply_column_type_changes(df, {'nonexistent_col': 'int64'})
        assert len(result) == len(df)  # Should not crash
        
        # Invalid imputation config
        result = apply_imputation(df, {'nonexistent_col': {'method': 'Mean', 'custom_value': None}})
        assert len(result) == len(df)  # Should not crash


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
