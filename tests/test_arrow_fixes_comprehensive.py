"""
Comprehensive tests for Arrow serialization fixes and enhanced features
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from components.enhanced_ui_components import (
    safe_mean, safe_median, safe_mode, 
    apply_imputation, render_enhanced_quality_report
)
from utils.data_preparation_enhanced import (
    safe_mean as prep_safe_mean,
    safe_median as prep_safe_median, 
    safe_mode as prep_safe_mode,
    DataPreparationTools
)


class TestArrowSerializationFixes:
    """Test Arrow serialization safe calculation functions"""
    
    def setup_method(self):
        """Set up test data"""
        # Create test data with various types
        self.numeric_data = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0])
        self.integer_data = pd.Series([1, 2, 3, np.nan, 5])
        self.categorical_data = pd.Series(['A', 'B', 'A', np.nan, 'C'])
        self.empty_series = pd.Series([np.nan, np.nan, np.nan])
        
        # Test DataFrame for imputation
        self.test_df = pd.DataFrame({
            'numeric_col': [1.0, 2.0, np.nan, 4.0, 5.0],
            'integer_col': [1, 2, np.nan, 4, 5],
            'categorical_col': ['A', 'B', np.nan, 'A', 'C'],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_safe_mean_numeric(self):
        """Test safe_mean with numeric data"""
        result = safe_mean(self.numeric_data)
        expected = np.mean([1.0, 2.0, 3.0, 5.0])  # excluding NaN
        assert abs(result - expected) < 1e-10
        
        # Test with all NaN
        result_empty = safe_mean(self.empty_series)
        assert result_empty == 0.0
        
        # Test prep version
        result_prep = prep_safe_mean(self.numeric_data)
        assert abs(result_prep - expected) < 1e-10
    
    def test_safe_median_numeric(self):
        """Test safe_median with numeric data"""
        result = safe_median(self.numeric_data)
        expected = np.median([1.0, 2.0, 3.0, 5.0])  # excluding NaN
        assert abs(result - expected) < 1e-10
        
        # Test with all NaN
        result_empty = safe_median(self.empty_series)
        assert result_empty == 0.0
        
        # Test prep version
        result_prep = prep_safe_median(self.numeric_data)
        assert abs(result_prep - expected) < 1e-10
    
    def test_safe_mode_numeric(self):
        """Test safe_mode with numeric data"""
        # Create data with clear mode
        mode_data = pd.Series([1, 1, 2, 3, 1, np.nan])
        result = safe_mode(mode_data)
        assert result == 1.0
        
        # Test with all NaN
        result_empty = safe_mode(self.empty_series)
        assert result_empty == 0.0
        
        # Test prep version
        result_prep = prep_safe_mode(mode_data)
        assert result_prep == 1.0
    
    def test_safe_mode_categorical(self):
        """Test safe_mode with categorical data"""
        # Create data with clear mode
        mode_data = pd.Series(['A', 'A', 'B', 'C', 'A', np.nan])
        result = safe_mode(mode_data)
        assert result == 'A'
        
        # Test with all NaN categorical
        empty_cat = pd.Series([np.nan, np.nan], dtype='object')
        result_empty = safe_mode(empty_cat)
        assert result_empty == 'Unknown'
        
        # Test prep version
        result_prep = prep_safe_mode(mode_data)
        assert result_prep == 'A'
    
    def test_apply_imputation_comprehensive(self):
        """Test the apply_imputation function with all methods"""
        # Test Mean imputation
        config_mean = {'numeric_col': {'method': 'Mean', 'custom_value': None}}
        result_mean = apply_imputation(self.test_df, config_mean)
        assert not result_mean['numeric_col'].isnull().any()
        expected_mean = safe_mean(self.test_df['numeric_col'])
        assert result_mean['numeric_col'].iloc[2] == expected_mean
        
        # Test Median imputation
        config_median = {'numeric_col': {'method': 'Median', 'custom_value': None}}
        result_median = apply_imputation(self.test_df, config_median)
        assert not result_median['numeric_col'].isnull().any()
        expected_median = safe_median(self.test_df['numeric_col'])
        assert result_median['numeric_col'].iloc[2] == expected_median
        
        # Test Mode imputation
        config_mode = {'categorical_col': {'method': 'Mode', 'custom_value': None}}
        result_mode = apply_imputation(self.test_df, config_mode)
        assert not result_mode['categorical_col'].isnull().any()
        expected_mode = safe_mode(self.test_df['categorical_col'])
        assert result_mode['categorical_col'].iloc[2] == expected_mode
        
        # Test Custom Value imputation
        config_custom = {'categorical_col': {'method': 'Custom Value', 'custom_value': 'CUSTOM'}}
        result_custom = apply_imputation(self.test_df, config_custom)
        assert result_custom['categorical_col'].iloc[2] == 'CUSTOM'
        
        # Test Drop Rows
        config_drop = {'numeric_col': {'method': 'Drop Rows', 'custom_value': None}}
        result_drop = apply_imputation(self.test_df, config_drop)
        assert len(result_drop) == 4  # One row should be dropped
        assert not result_drop['numeric_col'].isnull().any()
    
    def test_forward_backward_fill(self):
        """Test forward and backward fill methods"""
        # Create sequential data for fill testing
        sequential_df = pd.DataFrame({
            'col1': [1, np.nan, 3, np.nan, 5],
            'target': [0, 1, 0, 1, 0]
        })
        
        # Test Forward Fill
        config_ffill = {'col1': {'method': 'Forward Fill', 'custom_value': None}}
        result_ffill = apply_imputation(sequential_df, config_ffill)
        assert result_ffill['col1'].iloc[1] == 1.0  # Forward filled
        assert result_ffill['col1'].iloc[3] == 3.0  # Forward filled
        
        # Test Backward Fill
        config_bfill = {'col1': {'method': 'Backward Fill', 'custom_value': None}}
        result_bfill = apply_imputation(sequential_df, config_bfill)
        assert result_bfill['col1'].iloc[1] == 3.0  # Backward filled
        assert result_bfill['col1'].iloc[3] == 5.0  # Backward filled
    
    def test_interpolate_method(self):
        """Test interpolation method"""
        # Create data suitable for interpolation
        interp_df = pd.DataFrame({
            'col1': [1.0, np.nan, 3.0, np.nan, 5.0],
            'target': [0, 1, 0, 1, 0]
        })
        
        config_interp = {'col1': {'method': 'Interpolate', 'custom_value': None}}
        result_interp = apply_imputation(interp_df, config_interp)
        
        # Check that values are interpolated
        assert not result_interp['col1'].isnull().any()
        assert result_interp['col1'].iloc[1] == 2.0  # Linear interpolation
        assert result_interp['col1'].iloc[3] == 4.0  # Linear interpolation
    
    def test_error_handling(self):
        """Test error handling in safe functions"""
        # Test with problematic data that might cause Arrow issues
        problematic_series = pd.Series([1, 2, 3, "invalid", np.nan])
        
        # These should not raise exceptions
        result_mean = safe_mean(problematic_series)
        assert isinstance(result_mean, float)
        
        result_median = safe_median(problematic_series)
        assert isinstance(result_median, float)
        
        result_mode = safe_mode(problematic_series)
        assert result_mode is not None


class TestDataPreparationTools:
    """Test the enhanced data preparation tools"""
    
    def setup_method(self):
        """Set up test data"""
        self.prep_tools = DataPreparationTools()
        
        # Create comprehensive test dataset
        self.test_data = pd.DataFrame({
            'numeric_feature': [1.0, 2.0, np.nan, 4.0, 5.0, 1000.0],  # Has outlier and missing
            'categorical_feature': ['A', 'B', np.nan, 'A', 'C', 'A'],  # Has missing
            'high_missing_feature': [1.0, np.nan, np.nan, np.nan, np.nan, np.nan],  # >50% missing
            'duplicate_target': [0, 1, 0, 1, 0, 0],  # Some class imbalance
            'target': [0, 1, 0, 1, 0, 0]
        })
        
        # Add duplicate row
        self.test_data = pd.concat([self.test_data, self.test_data.iloc[[0]]], ignore_index=True)
    
    def test_analyze_preparation_needs(self):
        """Test the analyze_preparation_needs function"""
        analysis = self.prep_tools.analyze_preparation_needs(self.test_data, 'target')
        
        # Check that analysis identifies issues
        assert isinstance(analysis, dict)
        assert 'issues' in analysis
        assert 'suggestions' in analysis
        assert 'auto_fixes' in analysis
        
        issues = analysis['issues']
        suggestions = analysis['suggestions']
        auto_fixes = analysis['auto_fixes']
        
        # Should identify missing values
        assert any('missing values' in issue.lower() for issue in issues)
        
        # Should identify high missing features
        assert any('high_missing_feature' in issue for issue in issues)
        
        # Should identify duplicates
        assert any('duplicate' in issue.lower() for issue in issues)
        
        # Check auto fixes are suggested
        assert 'drop_high_missing_features' in auto_fixes
        assert 'impute_missing_values' in auto_fixes
        assert 'remove_duplicates' in auto_fixes
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        # Test with single class (should not cause stratification error warning)
        single_class_df = pd.DataFrame({
            'feature': [1, 2, 3],
            'target': [0, 0, 0]
        })
        
        analysis = self.prep_tools.analyze_preparation_needs(single_class_df, 'target')
        # Should handle gracefully
        assert isinstance(analysis, dict)
        
        # Test with empty dataframe
        empty_df = pd.DataFrame(columns=['feature', 'target'])
        analysis_empty = self.prep_tools.analyze_preparation_needs(empty_df, 'target')
        assert isinstance(analysis_empty, dict)


class TestIntegration:
    """Integration tests for the enhanced features working together"""
    
    def setup_method(self):
        """Set up integration test data"""
        # Create a realistic dataset for integration testing
        np.random.seed(42)
        n_samples = 100
        
        self.integration_df = pd.DataFrame({
            'age': np.random.normal(35, 10, n_samples),
            'income': np.random.exponential(50000, n_samples),
            'category': np.random.choice(['A', 'B', 'C'], n_samples),
            'score': np.random.normal(100, 15, n_samples),
            'target': np.random.choice([0, 1], n_samples)
        })
        
        # Introduce missing values
        missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
        self.integration_df.loc[missing_indices[:5], 'age'] = np.nan
        self.integration_df.loc[missing_indices[5:], 'category'] = np.nan
    
    def test_full_preparation_pipeline(self):
        """Test the full preparation pipeline"""
        prep_tools = DataPreparationTools()
        
        # Step 1: Analyze needs
        analysis = prep_tools.analyze_preparation_needs(self.integration_df, 'target')
        assert isinstance(analysis, dict)
        
        # Step 2: Apply imputation using enhanced UI component
        missing_cols = self.integration_df.columns[self.integration_df.isnull().any()].tolist()
        
        if missing_cols:
            # Configure imputation for missing columns
            imputation_config = {}
            for col in missing_cols:
                if self.integration_df[col].dtype in ['int64', 'float64']:
                    imputation_config[col] = {'method': 'Median', 'custom_value': None}
                else:
                    imputation_config[col] = {'method': 'Mode', 'custom_value': None}
            
            # Apply imputation
            result_df = apply_imputation(self.integration_df, imputation_config)
            
            # Verify no missing values remain in imputed columns
            for col in missing_cols:
                assert not result_df[col].isnull().any(), f"Column {col} still has missing values"
            
            # Verify data integrity
            assert len(result_df) == len(self.integration_df)
            assert list(result_df.columns) == list(self.integration_df.columns)
    
    def test_arrow_compatibility(self):
        """Test that all operations are Arrow-compatible"""
        # This test ensures that the fixed functions work with Arrow backend
        
        # Test safe calculation functions
        for col in self.integration_df.select_dtypes(include=[np.number]).columns:
            if col != 'target':
                mean_val = safe_mean(self.integration_df[col])
                median_val = safe_median(self.integration_df[col])
                mode_val = safe_mode(self.integration_df[col])
                
                assert isinstance(mean_val, (int, float))
                assert isinstance(median_val, (int, float))
                assert isinstance(mode_val, (int, float))
        
        # Test with categorical columns
        for col in self.integration_df.select_dtypes(include=['object', 'category']).columns:
            if col != 'target':
                mode_val = safe_mode(self.integration_df[col])
                assert mode_val is not None


def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("🧪 Running comprehensive Arrow serialization and enhancement tests...")
    
    # Run pytest with verbose output
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "--color=yes"
    ])


if __name__ == "__main__":
    run_comprehensive_tests()
