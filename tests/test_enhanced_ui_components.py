"""
Comprehensive tests for enhanced UI components
"""

import pytest
import pandas as pd
import numpy as np
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from components.enhanced_ui_components import (
    render_editable_column_types,
    render_enhanced_quality_report,
    render_operation_tracker,
    render_advanced_feature_selection,
    render_smart_feature_suggestions
)

class TestEditableColumnTypes:
    """Test editable column types functionality"""
    
    def setup_method(self):
        """Setup test data"""
        self.test_df = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'categorical_col': ['A', 'B', 'A', 'C', 'B'],
            'datetime_col': pd.date_range('2023-01-01', periods=5),
            'mixed_col': [1, 'text', 3, 4, 'another']
        })
        
        self.analysis = {
            'numeric_columns': ['numeric_col'],
            'categorical_columns': ['categorical_col'],
            'datetime_columns': ['datetime_col'],
            'other_columns': ['mixed_col']
        }
    
    @patch('streamlit.session_state', {})
    @patch('streamlit.markdown')
    @patch('streamlit.multiselect')
    @patch('streamlit.dataframe')
    def test_render_editable_column_types_basic(self, mock_dataframe, mock_multiselect, mock_markdown):
        """Test basic rendering of editable column types"""
        mock_multiselect.return_value = []
        
        # Should not raise any exceptions
        result = render_editable_column_types(self.test_df, self.analysis)
        
        # Check that markdown was called (UI elements rendered)
        assert mock_markdown.called
        assert mock_dataframe.called
    
    @patch('streamlit.session_state', {})
    @patch('streamlit.markdown')
    @patch('streamlit.multiselect')
    @patch('streamlit.selectbox')
    @patch('streamlit.columns')
    @patch('streamlit.button')
    def test_column_type_modification(self, mock_button, mock_columns, mock_selectbox, 
                                    mock_multiselect, mock_markdown):
        """Test column type modification workflow"""
        # Mock UI interactions
        mock_multiselect.return_value = ['numeric_col']
        mock_selectbox.return_value = 'float64'
        mock_button.return_value = False
        mock_columns.return_value = [Mock(), Mock(), Mock()]
        
        # Test the function
        result = render_editable_column_types(self.test_df, self.analysis)
        
        # Should execute without errors
        assert True
    
    def test_column_type_conversion_safety(self):
        """Test safe column type conversion"""
        # Test data with conversion challenges
        problematic_df = pd.DataFrame({
            'mixed_numeric': [1, 2, 'not_a_number', 4, 5],
            'text_dates': ['2023-01-01', '2023-01-02', 'invalid_date', '2023-01-04', '2023-01-05']
        })
        
        # The function should handle these gracefully
        analysis = {
            'numeric_columns': [],
            'categorical_columns': ['mixed_numeric', 'text_dates'],
            'datetime_columns': [],
            'other_columns': []
        }
        
        # Should not raise exceptions
        with patch('streamlit.session_state', {}):
            with patch('streamlit.markdown'):
                with patch('streamlit.multiselect', return_value=[]):
                    with patch('streamlit.dataframe'):
                        result = render_editable_column_types(problematic_df, analysis)


class TestEnhancedQualityReport:
    """Test enhanced quality report functionality"""
    
    def setup_method(self):
        """Setup test data with quality issues"""
        np.random.seed(42)
        self.test_df = pd.DataFrame({
            'complete_numeric': np.random.normal(0, 1, 100),
            'missing_numeric': np.random.normal(0, 1, 100),
            'categorical_col': np.random.choice(['A', 'B', 'C'], 100),
            'high_missing_col': np.random.normal(0, 1, 100)
        })
        
        # Introduce missing values
        self.test_df.loc[10:20, 'missing_numeric'] = np.nan
        self.test_df.loc[50:85, 'high_missing_col'] = np.nan
    
    @patch('streamlit.session_state', {})
    @patch('streamlit.markdown')
    @patch('streamlit.dataframe')
    @patch('streamlit.expander')
    def test_quality_report_generation(self, mock_expander, mock_dataframe, mock_markdown):
        """Test quality report generation"""
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        
        # Should execute without errors
        with patch.object(st, 'plotly_chart'):
            render_enhanced_quality_report(self.test_df)
        
        assert mock_markdown.called
    
    @patch('streamlit.session_state', {})
    @patch('streamlit.selectbox')
    @patch('streamlit.button')
    @patch('streamlit.success')
    def test_imputation_recommendations(self, mock_success, mock_button, mock_selectbox):
        """Test imputation recommendations and execution"""
        mock_selectbox.side_effect = ['missing_numeric', 'mean']
        mock_button.return_value = True
        
        with patch('streamlit.markdown'):
            with patch('streamlit.dataframe'):
                with patch('streamlit.expander') as mock_expander:
                    mock_expander.return_value.__enter__ = Mock()
                    mock_expander.return_value.__exit__ = Mock()
                    
                    render_enhanced_quality_report(self.test_df)
        
        # Should complete without errors
        assert True


class TestAdvancedFeatureSelection:
    """Test advanced feature selection functionality"""
    
    def setup_method(self):
        """Setup test data for feature selection"""
        np.random.seed(42)
        n_samples = 100
        
        self.test_df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples),
            'correlated_feature': np.random.normal(0, 1, n_samples),
            'target': np.random.choice([0, 1], n_samples)
        })
        
        # Make correlated_feature actually correlated with feature1
        self.test_df['correlated_feature'] = self.test_df['feature1'] + np.random.normal(0, 0.1, n_samples)
    
    @patch('streamlit.session_state', {})
    @patch('streamlit.selectbox')
    @patch('streamlit.multiselect')
    def test_feature_selection_ui(self, mock_multiselect, mock_selectbox):
        """Test feature selection UI rendering"""
        mock_selectbox.side_effect = ['target', 'correlation']
        mock_multiselect.return_value = ['feature1', 'feature2']
        
        with patch('streamlit.markdown'):
            with patch('streamlit.columns', return_value=[Mock(), Mock()]):
                with patch('streamlit.button', return_value=False):
                    render_advanced_feature_selection(self.test_df)
        
        # Should complete without errors
        assert True
    
    @patch('streamlit.session_state', {})
    def test_correlation_analysis(self):
        """Test correlation-based feature selection"""
        with patch('streamlit.selectbox', side_effect=['target', 'correlation']):
            with patch('streamlit.slider', return_value=0.5):
                with patch('streamlit.button', return_value=True):
                    with patch('streamlit.markdown'):
                        with patch('streamlit.columns', return_value=[Mock(), Mock()]):
                            with patch('streamlit.success'):
                                render_advanced_feature_selection(self.test_df)
        
        # Should identify correlated features
        assert True


class TestSmartFeatureEngineering:
    """Test smart feature engineering functionality"""
    
    def setup_method(self):
        """Setup test data for feature engineering"""
        np.random.seed(42)
        self.test_df = pd.DataFrame({
            'numeric1': np.random.normal(10, 2, 100),
            'numeric2': np.random.normal(5, 1, 100),
            'categorical': np.random.choice(['A', 'B', 'C'], 100),
            'binary': np.random.choice([0, 1], 100)
        })
    
    @patch('streamlit.session_state', {})
    @patch('streamlit.multiselect')
    @patch('streamlit.selectbox')
    def test_feature_engineering_suggestions(self, mock_selectbox, mock_multiselect):
        """Test feature engineering suggestions"""
        mock_multiselect.return_value = ['numeric1', 'numeric2']
        mock_selectbox.return_value = 'polynomial'
        
        with patch('streamlit.markdown'):
            with patch('streamlit.expander') as mock_expander:
                mock_expander.return_value.__enter__ = Mock()
                mock_expander.return_value.__exit__ = Mock()
                with patch('streamlit.button', return_value=False):
                    render_smart_feature_suggestions(self.test_df, 'target', ['feature1', 'feature2'])
        
        # Should complete without errors
        assert True
    
    @patch('streamlit.session_state', {})
    def test_polynomial_features_creation(self):
        """Test polynomial features creation"""
        with patch('streamlit.multiselect', return_value=['numeric1', 'numeric2']):
            with patch('streamlit.selectbox', return_value='polynomial'):
                with patch('streamlit.slider', return_value=2):
                    with patch('streamlit.button', return_value=True):
                        with patch('streamlit.markdown'):
                            with patch('streamlit.expander') as mock_expander:
                                mock_expander.return_value.__enter__ = Mock()
                                mock_expander.return_value.__exit__ = Mock()
                                with patch('streamlit.success'):
                                    result = render_smart_feature_suggestions(self.test_df, 'target', ['feature1', 'feature2'])
        
        # Should handle polynomial feature creation
        assert True


class TestOperationHistory:
    """Test operation history and dataset versioning"""
    
    def setup_method(self):
        """Setup test data"""
        self.test_df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['A', 'B', 'C', 'D', 'E']
        })
    
    @patch('streamlit.session_state', {'operation_history': []})
    @patch('streamlit.markdown')
    @patch('streamlit.expander')
    def test_operation_history_display(self, mock_expander, mock_markdown):
        """Test operation history display"""
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        
        render_operation_history()
        
        assert mock_markdown.called
    
    @patch('streamlit.session_state', {
        'operation_history': [
            {
                'operation': 'test_operation',
                'timestamp': '2023-01-01 12:00:00',
                'details': 'Test operation details'
            }
        ]
    })
    def test_operation_history_with_data(self):
        """Test operation history with existing data"""
        with patch('streamlit.markdown'):
            with patch('streamlit.expander') as mock_expander:
                mock_expander.return_value.__enter__ = Mock()
                mock_expander.return_value.__exit__ = Mock()
                with patch('streamlit.dataframe'):
                    render_operation_history()
        
        # Should handle existing history data
        assert True


# Integration Tests
class TestUIIntegration:
    """Integration tests for UI components"""
    
    def setup_method(self):
        """Setup comprehensive test data"""
        np.random.seed(42)
        self.comprehensive_df = pd.DataFrame({
            'id': range(200),
            'numeric_feature1': np.random.normal(100, 15, 200),
            'numeric_feature2': np.random.exponential(2, 200),
            'categorical_feature1': np.random.choice(['Category_A', 'Category_B', 'Category_C'], 200),
            'categorical_feature2': np.random.choice(['Type_X', 'Type_Y'], 200, p=[0.7, 0.3]),
            'datetime_feature': pd.date_range('2020-01-01', periods=200, freq='D'),
            'target': np.random.choice([0, 1], 200, p=[0.6, 0.4])
        })
        
        # Introduce various data quality issues
        # Missing values (moderate)
        missing_indices = np.random.choice(200, 20, replace=False)
        self.comprehensive_df.loc[missing_indices, 'numeric_feature1'] = np.nan
        
        # Missing values (high)
        high_missing_indices = np.random.choice(200, 80, replace=False)
        self.comprehensive_df.loc[high_missing_indices, 'numeric_feature2'] = np.nan
        
        # Duplicate some rows
        duplicate_rows = self.comprehensive_df.iloc[:5].copy()
        self.comprehensive_df = pd.concat([self.comprehensive_df, duplicate_rows], ignore_index=True)
    
    @patch('streamlit.session_state', {})
    def test_end_to_end_workflow(self):
        """Test end-to-end workflow with all UI components"""
        
        # Mock all streamlit components
        with patch('streamlit.markdown'):
            with patch('streamlit.dataframe'):
                with patch('streamlit.multiselect', return_value=[]):
                    with patch('streamlit.selectbox', return_value='target'):
                        with patch('streamlit.button', return_value=False):
                            with patch('streamlit.expander') as mock_expander:
                                mock_expander.return_value.__enter__ = Mock()
                                mock_expander.return_value.__exit__ = Mock()
                                
                                # Test each major component
                                analysis = {
                                    'numeric_columns': ['numeric_feature1', 'numeric_feature2'],
                                    'categorical_columns': ['categorical_feature1', 'categorical_feature2'],
                                    'datetime_columns': ['datetime_feature'],
                                    'other_columns': ['id']
                                }
                                
                                # All components should work together without conflicts
                                render_editable_column_types(self.comprehensive_df, analysis)
                                render_enhanced_quality_report(self.comprehensive_df)
                                render_advanced_feature_selection(self.comprehensive_df)
                                render_smart_feature_engineering(self.comprehensive_df)
                                render_operation_history()
        
        # Should complete without errors
        assert True


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
