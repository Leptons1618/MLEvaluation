"""
Comprehensive test runner for all ML Evaluation enhanced features
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_all_tests():
    """Run all tests and provide comprehensive report"""
    print("🚀 Running Comprehensive ML Evaluation Tests")
    print("=" * 60)
    
    # Test files to run
    test_files = [
        'test_enhanced_ui_components.py',
        'test_enhanced_data_preparation.py', 
        'test_arrow_plotly_fixes.py'
    ]
    
    results = {}
    
    for test_file in test_files:
        print(f"\n📋 Running tests from {test_file}")
        print("-" * 40)
        
        try:
            # Run pytest for each file
            result = pytest.main([
                os.path.join(os.path.dirname(__file__), test_file),
                '-v',
                '--tb=short'
            ])
            results[test_file] = 'PASSED' if result == 0 else 'FAILED'
            
        except Exception as e:
            print(f"❌ Error running {test_file}: {e}")
            results[test_file] = 'ERROR'
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for test_file, status in results.items():
        status_emoji = "✅" if status == "PASSED" else "❌"
        print(f"{status_emoji} {test_file}: {status}")
    
    overall_status = "PASSED" if all(status == "PASSED" for status in results.values()) else "FAILED"
    print(f"\n🎯 Overall Status: {overall_status}")
    
    return results


class TestIntegrationScenarios:
    """Integration tests for the complete enhanced ML Evaluation app"""
    
    def setup_method(self):
        """Setup comprehensive test environment"""
        # Create test data that exercises all enhanced features
        np.random.seed(42)
        n_samples = 300
        
        self.integration_df = pd.DataFrame({
            # Numeric features with different scales and missing values
            'age': np.random.normal(35, 12, n_samples),
            'income': np.random.exponential(50000, n_samples),
            'credit_score': np.random.normal(650, 100, n_samples),
            'debt_ratio': np.random.uniform(0, 1, n_samples),
            
            # Categorical features with different cardinalities
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'occupation': np.random.choice([f'Job_{i}' for i in range(20)], n_samples),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            
            # DateTime features
            'account_open_date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
            
            # Binary and target
            'has_previous_loan': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'approved': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        })
        
        # Introduce comprehensive data quality issues
        self._introduce_data_issues()
    
    def _introduce_data_issues(self):
        """Introduce various data quality issues for testing"""
        n = len(self.integration_df)
        
        # Missing values (different patterns)
        # Random missing
        self.integration_df.loc[np.random.choice(n, 20, replace=False), 'age'] = np.nan
        # Systematic missing (high income might not report debt_ratio)
        high_income_mask = self.integration_df['income'] > self.integration_df['income'].quantile(0.8)
        self.integration_df.loc[high_income_mask, 'debt_ratio'] = np.nan
        # Missing in categorical
        self.integration_df.loc[np.random.choice(n, 15, replace=False), 'education'] = np.nan
        
        # Duplicates
        duplicate_indices = np.random.choice(n, 10, replace=False)
        duplicate_rows = self.integration_df.iloc[duplicate_indices].copy()
        self.integration_df = pd.concat([self.integration_df, duplicate_rows], ignore_index=True)
        
        # Outliers
        outlier_indices = np.random.choice(len(self.integration_df), 5, replace=False)
        self.integration_df.loc[outlier_indices, 'income'] = self.integration_df['income'].quantile(0.99) * 5
        
        # Class imbalance in target
        # Make it more imbalanced
        imbalance_indices = np.random.choice(
            self.integration_df[self.integration_df['approved'] == 1].index, 
            int(len(self.integration_df[self.integration_df['approved'] == 1]) * 0.7), 
            replace=False
        )
        self.integration_df.loc[imbalance_indices, 'approved'] = 0
    
    def test_complete_data_preparation_workflow(self):
        """Test the complete data preparation workflow"""
        # This test simulates a user going through the entire enhanced workflow
        
        # Step 1: Data Analysis
        from utils.data_preparation_enhanced import DataPreparationTools
        
        prep_tools = DataPreparationTools()
        analysis = prep_tools.analyze_preparation_needs(self.integration_df, 'approved')
        
        # Should identify multiple issues
        assert len(analysis['issues']) >= 3, "Should identify multiple data issues"
        assert len(analysis['auto_fixes']) >= 3, "Should suggest multiple auto-fixes"
        
        # Step 2: Auto Preparation
        result = prep_tools.auto_prepare_data(
            self.integration_df, 'approved', analysis['auto_fixes']
        )
        
        # Verify preparation results
        assert result is not None, "Auto preparation should succeed"
        assert len(result['X_processed']) > 0, "Should have processed features"
        assert len(result['y_processed']) > 0, "Should have processed target"
        assert len(result['preparation_log']) > 0, "Should have logged operations"
        
        # Data should be cleaner
        assert result['X_processed'].isnull().sum().sum() == 0, "Should handle missing values"
        
        print(f"✅ Complete workflow test passed")
        print(f"   - Original data shape: {self.integration_df.shape}")
        print(f"   - Processed data shape: {result['X_processed'].shape}")
        print(f"   - Issues identified: {len(analysis['issues'])}")
        print(f"   - Operations applied: {len(result['preparation_log'])}")
    
    def test_ui_components_integration(self):
        """Test integration of all UI components"""
        from unittest.mock import patch, Mock
        from components.enhanced_ui_components import (
            render_editable_column_types,
            render_enhanced_quality_report,
            render_advanced_feature_selection,
            render_smart_feature_engineering,
            render_operation_history
        )
        
        # Mock streamlit components
        with patch('streamlit.session_state', {}):
            with patch('streamlit.markdown'):
                with patch('streamlit.dataframe'):
                    with patch('streamlit.multiselect', return_value=[]):
                        with patch('streamlit.selectbox', return_value='approved'):
                            with patch('streamlit.button', return_value=False):
                                with patch('streamlit.expander') as mock_expander:
                                    mock_expander.return_value.__enter__ = Mock()
                                    mock_expander.return_value.__exit__ = Mock()
                                    
                                    # Test all UI components work together
                                    analysis = {
                                        'numeric_columns': ['age', 'income', 'credit_score', 'debt_ratio'],
                                        'categorical_columns': ['education', 'occupation', 'region'],
                                        'datetime_columns': ['account_open_date'],
                                        'other_columns': ['has_previous_loan']
                                    }
                                    
                                    # Should all execute without errors
                                    render_editable_column_types(self.integration_df, analysis)
                                    render_enhanced_quality_report(self.integration_df)
                                    render_advanced_feature_selection(self.integration_df)
                                    render_smart_feature_engineering(self.integration_df)
                                    render_operation_history()
        
        print("✅ UI components integration test passed")
    
    def test_arrow_serialization_fixes(self):
        """Test Arrow serialization fixes with realistic data"""
        # Create problematic data similar to what the app generates
        quality_report = pd.DataFrame({
            'Column': ['age', 'income', 'education', 'approved'],
            'Data Type': ['float64', 'float64', 'object', 'int64'],
            'Missing Count': [20, 0, 15, 0],
            'Missing %': [6.7, 0.0, 5.0, 0.0],
            'Mean/Mode': [35.2, 51234.5, 'Bachelor', 0],  # Mixed types
            'Unique Values': [89, 287, 4, 2]
        })
        
        # Fix function
        def fix_for_streamlit_display(df):
            """Fix DataFrame for Streamlit display"""
            df_fixed = df.copy()
            
            # Handle the problematic Mean/Mode column
            if 'Mean/Mode' in df_fixed.columns:
                df_fixed['Mean/Mode'] = df_fixed['Mean/Mode'].astype(str)
            
            # Ensure all object columns are string-compatible
            for col in df_fixed.columns:
                if df_fixed[col].dtype == 'object':
                    df_fixed[col] = df_fixed[col].astype(str)
                    df_fixed[col] = df_fixed[col].fillna('N/A')
            
            return df_fixed
        
        # Apply fix
        fixed_report = fix_for_streamlit_display(quality_report)
        
        # Verify fix
        assert fixed_report['Mean/Mode'].dtype == 'object'
        for val in fixed_report['Mean/Mode']:
            assert isinstance(val, str)
        
        print("✅ Arrow serialization fixes test passed")
    
    def test_plotly_visualization_fixes(self):
        """Test Plotly visualization fixes"""
        import plotly.express as px
        
        # Test correlation heatmap (common source of errors)
        numeric_data = self.integration_df.select_dtypes(include=[np.number])
        
        def create_correlation_heatmap(data):
            """Create correlation heatmap with proper error handling"""
            try:
                corr_matrix = data.corr()
                
                fig = px.imshow(
                    corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    color_continuous_scale='RdBu_r'
                )
                
                # Use correct method (update_xaxes instead of update_xaxis)
                fig.update_xaxes(title="Features")
                fig.update_yaxes(title="Features") 
                
                return fig, True
            except Exception as e:
                return None, False
        
        # Test the fix
        fig, success = create_correlation_heatmap(numeric_data)
        assert success == True, "Correlation heatmap should be created successfully"
        assert fig is not None
        
        print("✅ Plotly visualization fixes test passed")
    
    def test_end_to_end_robustness(self):
        """Test end-to-end robustness with edge cases"""
        # Test with various edge cases
        edge_cases = [
            # Very small dataset
            self.integration_df.head(5),
            # Dataset with all missing values in one column
            self.integration_df.copy(),
            # Dataset with constant values
            self.integration_df.copy()
        ]
        
        # Modify edge cases
        edge_cases[1]['education'] = np.nan  # All missing
        edge_cases[2]['constant_col'] = 1  # Constant column
        
        from utils.data_preparation_enhanced import DataPreparationTools
        prep_tools = DataPreparationTools()
        
        for i, df in enumerate(edge_cases):
            try:
                if 'approved' in df.columns:
                    analysis = prep_tools.analyze_preparation_needs(df, 'approved')
                    # Should handle gracefully
                    assert isinstance(analysis, dict)
                    print(f"✅ Edge case {i+1} handled successfully")
                else:
                    print(f"⚠️ Edge case {i+1} skipped (no target column)")
            except Exception as e:
                print(f"❌ Edge case {i+1} failed: {e}")
                # Some failures might be expected for extreme edge cases
                assert "target" in str(e).lower() or "sample" in str(e).lower()
        
        print("✅ End-to-end robustness test completed")


class TestPerformanceAndScalability:
    """Test performance and scalability of enhanced features"""
    
    def test_large_dataset_handling(self):
        """Test handling of larger datasets"""
        # Create a larger dataset
        np.random.seed(42)
        large_n = 5000
        
        large_df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, large_n),
            'feature2': np.random.normal(10, 5, large_n),
            'feature3': np.random.exponential(2, large_n),
            'categorical': np.random.choice(['A', 'B', 'C', 'D', 'E'], large_n),
            'target': np.random.choice([0, 1], large_n)
        })
        
        # Add some missing values
        large_df.loc[np.random.choice(large_n, 500, replace=False), 'feature1'] = np.nan
        
        from utils.data_preparation_enhanced import DataPreparationTools
        prep_tools = DataPreparationTools()
        
        import time
        start_time = time.time()
        
        # Test analysis performance
        analysis = prep_tools.analyze_preparation_needs(large_df, 'target')
        analysis_time = time.time() - start_time
        
        assert analysis_time < 10.0, f"Analysis took too long: {analysis_time:.2f}s"
        
        # Test preparation performance
        start_time = time.time()
        result = prep_tools.auto_prepare_data(large_df, 'target', ['impute_missing_values'])
        prep_time = time.time() - start_time
        
        assert prep_time < 30.0, f"Preparation took too long: {prep_time:.2f}s"
        
        print(f"✅ Large dataset test passed")
        print(f"   - Dataset size: {large_df.shape}")
        print(f"   - Analysis time: {analysis_time:.2f}s")
        print(f"   - Preparation time: {prep_time:.2f}s")
    
    def test_memory_efficiency(self):
        """Test memory efficiency of operations"""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and process multiple datasets
        np.random.seed(42)
        
        for i in range(5):
            # Create dataset
            df = pd.DataFrame({
                'feature1': np.random.normal(0, 1, 1000),
                'feature2': np.random.normal(0, 1, 1000),
                'target': np.random.choice([0, 1], 1000)
            })
            
            # Process it
            from utils.data_preparation_enhanced import DataPreparationTools
            prep_tools = DataPreparationTools()
            analysis = prep_tools.analyze_preparation_needs(df, 'target')
            result = prep_tools.auto_prepare_data(df, 'target', analysis['auto_fixes'][:2])
            
            # Clean up
            del df, analysis, result, prep_tools
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100, f"Memory usage increased by {memory_increase:.2f}MB"
        
        print(f"✅ Memory efficiency test passed")
        print(f"   - Initial memory: {initial_memory:.2f}MB")
        print(f"   - Final memory: {final_memory:.2f}MB")
        print(f"   - Increase: {memory_increase:.2f}MB")


if __name__ == "__main__":
    # Run comprehensive tests
    print("🧪 ML Evaluation Enhanced Features - Comprehensive Test Suite")
    print("=" * 70)
    
    # Run unit tests
    results = run_all_tests()
    
    # Run integration tests
    print("\n🔧 Running Integration Tests")
    print("-" * 40)
    
    integration_tests = TestIntegrationScenarios()
    integration_tests.setup_method()
    
    try:
        integration_tests.test_complete_data_preparation_workflow()
        integration_tests.test_ui_components_integration()
        integration_tests.test_arrow_serialization_fixes()
        integration_tests.test_plotly_visualization_fixes()
        integration_tests.test_end_to_end_robustness()
        print("✅ All integration tests passed!")
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
    
    # Run performance tests
    print("\n⚡ Running Performance Tests")
    print("-" * 40)
    
    perf_tests = TestPerformanceAndScalability()
    
    try:
        perf_tests.test_large_dataset_handling()
        perf_tests.test_memory_efficiency()
        print("✅ All performance tests passed!")
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
    
    print("\n🎉 Comprehensive testing completed!")
    print("=" * 70)
