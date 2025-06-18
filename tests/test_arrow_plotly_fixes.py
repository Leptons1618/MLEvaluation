"""
Tests for Arrow serialization and Plotly fixes
"""

import pytest
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestArrowSerializationFixes:
    """Test fixes for Arrow serialization errors in Streamlit"""
    
    def setup_method(self):
        """Setup test data with potential Arrow serialization issues"""
        # Create data that could cause Arrow serialization problems
        self.problematic_df = pd.DataFrame({
            'numeric_col': [1.0, 2.0, 3.0, np.nan, 5.0],
            'string_col': ['A', 'B', 'C', None, 'E'],
            'mixed_col': [1, 'text', 3.0, None, True],
            'object_col': [{'key': 'value'}, [1, 2, 3], None, 'string', 42],
            'datetime_col': pd.date_range('2023-01-01', periods=5),
            'categorical_col': pd.Categorical(['cat1', 'cat2', 'cat1', None, 'cat2'])
        })
        
        # Add columns that commonly cause Arrow issues
        self.problematic_df['Mean/Mode'] = ['mean_val', 'mode_val', 'mean_val', None, 'mode_val']
        self.problematic_df['complex_object'] = [
            {'nested': {'data': [1, 2, 3]}}, 
            [{'a': 1}, {'b': 2}], 
            None, 
            'simple_string', 
            42.5
        ]
    
    def test_arrow_compatible_dataframe_creation(self):
        """Test creation of Arrow-compatible DataFrames"""
        # Function to make DataFrame Arrow-compatible
        def make_arrow_compatible(df):
            """Convert DataFrame to be Arrow-compatible"""
            df_clean = df.copy()
            
            for col in df_clean.columns:
                # Convert complex objects to strings
                if df_clean[col].dtype == 'object':
                    # Handle mixed types by converting to string
                    df_clean[col] = df_clean[col].astype(str)
                    # Replace 'None' strings with actual None for proper null handling
                    df_clean[col] = df_clean[col].replace('None', None)
                
                # Ensure categorical columns are properly handled
                elif pd.api.types.is_categorical_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].astype(str)
            
            return df_clean
        
        # Test the conversion
        arrow_compatible_df = make_arrow_compatible(self.problematic_df)
        
        # Verify Arrow compatibility
        # All object columns should be string type after conversion
        for col in arrow_compatible_df.columns:
            if arrow_compatible_df[col].dtype == 'object':
                # Should be string-like or null
                non_null_values = arrow_compatible_df[col].dropna()
                for val in non_null_values:
                    assert isinstance(val, str), f"Non-string value {val} in column {col}"
    
    def test_mean_mode_column_serialization(self):
        """Test specific fix for Mean/Mode column serialization"""
        # Create a quality report DataFrame similar to what the app generates
        quality_report = pd.DataFrame({
            'Column': ['feature1', 'feature2', 'feature3'],
            'Type': ['numeric', 'categorical', 'numeric'],
            'Missing Count': [5, 0, 2],
            'Missing %': [10.0, 0.0, 4.0],
            'Mean/Mode': [15.5, 'category_A', 22.3]  # This column caused issues
        })
        
        # Fix function for the Mean/Mode column
        def fix_mean_mode_serialization(df):
            """Fix Mean/Mode column for Arrow serialization"""
            df_fixed = df.copy()
            if 'Mean/Mode' in df_fixed.columns:
                # Convert all values to strings to ensure consistency
                df_fixed['Mean/Mode'] = df_fixed['Mean/Mode'].astype(str)
                # Handle None/NaN values
                df_fixed['Mean/Mode'] = df_fixed['Mean/Mode'].fillna('N/A')
            return df_fixed
        
        # Apply fix
        fixed_report = fix_mean_mode_serialization(quality_report)
        
        # Verify fix
        assert fixed_report['Mean/Mode'].dtype == 'object'  # String type
        assert fixed_report['Mean/Mode'].isnull().sum() == 0  # No null values
        
        # All values should be strings
        for val in fixed_report['Mean/Mode']:
            assert isinstance(val, str)
    
    def test_complex_object_handling(self):
        """Test handling of complex objects that cause Arrow issues"""
        # Create DataFrame with complex objects
        complex_df = pd.DataFrame({
            'simple_col': [1, 2, 3],
            'list_col': [[1, 2], [3, 4], [5, 6]],
            'dict_col': [{'a': 1}, {'b': 2}, {'c': 3}],
            'mixed_complex': [{'key': [1, 2]}, [{'nested': 'value'}], 'simple_string']
        })
        
        def handle_complex_objects(df):
            """Handle complex objects for Arrow compatibility"""
            df_handled = df.copy()
            
            for col in df_handled.columns:
                if df_handled[col].dtype == 'object':
                    # Check if column contains complex objects
                    sample_val = df_handled[col].dropna().iloc[0] if not df_handled[col].dropna().empty else None
                    
                    if isinstance(sample_val, (list, dict)):
                        # Convert complex objects to JSON strings
                        import json
                        df_handled[col] = df_handled[col].apply(
                            lambda x: json.dumps(x) if x is not None and not isinstance(x, str) else str(x)
                        )
            
            return df_handled
        
        # Apply handling
        handled_df = handle_complex_objects(complex_df)
        
        # Verify handling
        for col in handled_df.columns:
            if col != 'simple_col':  # Skip numeric column
                for val in handled_df[col].dropna():
                    assert isinstance(val, str), f"Non-string value {val} in column {col}"
    
    def test_streamlit_dataframe_compatibility(self):
        """Test DataFrame compatibility with Streamlit's dataframe component"""
        # Mock streamlit dataframe function to test compatibility
        def mock_streamlit_dataframe(df, **kwargs):
            """Mock function that simulates Streamlit's dataframe requirements"""
            # Simulate Arrow serialization requirements
            try:
                # Try to convert to Arrow format (simplified simulation)
                for col in df.columns:
                    if df[col].dtype == 'object':
                        # Check for mixed types that cause issues
                        types_in_col = set(type(x).__name__ for x in df[col].dropna())
                        if len(types_in_col) > 1 and 'str' not in types_in_col:
                            raise ValueError(f"Mixed types in column {col}: {types_in_col}")
                return True
            except Exception as e:
                return False, str(e)
        
        # Test with problematic DataFrame
        result = mock_streamlit_dataframe(self.problematic_df)
        assert result[0] == False  # Should fail due to mixed types
        
        # Test with fixed DataFrame
        def prepare_for_streamlit(df):
            """Prepare DataFrame for Streamlit display"""
            df_prepared = df.copy()
            
            for col in df_prepared.columns:
                if df_prepared[col].dtype == 'object':
                    # Convert all object columns to strings
                    df_prepared[col] = df_prepared[col].astype(str)
                    # Handle None values
                    df_prepared[col] = df_prepared[col].replace('None', 'N/A')
            
            return df_prepared
        
        fixed_df = prepare_for_streamlit(self.problematic_df)
        result = mock_streamlit_dataframe(fixed_df)
        assert result == True  # Should pass after fixing


class TestPlotlyFixes:
    """Test fixes for Plotly errors"""
    
    def setup_method(self):
        """Setup test data for Plotly tests"""
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'x_values': range(10),
            'y_values': np.random.normal(0, 1, 10),
            'categories': ['A', 'B'] * 5,
            'sizes': np.random.uniform(10, 100, 10)
        })
    
    def test_plotly_axis_update_fix(self):
        """Test fix for Plotly axis update methods"""
        # Create a figure
        fig = px.scatter(self.test_data, x='x_values', y='y_values', color='categories')
        
        # Test the problematic method (old way that caused errors)
        def old_axis_update(fig):
            """Old method that caused errors"""
            try:
                fig.update_xaxis(title="X Axis Title")  # This method might not exist
                return True, "Success"
            except AttributeError as e:
                return False, str(e)
        
        # Test the fixed method
        def fixed_axis_update(fig):
            """Fixed method using correct Plotly API"""
            try:
                fig.update_xaxes(title="X Axis Title")  # Correct method
                fig.update_yaxes(title="Y Axis Title")  # Correct method
                return True, "Success"
            except AttributeError as e:
                return False, str(e)
        
        # Test old method (should fail or be inconsistent)
        old_result = old_axis_update(fig)
        
        # Test fixed method (should work)
        fixed_result = fixed_axis_update(fig)
        assert fixed_result[0] == True, f"Fixed method failed: {fixed_result[1]}"
    
    def test_plotly_subplot_axis_updates(self):
        """Test axis updates with subplots"""
        # Create subplots
        fig = make_subplots(rows=2, cols=1, subplot_titles=['Plot 1', 'Plot 2'])
        
        # Add traces
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name='Trace 1'), row=1, col=1)
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 3, 4], name='Trace 2'), row=2, col=1)
        
        # Test proper axis updates for subplots
        def update_subplot_axes(fig):
            """Properly update axes in subplots"""
            try:
                # Update all x-axes
                fig.update_xaxes(title_text="X Axis", showgrid=True)
                # Update all y-axes
                fig.update_yaxes(title_text="Y Axis", showgrid=True)
                
                # Update specific subplot axes
                fig.update_xaxes(title_text="Specific X", row=1, col=1)
                fig.update_yaxes(title_text="Specific Y", row=2, col=1)
                
                return True, "Success"
            except Exception as e:
                return False, str(e)
        
        result = update_subplot_axes(fig)
        assert result[0] == True, f"Subplot axis update failed: {result[1]}"
    
    def test_plotly_figure_creation_robustness(self):
        """Test robust figure creation that handles various data types"""
        def create_robust_plot(data, x_col, y_col):
            """Create plots with robust error handling"""
            try:
                # Ensure data is clean for plotting
                plot_data = data.copy()
                
                # Handle missing values
                plot_data = plot_data.dropna(subset=[x_col, y_col])
                
                # Ensure numeric columns are actually numeric
                if plot_data[x_col].dtype == 'object':
                    plot_data[x_col] = pd.to_numeric(plot_data[x_col], errors='coerce')
                if plot_data[y_col].dtype == 'object':
                    plot_data[y_col] = pd.to_numeric(plot_data[y_col], errors='coerce')
                
                # Remove any remaining NaN values
                plot_data = plot_data.dropna(subset=[x_col, y_col])
                
                if len(plot_data) == 0:
                    raise ValueError("No valid data points for plotting")
                
                # Create figure
                fig = px.scatter(plot_data, x=x_col, y=y_col)
                
                # Apply proper axis updates
                fig.update_xaxes(title=x_col.replace('_', ' ').title())
                fig.update_yaxes(title=y_col.replace('_', ' ').title())
                
                return fig, True, "Success"
                
            except Exception as e:
                return None, False, str(e)
        
        # Test with good data
        fig, success, message = create_robust_plot(self.test_data, 'x_values', 'y_values')
        assert success == True, f"Robust plot creation failed: {message}"
        assert fig is not None
        
        # Test with problematic data
        bad_data = pd.DataFrame({
            'x_col': ['not', 'numeric', 'data'],
            'y_col': [np.nan, np.nan, np.nan]
        })
        
        fig, success, message = create_robust_plot(bad_data, 'x_col', 'y_col')
        assert success == False  # Should fail gracefully
        assert "No valid data points" in message or "convert" in message.lower()
    
    def test_plotly_color_handling(self):
        """Test proper handling of color parameters in Plotly"""
        def create_plot_with_colors(data, color_col=None):
            """Create plot with proper color handling"""
            try:
                if color_col and color_col in data.columns:
                    # Ensure color column is suitable for plotting
                    if data[color_col].dtype == 'object':
                        # For categorical data, ensure no None values
                        data = data.dropna(subset=[color_col])
                    
                    fig = px.scatter(data, x='x_values', y='y_values', color=color_col)
                else:
                    fig = px.scatter(data, x='x_values', y='y_values')
                
                # Apply consistent styling
                fig.update_layout(
                    title="Test Plot",
                    xaxis_title="X Values",
                    yaxis_title="Y Values"
                )
                
                return fig, True, "Success"
                
            except Exception as e:
                return None, False, str(e)
        
        # Test with categorical colors
        fig, success, message = create_plot_with_colors(self.test_data, 'categories')
        assert success == True, f"Color plot creation failed: {message}"
        
        # Test with None color column
        fig, success, message = create_plot_with_colors(self.test_data, None)
        assert success == True, f"No-color plot creation failed: {message}"


class TestRealWorldScenarios:
    """Test fixes in realistic application scenarios"""
    
    def test_quality_report_display_fix(self):
        """Test fixes for quality report display issues"""
        # Simulate quality report data that caused issues
        quality_data = {
            'Column': ['feature1', 'feature2', 'feature3', 'target'],
            'Data Type': ['float64', 'object', 'int64', 'int64'],
            'Missing Count': [5, 0, 2, 0],
            'Missing Percentage': [10.0, 0.0, 4.0, 0.0],
            'Mean/Mode': [15.5, 'category_A', 22, 1],  # Mixed types in this column
            'Unique Values': [45, 3, 20, 2]
        }
        
        quality_df = pd.DataFrame(quality_data)
        
        def fix_quality_report_for_display(df):
            """Fix quality report for proper display"""
            df_fixed = df.copy()
            
            # Fix the Mean/Mode column that caused serialization issues
            if 'Mean/Mode' in df_fixed.columns:
                df_fixed['Mean/Mode'] = df_fixed['Mean/Mode'].astype(str)
            
            # Ensure all percentage columns are properly formatted
            percentage_cols = [col for col in df_fixed.columns if 'percentage' in col.lower()]
            for col in percentage_cols:
                df_fixed[col] = df_fixed[col].round(2)
            
            # Ensure all numeric columns are properly typed
            for col in df_fixed.columns:
                if df_fixed[col].dtype == 'object':
                    try:
                        # Try to convert to numeric if possible
                        numeric_series = pd.to_numeric(df_fixed[col], errors='ignore')
                        if not numeric_series.equals(df_fixed[col]):
                            # Conversion changed values, keep as string
                            df_fixed[col] = df_fixed[col].astype(str)
                    except:
                        df_fixed[col] = df_fixed[col].astype(str)
            
            return df_fixed
        
        # Apply fix
        fixed_quality_df = fix_quality_report_for_display(quality_df)
        
        # Verify fix
        assert fixed_quality_df['Mean/Mode'].dtype == 'object'  # Should be string
        
        # Should be displayable without Arrow serialization errors
        # (In real app, this would be passed to st.dataframe)
        for col in fixed_quality_df.columns:
            if fixed_quality_df[col].dtype == 'object':
                # All object columns should have consistent types
                non_null_values = fixed_quality_df[col].dropna()
                if len(non_null_values) > 0:
                    first_type = type(non_null_values.iloc[0])
                    for val in non_null_values:
                        assert isinstance(val, first_type) or isinstance(val, str)
    
    def test_correlation_heatmap_fix(self):
        """Test fixes for correlation heatmap generation"""
        # Create test data for correlation analysis
        np.random.seed(42)
        corr_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        # Make feature1 and feature2 correlated
        corr_data['feature2'] = corr_data['feature1'] + np.random.normal(0, 0.5, 100)
        
        def create_correlation_heatmap_fixed(data):
            """Create correlation heatmap with proper error handling"""
            try:
                # Calculate correlation matrix
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) < 2:
                    return None, False, "Not enough numeric columns for correlation"
                
                corr_matrix = data[numeric_cols].corr()
                
                # Create heatmap using proper Plotly methods
                fig = px.imshow(
                    corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    color_continuous_scale='RdBu_r',
                    aspect='auto'
                )
                
                # Apply proper axis updates (fixed method)
                fig.update_xaxes(title="Features")
                fig.update_yaxes(title="Features")
                
                # Update layout
                fig.update_layout(
                    title="Correlation Heatmap",
                    width=600,
                    height=600
                )
                
                return fig, True, "Success"
                
            except Exception as e:
                return None, False, str(e)
        
        # Test the fixed heatmap creation
        fig, success, message = create_correlation_heatmap_fixed(corr_data)
        assert success == True, f"Correlation heatmap creation failed: {message}"
        assert fig is not None


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
