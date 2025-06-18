# Arrow Serialization and Plotly Error Fixes

## Overview

This document provides comprehensive guidance on fixing Arrow serialization errors in Streamlit and Plotly visualization errors that commonly occur in data science applications.

## Arrow Serialization Errors

### Common Error Messages

```
ArrowInvalid: Could not convert <value> with type <type>: did not recognize Python value type when inferring an Arrow data type
```

```
ArrowTypeError: Expected bytes, got a 'str' object
```

### Root Causes

1. **Mixed Data Types**: Columns containing different data types (e.g., strings and numbers)
2. **Complex Objects**: Lists, dictionaries, or custom objects in DataFrame columns
3. **Inconsistent Null Handling**: Mixed None, NaN, and string representations
4. **Categorical Data Issues**: Inconsistent category handling

### Solutions

#### 1. Fix Mixed Data Types

```python
def fix_mixed_types(df):
    """Convert mixed-type columns to string for Arrow compatibility"""
    df_fixed = df.copy()
    
    for col in df_fixed.columns:
        if df_fixed[col].dtype == 'object':
            # Convert all values to strings
            df_fixed[col] = df_fixed[col].astype(str)
            # Handle None values properly
            df_fixed[col] = df_fixed[col].replace('None', 'N/A')
    
    return df_fixed
```

#### 2. Handle Complex Objects

```python
import json

def handle_complex_objects(df):
    """Convert complex objects to JSON strings"""
    df_handled = df.copy()
    
    for col in df_handled.columns:
        if df_handled[col].dtype == 'object':
            # Check if column contains complex objects
            sample_val = df_handled[col].dropna().iloc[0] if not df_handled[col].dropna().empty else None
            
            if isinstance(sample_val, (list, dict)):
                # Convert to JSON strings
                df_handled[col] = df_handled[col].apply(
                    lambda x: json.dumps(x) if x is not None and not isinstance(x, str) else str(x)
                )
    
    return df_handled
```

#### 3. Fix Quality Report Serialization

```python
def fix_quality_report_serialization(quality_df):
    """Fix the common Mean/Mode column serialization issue"""
    df_fixed = quality_df.copy()
    
    # Fix the problematic Mean/Mode column
    if 'Mean/Mode' in df_fixed.columns:
        df_fixed['Mean/Mode'] = df_fixed['Mean/Mode'].astype(str)
        df_fixed['Mean/Mode'] = df_fixed['Mean/Mode'].fillna('N/A')
    
    # Ensure percentage columns are numeric
    percentage_cols = [col for col in df_fixed.columns if 'percentage' in col.lower() or '%' in col]
    for col in percentage_cols:
        df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce').fillna(0)
    
    return df_fixed
```

#### 4. Comprehensive DataFrame Preparation

```python
def prepare_dataframe_for_streamlit(df):
    """Comprehensive DataFrame preparation for Streamlit display"""
    df_prepared = df.copy()
    
    for col in df_prepared.columns:
        if df_prepared[col].dtype == 'object':
            # Handle mixed types
            try:
                # Try to convert to numeric if all values are numeric
                numeric_series = pd.to_numeric(df_prepared[col], errors='coerce')
                if not numeric_series.isnull().all():
                    # Keep as numeric if conversion successful
                    df_prepared[col] = numeric_series
                else:
                    # Convert to string
                    df_prepared[col] = df_prepared[col].astype(str)
                    df_prepared[col] = df_prepared[col].replace('None', 'N/A')
            except:
                # Fallback to string conversion
                df_prepared[col] = df_prepared[col].astype(str)
                df_prepared[col] = df_prepared[col].replace('None', 'N/A')
        
        # Handle categorical columns
        elif pd.api.types.is_categorical_dtype(df_prepared[col]):
            df_prepared[col] = df_prepared[col].astype(str)
    
    return df_prepared
```

### Implementation in UI Components

```python
# In enhanced_ui_components.py
def render_enhanced_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    # ... existing code ...
    
    # Fix quality report before display
    quality_df = calculate_quality_metrics(df)
    quality_df_fixed = prepare_dataframe_for_streamlit(quality_df)
    
    # Now safe to display
    st.dataframe(quality_df_fixed, use_container_width=True)
```

## Plotly Errors

### Common Error Messages

```
AttributeError: 'Figure' object has no attribute 'update_xaxis'
```

```
ValueError: Invalid property specified for object of type plotly.graph_objects.Figure: 'update_xaxis'
```

### Root Causes

1. **Incorrect Method Names**: Using `update_xaxis` instead of `update_xaxes`
2. **API Changes**: Plotly API changes between versions
3. **Subplot Handling**: Incorrect axis updates in subplots
4. **Data Type Issues**: Passing incompatible data types to plots

### Solutions

#### 1. Fix Axis Update Methods

```python
# ❌ Incorrect (causes errors)
fig.update_xaxis(title="X Axis Title")
fig.update_yaxis(title="Y Axis Title")

# ✅ Correct
fig.update_xaxes(title="X Axis Title")
fig.update_yaxes(title="Y Axis Title")
```

#### 2. Robust Plot Creation

```python
def create_correlation_heatmap(data):
    """Create correlation heatmap with proper error handling"""
    try:
        # Ensure data is numeric
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return None, "Not enough numeric columns for correlation"
        
        # Calculate correlation
        corr_matrix = data[numeric_cols].corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        
        # Apply proper axis updates
        fig.update_xaxes(title="Features")
        fig.update_yaxes(title="Features")
        
        # Update layout
        fig.update_layout(
            title="Feature Correlation Matrix",
            width=600,
            height=600
        )
        
        return fig, "Success"
        
    except Exception as e:
        return None, f"Error creating heatmap: {str(e)}"
```

#### 3. Subplot Handling

```python
def create_subplots_with_proper_axis_updates():
    """Create subplots with proper axis updates"""
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Plot 1', 'Plot 2', 'Plot 3', 'Plot 4']
    )
    
    # Add traces
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]), row=1, col=1)
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 3, 4]), row=1, col=2)
    
    # Update all axes
    fig.update_xaxes(title_text="X Axis", showgrid=True)
    fig.update_yaxes(title_text="Y Axis", showgrid=True)
    
    # Update specific subplot axes
    fig.update_xaxes(title_text="Specific X", row=1, col=1)
    fig.update_yaxes(title_text="Specific Y", row=2, col=1)
    
    return fig
```

#### 4. Data Validation for Plots

```python
def validate_plot_data(data, x_col, y_col, color_col=None):
    """Validate data before plotting"""
    plot_data = data.copy()
    
    # Check required columns exist
    if x_col not in plot_data.columns or y_col not in plot_data.columns:
        raise ValueError(f"Required columns not found: {x_col}, {y_col}")
    
    # Handle missing values
    plot_data = plot_data.dropna(subset=[x_col, y_col])
    
    # Ensure numeric columns are numeric
    if plot_data[x_col].dtype == 'object':
        plot_data[x_col] = pd.to_numeric(plot_data[x_col], errors='coerce')
    if plot_data[y_col].dtype == 'object':
        plot_data[y_col] = pd.to_numeric(plot_data[y_col], errors='coerce')
    
    # Remove any remaining NaN values
    plot_data = plot_data.dropna(subset=[x_col, y_col])
    
    if len(plot_data) == 0:
        raise ValueError("No valid data points for plotting")
    
    # Handle color column
    if color_col and color_col in plot_data.columns:
        if plot_data[color_col].dtype == 'object':
            plot_data = plot_data.dropna(subset=[color_col])
    
    return plot_data
```

### Implementation in UI Components

```python
# In enhanced_ui_components.py
def render_correlation_analysis(df: pd.DataFrame, feature_cols: List[str]):
    """Render correlation analysis with proper error handling"""
    try:
        # Validate and prepare data
        numeric_data = df[feature_cols].select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            st.warning("Need at least 2 numeric features for correlation analysis")
            return
        
        # Create correlation heatmap
        fig, message = create_correlation_heatmap(numeric_data)
        
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Could not create correlation plot: {message}")
            
    except Exception as e:
        st.error(f"Error in correlation analysis: {str(e)}")
```

## Best Practices

### For Arrow Serialization

1. **Always Convert Objects to Strings**: When displaying DataFrames in Streamlit
2. **Handle Null Values Consistently**: Use a standard representation like 'N/A'
3. **Validate Data Types**: Ensure consistency within columns
4. **Test with Edge Cases**: Try empty DataFrames, all-null columns, etc.

### For Plotly Visualizations

1. **Use Correct Method Names**: `update_xaxes`, not `update_xaxis`
2. **Validate Data Before Plotting**: Check for required columns and data types
3. **Handle Errors Gracefully**: Provide meaningful error messages
4. **Test with Various Data Types**: Numbers, strings, dates, etc.

### General Error Handling

```python
def safe_operation(operation_func, *args, **kwargs):
    """Safely execute operations with comprehensive error handling"""
    try:
        result = operation_func(*args, **kwargs)
        return result, True, "Success"
    except Exception as e:
        error_msg = f"Operation failed: {str(e)}"
        logging.error(error_msg)
        return None, False, error_msg
```

## Testing the Fixes

### Arrow Serialization Tests

```python
def test_arrow_serialization_fix():
    # Create problematic data
    problematic_df = pd.DataFrame({
        'mixed_col': [1, 'text', 3.0, None, True],
        'Mean/Mode': [15.5, 'category_A', 22.3, None, 'mixed']
    })
    
    # Apply fix
    fixed_df = prepare_dataframe_for_streamlit(problematic_df)
    
    # Verify fix
    assert fixed_df['Mean/Mode'].dtype == 'object'
    for val in fixed_df['Mean/Mode']:
        assert isinstance(val, str)
```

### Plotly Fixes Tests

```python
def test_plotly_axis_updates():
    import plotly.express as px
    
    # Create test data
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [2, 4, 6, 8, 10]
    })
    
    # Create figure
    fig = px.scatter(df, x='x', y='y')
    
    # Test correct axis updates
    try:
        fig.update_xaxes(title="X Axis")
        fig.update_yaxes(title="Y Axis")
        success = True
    except AttributeError:
        success = False
    
    assert success, "Axis updates should work with correct method names"
```

## Migration Guide

### Updating Existing Code

1. **Find and Replace**: Search for `update_xaxis` and replace with `update_xaxes`
2. **Add Data Validation**: Insert data validation before DataFrame display
3. **Wrap in Error Handling**: Add try-catch blocks around visualization code
4. **Test Thoroughly**: Run tests with various data types and edge cases

### Checklist for New Code

- [ ] Use `update_xaxes` and `update_yaxes` for axis updates
- [ ] Validate data types before DataFrame display
- [ ] Handle null values consistently
- [ ] Add comprehensive error handling
- [ ] Test with edge cases and various data types
- [ ] Document any data type assumptions

## Troubleshooting

### If Errors Persist

1. **Check Plotly Version**: Ensure compatible version
2. **Validate Data Types**: Print data types before operations
3. **Test with Minimal Data**: Use simple test cases
4. **Enable Debug Logging**: Add detailed logging
5. **Check Streamlit Version**: Ensure compatibility

### Debug Tools

```python
def debug_dataframe(df, name="DataFrame"):
    """Debug DataFrame for serialization issues"""
    print(f"\n=== {name} Debug Info ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Null values:\n{df.isnull().sum()}")
    
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_types = set(type(x).__name__ for x in df[col].dropna())
            print(f"Column '{col}' contains types: {unique_types}")
```

---

These fixes ensure robust data handling and visualization in the ML Evaluation application, preventing common serialization and plotting errors that can disrupt the user experience.
