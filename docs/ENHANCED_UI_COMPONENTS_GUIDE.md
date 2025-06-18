# Enhanced UI Components Guide

## Overview

This guide provides comprehensive documentation for the enhanced UI components that add advanced data preparation capabilities to the ML Evaluation platform.

## 🎯 Core Features

### 1. Enhanced Data Quality Report

The data quality report provides comprehensive insights into your dataset's characteristics and potential issues.

#### Features:
- **Dataset Overview**: Key statistics and dimensions
- **Missing Values Analysis**: Detailed breakdown of missing data patterns
- **Data Type Information**: Column types and conversion suggestions
- **Quality Metrics**: Completeness, uniqueness, and validity scores

#### Usage:
```python
from components.enhanced_ui_components import render_data_quality_report
import streamlit as st

# Render the quality report
render_data_quality_report(df)
```

#### Interactive Elements:
- **Column Selection**: Choose specific columns for analysis
- **Missing Value Visualization**: Bar charts and heatmaps
- **Quality Score Breakdown**: Metric-by-metric analysis
- **Export Options**: Download quality reports

### 2. Smart Imputation System

Advanced missing value handling with intelligent recommendations and multiple imputation strategies.

#### Available Methods:

**Numeric Columns:**
- **Mean**: Best for normally distributed data
- **Median**: Robust against outliers
- **Mode**: For discrete numeric values
- **Forward Fill**: For time series data
- **Backward Fill**: For time series data
- **Interpolation**: Linear interpolation between values
- **KNN**: K-nearest neighbors imputation
- **Drop Rows**: Remove rows with missing values

**Categorical Columns:**
- **Mode**: Most frequent value
- **Forward Fill**: Previous valid value
- **Backward Fill**: Next valid value
- **Custom Value**: User-specified replacement
- **Drop Rows**: Remove rows with missing values

#### Smart Recommendations:
The system automatically recommends the best imputation method based on:
- Data distribution (skewness, kurtosis)
- Missing value patterns
- Column data type
- Relationship with other columns

#### Usage Example:
```python
# Configure imputation for multiple columns
imputation_config = {
    'age': {'method': 'Median', 'custom_value': None},
    'category': {'method': 'Mode', 'custom_value': None},
    'income': {'method': 'Mean', 'custom_value': None}
}

# Apply imputation
result_df = apply_imputation(original_df, imputation_config)
```

### 3. Operation Tracking System

Comprehensive logging and tracking of all data transformations.

#### Features:
- **Complete History**: Track every operation with timestamps
- **Parameter Logging**: Store all operation parameters
- **Reversible Operations**: Reset to original state
- **Comparison Views**: Before/after comparisons
- **Export History**: Save operation logs

#### Operation Types Tracked:
1. **Data Type Changes**: Column type conversions
2. **Missing Value Imputation**: All imputation operations
3. **Feature Selection**: Manual and automatic selections
4. **Data Filtering**: Row and column filtering
5. **Feature Engineering**: New feature creation
6. **Data Scaling**: Normalization and standardization

#### Usage:
```python
# Operations are automatically logged
st.session_state.operation_log = []

# Each operation adds an entry
operation_entry = {
    'operation': 'Missing Value Imputation',
    'details': 'Applied median imputation to age column',
    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'parameters': imputation_config
}

st.session_state.operation_log.append(operation_entry)
```

### 4. Advanced Feature Selection

Multiple strategies for selecting relevant features for model training.

#### Selection Methods:

**Manual Selection:**
- Interactive multi-select interface
- Search and filter capabilities
- Drag-and-drop reordering
- Preview selected features

**Statistical Selection:**
- **F-statistic**: ANOVA F-test for classification
- **Mutual Information**: Information-theoretic approach
- **Chi-square**: For categorical features
- **Correlation**: Pearson correlation coefficient

**Correlation-Based Selection:**
- Correlation with target variable
- Feature-to-feature correlation
- Multicollinearity detection
- Threshold-based filtering

**Smart Suggestions:**
- Domain-specific recommendations
- Feature importance from preliminary models
- Interaction effect detection
- Redundancy elimination

#### Usage Example:
```python
from components.enhanced_ui_components import render_advanced_feature_selection

# Render feature selection interface
render_advanced_feature_selection(df, target_column='target')

# Access selected features
selected_features = st.session_state.get('selected_features', [])
```

### 5. Correlation Analysis

Comprehensive correlation analysis with interactive visualizations.

#### Features:
- **Correlation Matrix**: Full correlation heatmap
- **Target Correlation**: Features vs target correlation
- **Top Correlations**: Most correlated feature pairs
- **Threshold Filtering**: Filter by correlation strength
- **Interactive Charts**: Plotly-based visualizations

#### Visualization Types:
1. **Heatmaps**: Full correlation matrices
2. **Bar Charts**: Target correlation rankings
3. **Scatter Plots**: Feature relationships
4. **Network Graphs**: Correlation networks

#### Usage:
```python
from components.enhanced_ui_components import render_correlation_analysis

# Render correlation analysis
render_correlation_analysis(df, target_column, all_features)
```

### 6. Feature Engineering Suggestions

Intelligent suggestions for creating new features and improving existing ones.

#### Suggestion Categories:

**Categorical Encoding:**
- One-hot encoding for low cardinality
- Label encoding for ordinal data
- Target encoding for high cardinality
- Binary encoding for memory efficiency

**Numerical Transformations:**
- Log transformation for skewed data
- Square root transformation
- Polynomial features
- Binning for continuous variables

**Scaling and Normalization:**
- StandardScaler for normal distributions
- MinMaxScaler for bounded ranges
- RobustScaler for outlier-prone data
- QuantileTransformer for non-normal data

**Feature Combinations:**
- Arithmetic operations (sum, difference, ratio)
- Interaction terms
- Polynomial combinations
- Domain-specific combinations

**Dimensionality Reduction:**
- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)
- t-SNE for visualization
- UMAP for non-linear reduction

#### Usage Example:
```python
from components.enhanced_ui_components import render_feature_engineering_suggestions

# Get suggestions for feature engineering
suggestions = render_feature_engineering_suggestions(df, target_column)

# Apply selected suggestions
for suggestion in suggestions:
    if suggestion['selected']:
        df = apply_feature_engineering(df, suggestion)
```

## 🛠️ Configuration Options

### Global Settings
```python
# Configure enhanced UI components
ENHANCED_UI_CONFIG = {
    'enable_smart_recommendations': True,
    'max_correlation_features': 20,
    'default_imputation_method': 'auto',
    'enable_operation_tracking': True,
    'chart_theme': 'plotly_white'
}
```

### Performance Settings
```python
# Optimize for large datasets
PERFORMANCE_CONFIG = {
    'max_rows_for_correlation': 10000,
    'sample_size_for_suggestions': 5000,
    'enable_progress_bars': True,
    'chunked_processing': True
}
```

## 🎨 Customization

### Theme Customization
```python
# Custom color schemes
CUSTOM_THEME = {
    'primary_color': '#1f77b4',
    'secondary_color': '#ff7f0e',
    'success_color': '#2ca02c',
    'warning_color': '#ff9900',
    'error_color': '#d62728'
}
```

### Layout Customization
```python
# Custom layout options
LAYOUT_CONFIG = {
    'sidebar_width': 300,
    'main_content_width': 800,
    'chart_height': 400,
    'table_height': 300
}
```

## 📊 Integration Examples

### Complete Data Preparation Workflow
```python
import streamlit as st
from components.enhanced_ui_components import *

def complete_data_prep_workflow(df, target_column):
    # Step 1: Quality Report
    st.header("📊 Data Quality Assessment")
    render_data_quality_report(df)
    
    # Step 2: Handle Missing Values
    st.header("🔧 Missing Value Treatment")
    if st.session_state.get('missing_values_handled'):
        df = st.session_state.modified_df
    
    # Step 3: Feature Selection
    st.header("🎯 Feature Selection")
    render_advanced_feature_selection(df, target_column)
    
    # Step 4: Correlation Analysis
    st.header("🔍 Correlation Analysis")
    selected_features = st.session_state.get('selected_features', [])
    render_correlation_analysis(df, target_column, selected_features)
    
    # Step 5: Feature Engineering
    st.header("⚙️ Feature Engineering")
    render_feature_engineering_suggestions(df, target_column)
    
    # Step 6: Operation Tracking
    st.header("📝 Operation History")
    render_operation_tracker()
    
    return df
```

### Custom Imputation Strategy
```python
def custom_imputation_strategy(df, strategy='smart'):
    """
    Apply custom imputation strategy based on data characteristics
    """
    imputation_config = {}
    
    for column in df.columns:
        if df[column].isnull().any():
            if strategy == 'smart':
                # Smart strategy based on data distribution
                if df[column].dtype in ['int64', 'float64']:
                    skewness = df[column].skew()
                    if abs(skewness) > 1:
                        method = 'Median'  # For skewed distributions
                    else:
                        method = 'Mean'    # For normal distributions
                else:
                    method = 'Mode'        # For categorical data
            elif strategy == 'conservative':
                method = 'Median' if df[column].dtype in ['int64', 'float64'] else 'Mode'
            elif strategy == 'aggressive':
                method = 'Mean' if df[column].dtype in ['int64', 'float64'] else 'Mode'
            
            imputation_config[column] = {'method': method, 'custom_value': None}
    
    return apply_imputation(df, imputation_config)
```

## 🚀 Advanced Features

### Automated Feature Engineering Pipeline
```python
def automated_feature_engineering(df, target_column):
    """
    Automated feature engineering pipeline
    """
    # Categorical encoding
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col != target_column:
            # Apply appropriate encoding based on cardinality
            cardinality = df[col].nunique()
            if cardinality <= 10:
                # One-hot encoding for low cardinality
                df = pd.get_dummies(df, columns=[col], prefix=col)
            else:
                # Label encoding for high cardinality
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
    
    # Numerical transformations
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_columns:
        if col != target_column:
            # Log transformation for skewed data
            if df[col].skew() > 1:
                df[f'{col}_log'] = np.log1p(df[col])
            
            # Polynomial features
            df[f'{col}_squared'] = df[col] ** 2
    
    # Feature interactions
    for i, col1 in enumerate(numerical_columns):
        for col2 in numerical_columns[i+1:]:
            if col1 != target_column and col2 != target_column:
                df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
    
    return df
```

### Real-time Data Quality Monitoring
```python
def real_time_quality_monitoring(df):
    """
    Real-time monitoring of data quality metrics
    """
    quality_metrics = {
        'completeness': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'uniqueness': df.nunique().sum() / len(df),
        'consistency': calculate_consistency_score(df),
        'validity': calculate_validity_score(df)
    }
    
    # Display real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Completeness", f"{quality_metrics['completeness']:.1f}%")
    with col2:
        st.metric("Uniqueness", f"{quality_metrics['uniqueness']:.2f}")
    with col3:
        st.metric("Consistency", f"{quality_metrics['consistency']:.2f}")
    with col4:
        st.metric("Validity", f"{quality_metrics['validity']:.2f}")
    
    return quality_metrics
```

## 🔧 Troubleshooting

### Common Issues and Solutions

**Issue: Arrow Serialization Errors**
```python
# Solution: Use safe calculation functions
mean_val = safe_mean(df['column'])  # Instead of df['column'].mean()
median_val = safe_median(df['column'])  # Instead of df['column'].median()
mode_val = safe_mode(df['column'])  # Instead of df['column'].mode()
```

**Issue: Memory Issues with Large Datasets**
```python
# Solution: Use chunked processing
def process_large_dataset(df, chunk_size=10000):
    results = []
    for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
        processed_chunk = apply_imputation(chunk, config)
        results.append(processed_chunk)
    return pd.concat(results, ignore_index=True)
```

**Issue: Slow Correlation Calculations**
```python
# Solution: Sample data for correlation analysis
def fast_correlation_analysis(df, sample_size=5000):
    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=42)
    else:
        sample_df = df
    return sample_df.corr()
```

This comprehensive guide provides all the information needed to effectively use the enhanced UI components for advanced data preparation tasks.
