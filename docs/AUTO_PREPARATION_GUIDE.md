# Auto Data Preparation System Documentation

## Overview

The ML Evaluation application now includes a comprehensive auto data preparation system that helps users automatically detect and fix common data quality issues. This system addresses the frequent stratification error and provides intelligent recommendations for data preparation.

## Key Features

### 🔍 Intelligent Issue Detection

The system automatically analyzes your dataset and detects:

1. **Missing Values**
   - Target column missing values (critical)
   - Feature column missing values
   - High missing value features (>50%)

2. **Class Distribution Issues**
   - Classes with only 1 sample (causes stratification error)
   - Class imbalance (ratio > 5:1)
   - Insufficient samples for stratified splitting

3. **Data Quality Issues**
   - Duplicate rows
   - Feature scaling problems (different scales)
   - High cardinality categorical features

4. **Dataset Size Issues**
   - Too many features (dimensionality)
   - Small datasets that need special handling

### 🛠️ Auto-Preparation Tools

#### Critical Fixes (Required)
- **🚨 Fix Small Classes**: Removes classes with <2 samples to prevent stratification errors
- **🎯 Clean Target Column**: Removes rows with missing target values

#### High Priority Fixes (Recommended)
- **🔄 Remove Duplicates**: Eliminates duplicate rows
- **⚖️ Balance Classes**: Uses SMOTE or sampling to address class imbalance

#### Medium Priority Fixes
- **🔧 Handle Missing Values**: Imputes missing values (median for numeric, mode for categorical)
- **📏 Scale Features**: Normalizes feature scales using StandardScaler

#### Low Priority Fixes
- **🗑️ Drop High-Missing Features**: Removes features with >50% missing values
- **🎯 Feature Selection**: Selects most important features to reduce dimensionality

### 🚀 Auto-Preparation Options

1. **Auto-Fix Critical Issues**: Applies only critical fixes to make dataset trainable
2. **Auto-Fix All Recommended**: Applies critical, high, and medium priority fixes
3. **Custom Fix Selection**: Choose specific fixes to apply

## Stratification Error Solution

The system handles the common error:
```
"The least populated class in y has only 1 member, which is too few. 
The minimum number of groups for any class cannot be less than 2."
```

### Automatic Solutions:
1. **Detection**: Identifies classes with <2 samples
2. **Removal**: Automatically removes small classes
3. **Fallback**: Uses non-stratified splitting when needed
4. **Adjustment**: Adapts test size for very small datasets

## How to Use

### 1. Basic Usage
1. Upload your dataset in the "Data Upload & Preparation" page
2. Click "🔍 Analyze Preparation Needs"
3. Review detected issues and recommendations
4. Click "🔧 Auto-Fix Critical Issues" for basic preparation

### 2. Advanced Usage
1. Follow steps 1-3 above
2. Click "⚡ Auto-Fix All Recommended" for comprehensive preparation
3. Or use "🎛️ Custom Fix Selection" to choose specific fixes
4. Review the preparation log to see what was applied

### 3. Testing with Sample Data
Use the provided sample datasets:
- `problematic_dataset.csv`: Multiple issues (missing data, duplicates, class imbalance)
- `missing_values_dataset.csv`: Different levels of missing data
- `single_class_dataset.csv`: Classes with single samples (stratification error)

## Technical Details

### Enhanced Data Preparation Module
- **Location**: `src/utils/data_preparation_enhanced.py`
- **Main Class**: `DataPreparationTools`
- **Key Methods**:
  - `analyze_preparation_needs()`: Detects issues
  - `get_preparation_recommendations()`: Provides prioritized recommendations
  - `auto_prepare_dataset()`: Applies selected fixes

### Basic Data Preparation (Fallback)
- **Location**: `src/utils/data_preparation.py`
- **Enhanced with**: Robust error handling and automatic fallbacks
- **Handles**: Stratification errors, small datasets, edge cases

### UI Integration
- **Location**: `src/components/ui_components.py`
- **Function**: `render_dataset_preparation()`
- **Features**: Interactive preparation interface with progress tracking

## Error Handling

### Stratification Errors
```python
# Automatic detection and handling
class_counts = pd.Series(y).value_counts()
min_class_size = class_counts.min()
use_stratify = min_class_size >= 2

if not use_stratify:
    # Fall back to non-stratified split
    train_test_split(..., stratify=None)
```

### Small Dataset Handling
```python
# Adjust test size for very small datasets
if actual_test_samples < num_classes:
    adjusted_test_size = max(0.3, num_classes / total_samples)
```

### Missing Value Imputation
```python
# Smart imputation based on data type
- Numeric features: Median imputation
- Categorical features: Mode imputation
- High missing (>50%): Option to drop
```

## Best Practices

1. **Start with Analysis**: Always run "Analyze Preparation Needs" first
2. **Apply Critical Fixes**: Ensure critical issues are resolved
3. **Review Logs**: Check the preparation log to understand what was changed
4. **Validate Results**: Review the preparation summary metrics
5. **Iterative Approach**: Apply fixes gradually and validate at each step

## Example Workflow

```python
# 1. Load and analyze
df = load_uploaded_file(uploaded_file)
analysis = data_prep_tools.analyze_preparation_needs(df, target_column)

# 2. Get recommendations
recommendations = data_prep_tools.get_preparation_recommendations(analysis)

# 3. Apply fixes
critical_fixes = ['handle_small_classes', 'remove_missing_target']
prepared_data = data_prep_tools.auto_prepare_dataset(df, target_column, critical_fixes)

# 4. Use in ML pipeline
X_train = prepared_data['X_train']
y_train = prepared_data['y_train']
```

## Dependencies

- **Required**: pandas, numpy, scikit-learn
- **Optional**: imbalanced-learn (for advanced class balancing)
- **Install**: `pip install imbalanced-learn`

## Future Enhancements

- Outlier detection and handling
- Advanced imputation methods (KNN, iterative)
- Feature engineering suggestions
- Data drift detection
- Export functionality for prepared datasets

---

This auto-preparation system makes machine learning more accessible by automatically handling common data quality issues and providing clear guidance for data preparation decisions.
