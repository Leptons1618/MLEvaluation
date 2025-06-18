"""
Enhanced UI components with advanced data preparation features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

# Import Arrow compatibility utilities
try:
    from utils.arrow_compatibility import (
        make_dataframe_arrow_compatible, 
        safe_dataframe_display,
        create_arrow_safe_summary
    )
except ImportError:
    # Fallback if not available
    def make_dataframe_arrow_compatible(df):
        return df
    def safe_dataframe_display(df):
        return df
    def create_arrow_safe_summary(df):
        return df


def safe_mean(series: pd.Series) -> float:
    """Calculate mean in an Arrow-safe way"""
    try:
        # Convert to numpy array to avoid Arrow issues
        clean_values = series.dropna().values
        if len(clean_values) == 0:
            return 0.0
        return float(np.mean(clean_values))
    except Exception:
        return 0.0


def safe_median(series: pd.Series) -> float:
    """Calculate median in an Arrow-safe way"""
    try:
        # Convert to numpy array to avoid Arrow issues
        clean_values = series.dropna().values
        if len(clean_values) == 0:
            return 0.0
        return float(np.median(clean_values))
    except Exception:
        return 0.0


def safe_mode(series: pd.Series):
    """Calculate mode in an Arrow-safe way"""
    try:
        # For numeric columns
        if series.dtype in ['int64', 'float64', 'Int64', 'Float64']:
            clean_values = series.dropna().values
            if len(clean_values) == 0:
                return 0.0
            # Use scipy.stats.mode for better performance
            try:
                from scipy import stats
                mode_result = stats.mode(clean_values, keepdims=False)
                return float(mode_result.mode)
            except ImportError:
                # Fallback to pandas mode
                mode_series = pd.Series(clean_values).mode()
                return float(mode_series.iloc[0]) if len(mode_series) > 0 else 0.0
        else:
            # For categorical columns
            mode_series = series.mode()
            return mode_series.iloc[0] if len(mode_series) > 0 else 'Unknown'
    except Exception:
        return 'Unknown' if series.dtype == 'object' else 0.0


def render_editable_column_types(df: pd.DataFrame, analysis: Dict[str, Any]) -> pd.DataFrame:
    """Render editable column types interface"""
    st.markdown("### 🔧 Column Type Configuration")
    st.markdown("*Configure column data types for better analysis*")
    
    # Initialize session state for column types if not exists
    if 'column_type_changes' not in st.session_state:
        st.session_state.column_type_changes = {}
    
    # Create editable column configuration
    col_config = []
    
    for col in df.columns:
        current_type = str(df[col].dtype)
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        unique_vals = df[col].nunique()
        
        # Determine current category
        if col in analysis['numeric_columns']:
            current_category = "Numeric"
        elif col in analysis['categorical_columns']:
            current_category = "Categorical"
        elif col in analysis['datetime_columns']:
            current_category = "DateTime"
        else:
            current_category = "Other"
        
        col_config.append({
            'Column': col,
            'Current Type': current_type,
            'Category': current_category,
            'Missing %': f"{missing_pct:.1f}%",
            'Unique Values': unique_vals,
            'Sample Values': ', '.join(str(x) for x in df[col].dropna().head(3).tolist())
        })
    
    # Display current configuration
    config_df = pd.DataFrame(col_config)
    st.dataframe(safe_dataframe_display(config_df), use_container_width=True, height=300)
    
    # Interactive type editor
    st.markdown("#### 🎛️ Modify Column Types")
    
    # Select columns to modify
    columns_to_modify = st.multiselect(
        "Select columns to modify",
        options=df.columns.tolist(),
        help="Choose columns whose data types you want to change"
    )
    
    if columns_to_modify:
        st.markdown("**Configure selected columns:**")
        
        changes_made = False
        for col in columns_to_modify:
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.text(f"📊 {col}")
                st.caption(f"Current: {str(df[col].dtype)}")
            
            with col2:
                new_type = st.selectbox(
                    f"New type for {col}",
                    options=["Keep Current", "int64", "float64", "object", "datetime64", "category", "boolean"],
                    key=f"type_select_{col}",
                    help=f"Select new data type for {col}"
                )
                
                if new_type != "Keep Current":
                    st.session_state.column_type_changes[col] = new_type
                    changes_made = True
            
            with col3:
                if st.button(f"🔄", key=f"reset_{col}", help=f"Reset {col} to original type"):
                    if col in st.session_state.column_type_changes:
                        del st.session_state.column_type_changes[col]
                        st.rerun()
        
        # Apply changes button
        if changes_made and st.button("✅ Apply Type Changes", type="primary"):
            df_modified = apply_column_type_changes(df, st.session_state.column_type_changes)
            st.session_state.modified_df = df_modified
            st.session_state.operation_log = st.session_state.get('operation_log', [])
            st.session_state.operation_log.append({
                'operation': 'Column Type Changes',
                'details': f"Modified types for: {list(st.session_state.column_type_changes.keys())}",
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'changes': st.session_state.column_type_changes.copy()
            })
            st.success("✅ Column types updated successfully!")
            st.rerun()
    
    return df

def apply_column_type_changes(df: pd.DataFrame, type_changes: Dict[str, str]) -> pd.DataFrame:
    """Apply column type changes to dataframe"""
    df_modified = df.copy()
    
    for col, new_type in type_changes.items():
        try:
            if new_type == "int64":
                df_modified[col] = pd.to_numeric(df_modified[col], errors='coerce').astype('Int64')
            elif new_type == "float64":
                df_modified[col] = pd.to_numeric(df_modified[col], errors='coerce')
            elif new_type == "object":
                df_modified[col] = df_modified[col].astype(str)
            elif new_type == "datetime64":
                df_modified[col] = pd.to_datetime(df_modified[col], errors='coerce')
            elif new_type == "category":
                df_modified[col] = df_modified[col].astype('category')
            elif new_type == "boolean":
                df_modified[col] = df_modified[col].astype('boolean')
        except Exception as e:
            st.warning(f"⚠️ Could not convert {col} to {new_type}: {str(e)}")
    
    return df_modified

def render_enhanced_quality_report(df: pd.DataFrame, analysis: Dict[str, Any]):
    """Render enhanced quality report with imputation options"""
    st.markdown("### 📊 Enhanced Data Quality Report")
      # Quality overview
    quality_metrics = {
        'Total Rows': len(df),
        'Total Columns': len(df.columns),
        'Missing Values': int(df.isnull().sum().sum()),
        'Duplicate Rows': int(df.duplicated().sum()),
        'Memory Usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
    }
    
    # Display metrics
    metric_cols = st.columns(len(quality_metrics))
    for i, (metric, value) in enumerate(quality_metrics.items()):
        with metric_cols[i]:
            st.metric(metric, value)
      # Missing values analysis
    st.markdown("#### 🔍 Missing Values Analysis")
    
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': [int(df[col].isnull().sum()) for col in df.columns],
        'Missing Percentage': [df[col].isnull().sum() / len(df) * 100 for col in df.columns],
        'Data Type': [str(df[col].dtype) for col in df.columns]
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Percentage', ascending=False)
    
    if len(missing_df) > 0:
        st.dataframe(safe_dataframe_display(missing_df), use_container_width=True)
        
        # Interactive imputation
        st.markdown("#### 🛠️ Handle Missing Values")
        
        # Select columns for imputation
        impute_columns = st.multiselect(
            "Select columns to handle missing values",
            options=missing_df['Column'].tolist(),
            help="Choose columns where you want to handle missing values"
        )
        
        if impute_columns:
            imputation_config = {}
            
            for col in impute_columns:
                st.markdown(f"**📊 {col}** (Missing: {missing_df[missing_df['Column'] == col]['Missing Percentage'].iloc[0]:.1f}%)")
                
                col_type = str(df[col].dtype)
                is_numeric = df[col].dtype in ['int64', 'float64', 'Int64', 'Float64']
                
                impute_col1, impute_col2 = st.columns([2, 1])
                
                with impute_col1:
                    if is_numeric:
                        method_options = ["Mean", "Median", "Mode", "Forward Fill", "Backward Fill", "Interpolate", "KNN", "Drop Rows"]
                        recommended = "Median" if df[col].skew() > 1 else "Mean"
                    else:
                        method_options = ["Mode", "Forward Fill", "Backward Fill", "Custom Value", "Drop Rows"]
                        recommended = "Mode"
                    
                    selected_method = st.selectbox(
                        f"Method for {col}",
                        options=method_options,
                        index=method_options.index(recommended),
                        key=f"impute_method_{col}",
                        help=f"Recommended: {recommended}"
                    )
                    
                    # Custom value input if needed
                    custom_value = None
                    if selected_method == "Custom Value":
                        custom_value = st.text_input(f"Custom value for {col}", key=f"custom_value_{col}")
                    
                    imputation_config[col] = {
                        'method': selected_method,
                        'custom_value': custom_value
                    }
                
                with impute_col2:
                    st.info(f"💡 **{recommended}**\nRecommended for this column")
            
            # Apply imputation
            if st.button("🔧 Apply Imputation", type="primary"):
                df_imputed = apply_imputation(df, imputation_config)
                st.session_state.modified_df = df_imputed
                
                # Log operation
                st.session_state.operation_log = st.session_state.get('operation_log', [])
                st.session_state.operation_log.append({
                    'operation': 'Missing Value Imputation',
                    'details': f"Applied imputation to {len(imputation_config)} columns",
                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'methods': imputation_config
                })
                
                st.success("✅ Imputation applied successfully!")
                st.rerun()
    
    else:
        st.success("✅ No missing values found!")

def apply_imputation(df: pd.DataFrame, imputation_config: Dict[str, Dict]) -> pd.DataFrame:
    """Apply imputation methods to dataframe with Arrow-safe calculations"""
    df_imputed = df.copy()
    
    for col, config in imputation_config.items():
        method = config['method']
        
        try:
            if method == "Mean":
                # Safe mean calculation for Arrow compatibility
                mean_val = safe_mean(df_imputed[col])
                df_imputed[col] = df_imputed[col].fillna(mean_val)
            elif method == "Median":
                # Safe median calculation for Arrow compatibility
                median_val = safe_median(df_imputed[col])
                df_imputed[col] = df_imputed[col].fillna(median_val)
            elif method == "Mode":
                # Safe mode calculation for Arrow compatibility
                mode_val = safe_mode(df_imputed[col])
                df_imputed[col] = df_imputed[col].fillna(mode_val)
            elif method == "Forward Fill":
                df_imputed[col] = df_imputed[col].fillna(method='ffill')
            elif method == "Backward Fill":
                df_imputed[col] = df_imputed[col].fillna(method='bfill')
            elif method == "Interpolate":
                df_imputed[col] = df_imputed[col].interpolate()
            elif method == "KNN":
                # Use KNN imputer for numeric columns
                if df_imputed[col].dtype in ['int64', 'float64', 'Int64', 'Float64']:
                    from sklearn.impute import KNNImputer
                    imputer = KNNImputer(n_neighbors=5)
                    df_imputed[col] = imputer.fit_transform(df_imputed[[col]]).flatten()
            elif method == "Custom Value":
                df_imputed[col] = df_imputed[col].fillna(config['custom_value'])
            elif method == "Drop Rows":
                df_imputed = df_imputed.dropna(subset=[col])
        
        except Exception as e:
            st.warning(f"⚠️ Error applying {method} to {col}: {str(e)}")
    
    # Make the final DataFrame Arrow-compatible
    df_imputed = make_dataframe_arrow_compatible(df_imputed)
    
    return df_imputed

def render_operation_tracker():
    """Render operation tracking interface"""
    st.markdown("### 📝 Operation History")
    
    if 'operation_log' not in st.session_state:
        st.session_state.operation_log = []
    
    if st.session_state.operation_log:
        # Display operation history
        st.markdown("#### 📋 Performed Operations")
        
        for i, operation in enumerate(reversed(st.session_state.operation_log), 1):
            with st.expander(f"🔄 {operation['operation']} - {operation['timestamp']}", expanded=False):
                st.markdown(f"**Details:** {operation['details']}")
                if 'changes' in operation:
                    st.json(operation['changes'])
                if 'methods' in operation:                    st.json(operation['methods'])
        
        # Dataset comparison
        if 'modified_df' in st.session_state and 'original_df' in st.session_state:
            st.markdown("#### 📊 Dataset Comparison")
            
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.markdown("**Original Dataset**")
                orig_df = st.session_state.original_df
                st.metric("Rows", len(orig_df))
                st.metric("Columns", len(orig_df.columns))
                st.metric("Missing Values", int(orig_df.isnull().sum().sum()))
            
            with comp_col2:
                st.markdown("**Current Dataset**")
                curr_df = st.session_state.modified_df
                st.metric("Rows", len(curr_df), delta=int(len(curr_df) - len(orig_df)))
                st.metric("Columns", len(curr_df.columns), delta=int(len(curr_df.columns) - len(orig_df.columns)))
                
                # Convert numpy.int64 to Python int for Streamlit compatibility
                curr_missing = int(curr_df.isnull().sum().sum())
                orig_missing = int(orig_df.isnull().sum().sum())
                missing_delta = int(curr_missing - orig_missing)
                
                st.metric("Missing Values", curr_missing, delta=missing_delta)
        
        # Reset button
        if st.button("🔄 Reset to Original", type="secondary"):
            if 'original_df' in st.session_state:
                st.session_state.modified_df = st.session_state.original_df.copy()
                st.session_state.operation_log = []
                st.success("✅ Reset to original dataset!")
                st.rerun()
    
    else:
        st.info("📝 No operations performed yet.")

def render_advanced_feature_selection(df: pd.DataFrame, target_column: str):
    """Render advanced feature selection interface"""
    st.markdown("### 🎯 Advanced Feature Selection")
    
    # Initialize feature selection state
    if 'selected_features' not in st.session_state:
        all_features = [col for col in df.columns if col != target_column]
        st.session_state.selected_features = all_features.copy()
    
    # Feature overview
    all_features = [col for col in df.columns if col != target_column]
    
    st.markdown("#### 📊 Feature Overview")
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    with feat_col1:
        st.metric("Total Features", len(all_features))
    with feat_col2:
        st.metric("Selected Features", len(st.session_state.selected_features))
    with feat_col3:
        st.metric("Target Column", target_column)
    
    # Feature selection tabs
    selection_tab1, selection_tab2, selection_tab3, selection_tab4 = st.tabs([
        "📋 Manual Selection", 
        "📈 Correlation Analysis", 
        "🔍 Statistical Selection",
        "🧠 Smart Suggestions"
    ])
    
    with selection_tab1:
        render_manual_feature_selection(df, target_column, all_features)
    
    with selection_tab2:
        render_correlation_analysis(df, target_column, all_features)
    
    with selection_tab3:
        render_statistical_feature_selection(df, target_column, all_features)
    
    with selection_tab4:
        render_smart_feature_suggestions(df, target_column, all_features)

def render_manual_feature_selection(df: pd.DataFrame, target_column: str, all_features: List[str]):
    """Render manual feature selection interface"""
    st.markdown("#### 🎛️ Choose Features Manually")
    
    # Feature selector
    selected_features = st.multiselect(
        "Select features to include",
        options=all_features,
        default=st.session_state.selected_features,
        help="Choose which features to include in your model"
    )
      # Feature information table
    if selected_features:
        feature_info = []
        for feature in selected_features:
            # Handle Mean/Mode calculation more safely
            if df[feature].dtype in ['int64', 'float64']:
                mean_mode_value = f"{df[feature].mean():.2f}" if not pd.isna(df[feature].mean()) else "N/A"
            else:
                mode_series = df[feature].mode()
                mean_mode_value = str(mode_series.iloc[0]) if len(mode_series) > 0 else "N/A"
            
            feature_info.append({
                'Feature': feature,
                'Type': 'Numeric' if df[feature].dtype in ['int64', 'float64'] else 'Categorical',
                'Missing %': f"{df[feature].isnull().sum() / len(df) * 100:.1f}%",
                'Unique Values': df[feature].nunique(),
                'Mean/Mode': mean_mode_value
            })
        
        feature_df = pd.DataFrame(feature_info)
        st.dataframe(safe_dataframe_display(feature_df), use_container_width=True)
        
        # Update session state
        if st.button("✅ Update Feature Selection"):
            st.session_state.selected_features = selected_features
            st.session_state.operation_log = st.session_state.get('operation_log', [])
            st.session_state.operation_log.append({
                'operation': 'Manual Feature Selection',
                'details': f"Selected {len(selected_features)} features manually",
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'features': selected_features
            })
            st.success(f"✅ Updated feature selection: {len(selected_features)} features selected")

def render_correlation_analysis(df: pd.DataFrame, target_column: str, all_features: List[str]):
    """Render correlation analysis interface"""
    st.markdown("#### 📈 Correlation with Target")
    
    # Calculate correlations
    numeric_features = [col for col in all_features if df[col].dtype in ['int64', 'float64']]
    
    if len(numeric_features) > 0:
        # Target encoding for correlation calculation
        target_encoded = df[target_column]
        if df[target_column].dtype == 'object':
            le = LabelEncoder()
            target_encoded = le.fit_transform(df[target_column].astype(str))
        
        correlations = []
        for feature in numeric_features:
            corr = df[feature].corr(pd.Series(target_encoded))
            correlations.append({
                'Feature': feature,
                'Correlation': corr,
                'Abs Correlation': abs(corr),
                'Strength': 'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.3 else 'Weak'
            })
        
        corr_df = pd.DataFrame(correlations).sort_values('Abs Correlation', ascending=False)
        
        # Display correlation table
        st.dataframe(safe_dataframe_display(corr_df), use_container_width=True)
          # Correlation visualization
        if len(correlations) > 0:
            fig = px.bar(
                corr_df.head(10), 
                x='Feature', 
                y='Correlation',
                color='Abs Correlation',
                title='Top 10 Feature Correlations with Target',
                color_continuous_scale='viridis'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Select features by correlation threshold
        st.markdown("**🎚️ Select by Correlation Threshold**")
        corr_threshold = st.slider(
            "Minimum correlation threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="Select features with correlation above this threshold"
        )
        
        high_corr_features = [row['Feature'] for _, row in corr_df.iterrows() if row['Abs Correlation'] >= corr_threshold]
        
        if st.button(f"🎯 Select {len(high_corr_features)} High-Correlation Features"):
            # Include non-numeric features as well
            all_selected = high_corr_features + [col for col in all_features if col not in numeric_features]
            st.session_state.selected_features = all_selected
            st.session_state.operation_log = st.session_state.get('operation_log', [])
            st.session_state.operation_log.append({
                'operation': 'Correlation-Based Feature Selection',
                'details': f"Selected {len(all_selected)} features with correlation >= {corr_threshold}",
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'threshold': corr_threshold,
                'features': all_selected
            })
            st.success(f"✅ Selected {len(all_selected)} features based on correlation analysis")
            st.rerun()
    
    else:
        st.info("📊 No numeric features available for correlation analysis")

def render_statistical_feature_selection(df: pd.DataFrame, target_column: str, all_features: List[str]):
    """Render statistical feature selection interface"""
    st.markdown("#### 🔍 Statistical Feature Selection")
    
    # Prepare data for statistical selection
    X = df[all_features].copy()
    y = df[target_column]
    
    # Encode categorical features
    categorical_features = [col for col in all_features if df[col].dtype == 'object']
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Encode target if categorical
    if y.dtype == 'object':
        target_le = LabelEncoder()
        y = target_le.fit_transform(y.astype(str))
    
    # Statistical methods
    method = st.selectbox(
        "Select statistical method",
        options=["f_classif", "mutual_info_classif", "chi2"],
        help="Choose the statistical method for feature selection"
    )
    
    k_features = st.slider(
        "Number of features to select",
        min_value=1,
        max_value=len(all_features),
        value=min(10, len(all_features)),
        help="Number of top features to select"
    )
    
    if st.button("🔬 Run Statistical Selection"):
        try:
            if method == "f_classif":
                selector = SelectKBest(score_func=f_classif, k=k_features)
            elif method == "mutual_info_classif":
                selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
            else:  # chi2
                from sklearn.feature_selection import chi2
                # Make features non-negative for chi2
                X_chi2 = X - X.min() + 1
                selector = SelectKBest(score_func=chi2, k=k_features)
                X = X_chi2
            
            X_selected = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True)
            selected_features = [all_features[i] for i in selected_indices]
            feature_scores = selector.scores_
            
            # Display results
            score_df = pd.DataFrame({
                'Feature': all_features,
                'Score': feature_scores,
                'Selected': [i in selected_indices for i in range(len(all_features))]
            }).sort_values('Score', ascending=False)
            
            st.dataframe(safe_dataframe_display(score_df), use_container_width=True)
              # Visualization
            fig = px.bar(
                score_df.head(15),
                x='Feature',
                y='Score',
                color='Selected',
                title=f'Feature Scores ({method})',
                color_discrete_map={True: 'green', False: 'lightgray'}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Apply selection
            if st.button(f"✅ Apply Statistical Selection ({len(selected_features)} features)"):
                st.session_state.selected_features = selected_features
                st.session_state.operation_log = st.session_state.get('operation_log', [])
                st.session_state.operation_log.append({
                    'operation': 'Statistical Feature Selection',
                    'details': f"Selected {len(selected_features)} features using {method}",
                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'method': method,
                    'k_features': k_features,
                    'features': selected_features
                })
                st.success(f"✅ Applied statistical feature selection: {len(selected_features)} features")
                st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error in statistical selection: {str(e)}")

def render_smart_feature_suggestions(df: pd.DataFrame, target_column: str, all_features: List[str]):
    """Render smart feature engineering suggestions"""
    st.markdown("#### 🧠 Smart Feature Engineering Suggestions")
    
    suggestions = []
    
    # Analyze features for suggestions
    numeric_features = [col for col in all_features if df[col].dtype in ['int64', 'float64']]
    categorical_features = [col for col in all_features if df[col].dtype == 'object']
    
    # Suggestion 1: Feature combinations
    if len(numeric_features) >= 2:
        suggestions.append({
            'title': '🔢 Numeric Feature Combinations',
            'description': 'Create new features by combining existing numeric features',
            'action': 'feature_combinations',
            'details': f'Available numeric features: {len(numeric_features)}'
        })
    
    # Suggestion 2: Categorical encoding improvements
    if len(categorical_features) > 0:
        high_cardinality_cats = [col for col in categorical_features if df[col].nunique() > 10]
        if high_cardinality_cats:
            suggestions.append({
                'title': '🏷️ Advanced Categorical Encoding',
                'description': 'Apply target encoding or one-hot encoding to categorical features',
                'action': 'categorical_encoding',
                'details': f'High cardinality features: {high_cardinality_cats}'
            })
    
    # Suggestion 3: Feature scaling
    if len(numeric_features) > 1:
        # Check if features have different scales
        ranges = {col: df[col].max() - df[col].min() for col in numeric_features}
        max_range = max(ranges.values())
        min_range = min(r for r in ranges.values() if r > 0)
        
        if max_range / min_range > 100:
            suggestions.append({
                'title': '📏 Feature Scaling',
                'description': 'Standardize features with different scales',
                'action': 'feature_scaling',
                'details': f'Scale ratio: {max_range/min_range:.1f}:1'
            })
    
    # Suggestion 4: Dimensionality reduction
    if len(all_features) > 20:
        suggestions.append({
            'title': '🎯 Dimensionality Reduction',
            'description': 'Apply PCA to reduce feature dimensions',
            'action': 'dimensionality_reduction',
            'details': f'Current features: {len(all_features)}'
        })
    
    # Suggestion 5: Feature interaction detection
    if len(numeric_features) >= 2:
        suggestions.append({
            'title': '🔗 Feature Interactions',
            'description': 'Detect and create interaction features',
            'action': 'feature_interactions',
            'details': 'Polynomial and interaction features'
        })
    
    # Display suggestions
    if suggestions:
        for suggestion in suggestions:
            with st.expander(f"💡 {suggestion['title']}", expanded=False):
                st.markdown(f"**Description:** {suggestion['description']}")
                st.markdown(f"**Details:** {suggestion['details']}")
                
                if st.button(f"🚀 Apply {suggestion['title']}", key=f"apply_{suggestion['action']}"):
                    apply_feature_suggestion(df, suggestion['action'], all_features, target_column)
    
    else:
        st.info("🤖 No specific feature engineering suggestions for this dataset.")

def apply_feature_suggestion(df: pd.DataFrame, action: str, all_features: List[str], target_column: str):
    """Apply feature engineering suggestions"""
    
    if action == 'feature_combinations':
        st.markdown("**🔢 Feature Combination Options**")
        
        numeric_features = [col for col in all_features if df[col].dtype in ['int64', 'float64']]
        
        combination_type = st.selectbox(
            "Select combination type",
            options=["Sum", "Difference", "Product", "Ratio", "All"],
            help="Choose how to combine features"
        )
        
        feature_pairs = st.multiselect(
            "Select feature pairs to combine",
            options=[(f1, f2) for i, f1 in enumerate(numeric_features) for f2 in numeric_features[i+1:]],
            format_func=lambda x: f"{x[0]} + {x[1]}",
            help="Choose pairs of features to combine"
        )
        
        if feature_pairs and st.button("✅ Create Combined Features"):
            df_combined = create_combined_features(df, feature_pairs, combination_type)
            st.session_state.modified_df = df_combined
            st.session_state.operation_log = st.session_state.get('operation_log', [])
            st.session_state.operation_log.append({
                'operation': 'Feature Combination',
                'details': f"Created {len(feature_pairs)} combined features using {combination_type}",
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'pairs': feature_pairs,
                'method': combination_type
            })
            st.success("✅ Combined features created successfully!")
            st.rerun()
    
    # Add other feature engineering implementations here...

def create_combined_features(df: pd.DataFrame, feature_pairs: List[Tuple[str, str]], combination_type: str) -> pd.DataFrame:
    """Create combined features from feature pairs"""
    df_combined = df.copy()
    
    for f1, f2 in feature_pairs:
        if combination_type in ["Sum", "All"]:
            df_combined[f"{f1}_plus_{f2}"] = df_combined[f1] + df_combined[f2]
        if combination_type in ["Difference", "All"]:
            df_combined[f"{f1}_minus_{f2}"] = df_combined[f1] - df_combined[f2]
        if combination_type in ["Product", "All"]:
            df_combined[f"{f1}_times_{f2}"] = df_combined[f1] * df_combined[f2]
        if combination_type in ["Ratio", "All"]:
            # Avoid division by zero
            df_combined[f"{f1}_div_{f2}"] = df_combined[f1] / (df_combined[f2] + 1e-8)
    
    return df_combined

def create_column_type_selector(df: pd.DataFrame, column: str) -> str:
    """Create a column type selector for UI"""
    current_type = str(df[column].dtype)
    type_options = ['int64', 'float64', 'object', 'category', 'bool']
    
    # Determine best options based on data
    if df[column].dtype in ['int64', 'float64']:
        recommended = current_type
    elif df[column].dtype == 'object':
        # Check if it can be converted to numeric
        try:
            pd.to_numeric(df[column], errors='coerce')
            recommended = 'float64'
        except:
            recommended = 'object'
    else:
        recommended = current_type
    
    return recommended


def recommend_imputation_method(series: pd.Series, column_name: str) -> str:
    """Recommend imputation method based on data characteristics"""
    
    if series.dtype in ['int64', 'float64', 'Int64', 'Float64']:
        # For numeric data, recommend median for skewed data, mean for normal
        try:
            skewness = abs(series.skew())
            if skewness > 2:  # Highly skewed
                return 'Median'
            else:
                return 'Mean'
        except:
            return 'Median'
    else:
        # For categorical data, recommend mode
        return 'Mode'
