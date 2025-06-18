"""
Enhanced data preparation utilities with auto-preparation tools and suggestions
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from utils.logging_config import get_logger

# Optional imports for advanced features
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False
    logger = get_logger('data_preparation_enhanced')
    logger.warning("imbalanced-learn not available. Install with: pip install imbalanced-learn")

logger = get_logger('data_preparation_enhanced')


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


class DataPreparationTools:
    """Enhanced data preparation toolkit with automated suggestions"""
    
    def __init__(self):
        self.preparation_history = []
        
    def analyze_preparation_needs(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Analyze dataset and suggest preparation steps
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            
        Returns:
            Dictionary containing analysis and suggestions
        """
        issues = []
        suggestions = []
        auto_fixes = []
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # 1. Missing values analysis
        missing_features = X.isnull().sum()
        missing_target = y.isnull().sum()
        
        if missing_target > 0:
            issues.append(f"Target column has {missing_target} missing values")
            suggestions.append("Remove rows with missing target values")
            auto_fixes.append("remove_missing_target")
        
        if missing_features.sum() > 0:
            high_missing = missing_features[missing_features > len(df) * 0.5]
            if len(high_missing) > 0:
                issues.append(f"Features with >50% missing values: {list(high_missing.index)}")
                suggestions.append("Consider dropping high-missing features or using advanced imputation")
                auto_fixes.append("drop_high_missing_features")
            
            moderate_missing = missing_features[(missing_features > 0) & (missing_features <= len(df) * 0.5)]
            if len(moderate_missing) > 0:
                issues.append(f"Features with missing values: {list(moderate_missing.index)}")
                suggestions.append("Impute missing values using mean/median/mode")
                auto_fixes.append("impute_missing_values")
        
        # 2. Duplicate rows analysis
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Dataset has {duplicates} duplicate rows")
            suggestions.append("Remove duplicate rows")
            auto_fixes.append("remove_duplicates")
        
        # 3. Class imbalance analysis
        class_counts = y.value_counts()
        min_class_size = class_counts.min()
        max_class_size = class_counts.max()
        imbalance_ratio = max_class_size / min_class_size
        
        if min_class_size < 2:
            issues.append(f"Some classes have only {min_class_size} sample(s) - will cause stratification error")
            suggestions.append("Remove classes with <2 samples or use non-stratified split")
            auto_fixes.append("handle_small_classes")
        elif imbalance_ratio > 5:
            issues.append(f"Class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
            suggestions.append("Consider using SMOTE, over-sampling, or under-sampling")
            auto_fixes.append("balance_classes")
        
        # 4. Feature scaling analysis
        numeric_features = X.select_dtypes(include=[np.number]).columns
        if len(numeric_features) > 0:
            # Check if features have very different scales
            numeric_data = X[numeric_features]
            ranges = numeric_data.max() - numeric_data.min()
            max_range = ranges.max()
            min_range = ranges[ranges > 0].min() if len(ranges[ranges > 0]) > 0 else 1
            
            if max_range / min_range > 100:
                issues.append("Features have very different scales")
                suggestions.append("Consider feature scaling (StandardScaler, MinMaxScaler, or RobustScaler)")
                auto_fixes.append("scale_features")
        
        # 5. Categorical features analysis
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_features) > 0:
            high_cardinality = []
            for col in categorical_features:
                unique_count = X[col].nunique()
                if unique_count > 50:
                    high_cardinality.append(col)
            
            if high_cardinality:
                issues.append(f"High cardinality categorical features: {high_cardinality}")
                suggestions.append("Consider feature engineering or dimensionality reduction")
                auto_fixes.append("handle_high_cardinality")
          # 6. Feature selection analysis
        if len(X.columns) > 20:
            issues.append(f"Dataset has {len(X.columns)} features")
            suggestions.append("Consider feature selection to reduce dimensionality")
            auto_fixes.append("select_features")
        
        # 7. Outlier analysis
        numeric_features = X.select_dtypes(include=[np.number]).columns
        if len(numeric_features) > 0:
            outlier_count = 0
            for col in numeric_features:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = X[(X[col] < Q1 - 1.5 * IQR) | (X[col] > Q3 + 1.5 * IQR)]
                outlier_count += len(outliers)
            
            if outlier_count > 0:
                issues.append(f"Detected {outlier_count} potential outliers")
                suggestions.append("Consider outlier detection and removal")
                auto_fixes.append("detect_outliers")
        
        return {
            'issues': issues,
            'suggestions': suggestions,
            'auto_fixes': auto_fixes,
            'class_distribution': class_counts.to_dict(),
            'missing_summary': {
                'total_missing': missing_features.sum(),
                'features_with_missing': len(missing_features[missing_features > 0]),
                'high_missing_features': list(missing_features[missing_features > len(df) * 0.5].index)
            },
            'feature_info': {
                'numeric_features': len(numeric_features),
                'categorical_features': len(categorical_features),
                'total_features': len(X.columns)
            }
        }
    
    def get_preparation_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get prioritized list of preparation recommendations
        
        Args:
            analysis: Analysis results from analyze_preparation_needs
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Priority 1: Critical issues that prevent training
        if 'handle_small_classes' in analysis['auto_fixes']:
            recommendations.append({
                'priority': 'Critical',
                'title': '🚨 Fix Small Classes',
                'description': 'Some classes have too few samples for stratified splitting',
                'action': 'handle_small_classes',
                'auto_fixable': True,
                'impact': 'Prevents training errors'
            })
        
        if 'remove_missing_target' in analysis['auto_fixes']:
            recommendations.append({
                'priority': 'Critical',
                'title': '🎯 Clean Target Column',
                'description': 'Remove rows with missing target values',
                'action': 'remove_missing_target',
                'auto_fixable': True,
                'impact': 'Required for supervised learning'
            })
        
        # Priority 2: High impact improvements
        if 'remove_duplicates' in analysis['auto_fixes']:
            recommendations.append({
                'priority': 'High',
                'title': '🔄 Remove Duplicates',
                'description': 'Remove duplicate rows to improve data quality',
                'action': 'remove_duplicates',
                'auto_fixable': True,
                'impact': 'Improves model generalization'
            })
        
        if 'balance_classes' in analysis['auto_fixes']:
            recommendations.append({
                'priority': 'High',
                'title': '⚖️ Balance Classes',
                'description': 'Address class imbalance using SMOTE or sampling',
                'action': 'balance_classes',
                'auto_fixable': True,
                'impact': 'Improves model performance on minority classes'
            })
        
        # Priority 3: Medium impact improvements
        if 'impute_missing_values' in analysis['auto_fixes']:
            recommendations.append({
                'priority': 'Medium',
                'title': '🔧 Handle Missing Values',
                'description': 'Impute missing values in feature columns',
                'action': 'impute_missing_values',
                'auto_fixable': True,
                'impact': 'Increases data utilization'
            })
        
        if 'scale_features' in analysis['auto_fixes']:
            recommendations.append({
                'priority': 'Medium',
                'title': '📏 Scale Features',
                'description': 'Normalize feature scales for better model performance',
                'action': 'scale_features',
                'auto_fixable': True,
                'impact': 'Improves gradient-based algorithms'
            })
          # Priority 4: Optional improvements
        if 'drop_high_missing_features' in analysis['auto_fixes']:
            recommendations.append({
                'priority': 'Low',
                'title': '🗑️ Drop High-Missing Features',
                'description': 'Remove features with >50% missing values',
                'action': 'drop_high_missing_features',
                'auto_fixable': True,
                'impact': 'Reduces noise, may lose information'
            })
        
        if 'handle_high_cardinality' in analysis['auto_fixes']:
            recommendations.append({
                'priority': 'Low',
                'title': '🏷️ Reduce High Cardinality',
                'description': 'Group rare categories in high-cardinality features',
                'action': 'handle_high_cardinality',
                'auto_fixable': True,
                'impact': 'Simplifies categorical features'
            })
        
        if 'select_features' in analysis['auto_fixes']:
            recommendations.append({
                'priority': 'Low',
                'title': '🎯 Feature Selection',
                'description': 'Select most important features',
                'action': 'select_features',
                'auto_fixable': True,
                'impact': 'Reduces overfitting and training time'
            })
        
        if 'detect_outliers' in analysis['auto_fixes']:
            recommendations.append({
                'priority': 'Low',
                'title': '🔍 Outlier Detection',
                'description': 'Detect and remove statistical outliers',
                'action': 'detect_outliers',
                'auto_fixable': True,
                'impact': 'May improve model robustness'
            })
        
        return recommendations
    
    def auto_prepare_dataset(self, df: pd.DataFrame, target_column: str, 
                           selected_fixes: List[str] = None, 
                           test_size: float = 0.2, 
                           random_state: int = 42) -> Dict[str, Any]:
        """
        Automatically prepare dataset with selected fixes
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            selected_fixes: List of fixes to apply (if None, applies all critical fixes)
            test_size: Proportion of test set
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary containing prepared dataset components and preparation log
        """
        preparation_log = []
        df_processed = df.copy()
        
        # Analyze preparation needs
        analysis = self.analyze_preparation_needs(df_processed, target_column)
        
        # Apply critical fixes by default if none specified
        if selected_fixes is None:
            selected_fixes = ['handle_small_classes', 'remove_missing_target', 'remove_duplicates']
          # Apply selected fixes in order of priority
        fix_order = [
            'remove_missing_target',
            'handle_small_classes', 
            'remove_duplicates',
            'drop_high_missing_features',
            'impute_missing_values',
            'balance_classes',
            'scale_features',
            'handle_high_cardinality',
            'detect_outliers',
            'select_features'
        ]
        
        for fix in fix_order:
            if fix in selected_fixes:
                df_processed, log_entry = self._apply_fix(df_processed, target_column, fix)
                preparation_log.append(log_entry)
        
        # Final preparation for ML
        try:
            prepared_data = self._prepare_for_ml_safe(df_processed, target_column, test_size, random_state)
            preparation_log.append({
                'step': 'final_preparation',
                'action': 'Create ML-ready dataset',
                'status': 'success',
                'details': f'Successfully split into train/test sets'
            })
            
            prepared_data['preparation_log'] = preparation_log
            return prepared_data
            
        except Exception as e:
            logger.error(f"Error in final ML preparation: {str(e)}")
            preparation_log.append({
                'step': 'final_preparation',
                'action': 'Create ML-ready dataset',
                'status': 'error',
                'details': str(e)
            })
            raise e
    
    def _apply_fix(self, df: pd.DataFrame, target_column: str, fix_type: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply a specific data preparation fix"""
        df_result = df.copy()
        log_entry = {
            'step': fix_type,
            'action': '',
            'status': 'success',
            'details': ''
        }
        
        try:
            if fix_type == 'remove_missing_target':
                initial_rows = len(df_result)
                df_result = df_result.dropna(subset=[target_column])
                removed_rows = initial_rows - len(df_result)
                log_entry.update({
                    'action': 'Remove missing target values',
                    'details': f'Removed {removed_rows} rows with missing target values'
                })
            
            elif fix_type == 'handle_small_classes':
                y = df_result[target_column]
                class_counts = y.value_counts()
                small_classes = class_counts[class_counts < 2].index
                
                if len(small_classes) > 0:
                    # Remove classes with <2 samples
                    initial_rows = len(df_result)
                    df_result = df_result[~df_result[target_column].isin(small_classes)]
                    removed_rows = initial_rows - len(df_result)
                    log_entry.update({
                        'action': 'Remove small classes',
                        'details': f'Removed {removed_rows} rows from classes with <2 samples: {list(small_classes)}'
                    })
                else:
                    log_entry.update({
                        'action': 'Check small classes',
                        'details': 'No classes with <2 samples found'
                    })
            
            elif fix_type == 'remove_duplicates':
                initial_rows = len(df_result)
                df_result = df_result.drop_duplicates()
                removed_rows = initial_rows - len(df_result)
                log_entry.update({
                    'action': 'Remove duplicate rows',
                    'details': f'Removed {removed_rows} duplicate rows'
                })
            
            elif fix_type == 'drop_high_missing_features':
                X = df_result.drop(columns=[target_column])
                missing_threshold = len(df_result) * 0.5
                high_missing_cols = X.columns[X.isnull().sum() > missing_threshold].tolist()
                
                if high_missing_cols:
                    df_result = df_result.drop(columns=high_missing_cols)
                    log_entry.update({
                        'action': 'Drop high-missing features',
                        'details': f'Dropped {len(high_missing_cols)} features with >50% missing: {high_missing_cols}'
                    })
                else:
                    log_entry.update({
                        'action': 'Check high-missing features',
                        'details': 'No features with >50% missing values found'
                    })
            
            elif fix_type == 'impute_missing_values':
                X = df_result.drop(columns=[target_column])
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                categorical_cols = X.select_dtypes(include=['object', 'category']).columns
                
                imputed_features = []
                  # Impute numeric features with median
                if len(numeric_cols) > 0:
                    for col in numeric_cols:
                        if X[col].isnull().sum() > 0:
                            median_val = safe_median(X[col])
                            df_result[col] = df_result[col].fillna(median_val)
                            imputed_features.append(f"{col} (median)")
                
                # Impute categorical features with mode
                if len(categorical_cols) > 0:
                    for col in categorical_cols:
                        if X[col].isnull().sum() > 0:
                            mode_val = safe_mode(X[col])
                            df_result[col] = df_result[col].fillna(mode_val)
                            imputed_features.append(f"{col} (mode)")
                
                log_entry.update({
                    'action': 'Impute missing values',
                    'details': f'Imputed {len(imputed_features)} features: {imputed_features}'
                })
            
            elif fix_type == 'balance_classes':
                if IMBALANCED_LEARN_AVAILABLE:
                    X = df_result.drop(columns=[target_column])
                    y = df_result[target_column]
                    
                    # Encode categorical features temporarily for SMOTE
                    categorical_columns = list(X.select_dtypes(include=['object', 'category']).columns)
                    temp_encoders = {}
                    X_encoded = X.copy()
                    
                    for col in categorical_columns:
                        le = LabelEncoder()
                        X_encoded[col] = le.fit_transform(X[col].astype(str))
                        temp_encoders[col] = le
                    
                    # Encode target if categorical
                    if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
                        target_le = LabelEncoder()
                        y_encoded = target_le.fit_transform(y.astype(str))
                    else:
                        y_encoded = y
                        target_le = None
                    
                    # Apply SMOTE
                    try:
                        smote = SMOTE(random_state=42)
                        X_balanced, y_balanced = smote.fit_resample(X_encoded, y_encoded)
                        
                        # Decode categorical features back
                        for col in categorical_columns:
                            X_balanced[col] = temp_encoders[col].inverse_transform(X_balanced[col].astype(int))
                        
                        # Decode target back
                        if target_le:
                            y_balanced = target_le.inverse_transform(y_balanced)
                        
                        # Recreate dataframe
                        df_balanced = X_balanced.copy()
                        df_balanced[target_column] = y_balanced
                        df_result = df_balanced
                        
                        original_samples = len(y)
                        new_samples = len(y_balanced)
                        log_entry.update({
                            'action': 'Balance classes with SMOTE',
                            'details': f'Increased samples from {original_samples} to {new_samples} using SMOTE'
                        })
                    except Exception as smote_error:
                        log_entry.update({
                            'action': 'Balance classes (SMOTE failed)',
                            'status': 'warning',
                            'details': f'SMOTE failed: {str(smote_error)}. Consider manual balancing.'
                        })
                else:
                    log_entry.update({
                        'action': 'Balance classes (not available)',
                        'status': 'warning',
                        'details': 'imbalanced-learn not available. Install with: pip install imbalanced-learn'
                    })
            
            elif fix_type == 'scale_features':
                X = df_result.drop(columns=[target_column])
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) > 0:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    
                    # Scale numeric features
                    X_scaled = X.copy()
                    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
                    
                    # Update dataframe
                    df_result = X_scaled.copy()
                    df_result[target_column] = df_result.get(target_column, df[target_column])
                    
                    log_entry.update({
                        'action': 'Scale numeric features',
                        'details': f'Applied StandardScaler to {len(numeric_cols)} numeric features'
                    })
                else:
                    log_entry.update({
                        'action': 'Scale features (no numeric features)',
                        'details': 'No numeric features found to scale'
                    })
            
            elif fix_type == 'handle_high_cardinality':
                X = df_result.drop(columns=[target_column])
                categorical_cols = X.select_dtypes(include=['object', 'category']).columns
                
                modified_features = []
                for col in categorical_cols:
                    unique_count = X[col].nunique()
                    if unique_count > 50:
                        # Keep top 20 categories, group others as 'Other'
                        top_categories = X[col].value_counts().head(20).index
                        df_result[col] = df_result[col].apply(
                            lambda x: x if x in top_categories else 'Other'
                        )
                        modified_features.append(f"{col} ({unique_count} -> {df_result[col].nunique()})")
                
                if modified_features:
                    log_entry.update({
                        'action': 'Reduce high cardinality features',
                        'details': f'Grouped rare categories for: {modified_features}'
                    })
                else:
                    log_entry.update({
                        'action': 'Check high cardinality features',
                        'details': 'No high cardinality features found'
                    })
            
            elif fix_type == 'select_features':
                X = df_result.drop(columns=[target_column])
                y = df_result[target_column]
                
                if len(X.columns) > 20:
                    # Encode categorical features for feature selection
                    X_encoded = X.copy()
                    categorical_columns = list(X.select_dtypes(include=['object', 'category']).columns)
                    
                    for col in categorical_columns:
                        le = LabelEncoder()
                        X_encoded[col] = le.fit_transform(X[col].astype(str))
                    
                    # Encode target if categorical
                    if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
                        target_le = LabelEncoder()
                        y_encoded = target_le.fit_transform(y.astype(str))
                    else:
                        y_encoded = y
                    
                    # Select top features
                    try:
                        k_features = min(20, len(X.columns) // 2)  # Select top 20 or half of features
                        selector = SelectKBest(score_func=f_classif, k=k_features)
                        X_selected = selector.fit_transform(X_encoded, y_encoded)
                        
                        # Get selected feature names
                        selected_indices = selector.get_support(indices=True)
                        selected_features = X.columns[selected_indices]
                        
                        # Update dataframe
                        df_result = X[selected_features].copy()
                        df_result[target_column] = y
                        
                        log_entry.update({
                            'action': 'Feature selection',
                            'details': f'Selected {len(selected_features)} best features from {len(X.columns)}'
                        })
                    except Exception as selection_error:
                        log_entry.update({
                            'action': 'Feature selection (failed)',
                            'status': 'warning',
                            'details': f'Feature selection failed: {str(selection_error)}'
                        })
                else:
                    log_entry.update({
                        'action': 'Feature selection (not needed)',
                        'details': f'Dataset has only {len(X.columns)} features, selection not needed'
                    })
            
            elif fix_type == 'detect_outliers':
                # Simple outlier detection using IQR method
                X = df_result.drop(columns=[target_column])
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                
                outlier_rows = set()
                outlier_info = []
                
                for col in numeric_cols:
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    col_outliers = X[(X[col] < lower_bound) | (X[col] > upper_bound)].index
                    outlier_rows.update(col_outliers)
                    
                    if len(col_outliers) > 0:
                        outlier_info.append(f"{col}: {len(col_outliers)} outliers")
                
                if outlier_rows and len(outlier_rows) < len(df_result) * 0.1:  # Remove only if <10% are outliers
                    df_result = df_result.drop(index=list(outlier_rows))
                    log_entry.update({
                        'action': 'Remove outliers',
                        'details': f'Removed {len(outlier_rows)} outlier rows. Features: {", ".join(outlier_info)}'
                    })
                else:
                    log_entry.update({
                        'action': 'Outlier detection',
                        'details': f'Found {len(outlier_rows)} potential outliers, but kept them (>10% of data or no outliers found)'
                    })

            # ...existing code...
            
        except Exception as e:
            log_entry.update({
                'status': 'error',
                'details': f'Error applying {fix_type}: {str(e)}'
            })
            logger.error(f"Error applying fix {fix_type}: {str(e)}")
        
        return df_result, log_entry
    
    def _prepare_for_ml_safe(self, df: pd.DataFrame, target_column: str, 
                            test_size: float, random_state: int) -> Dict[str, Any]:
        """
        Safely prepare dataset for ML with improved error handling
        """
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical features
        categorical_columns = list(X.select_dtypes(include=['object', 'category']).columns)
        label_encoders = {}
        
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Handle target encoding if categorical
        target_encoder = None
        if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))
        
        # Check class distribution for stratification
        class_counts = pd.Series(y).value_counts()
        min_class_size = class_counts.min()
        
        # Use stratification only if all classes have at least 2 samples
        use_stratify = min_class_size >= 2
        
        if not use_stratify:
            logger.warning(f"Disabling stratification due to small class sizes (min: {min_class_size})")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y if use_stratify else None
        )
        
        # Get feature names and target names
        feature_names = list(X.columns)
        
        if target_encoder:
            target_names = list(target_encoder.classes_)
        else:
            target_names = [str(i) for i in sorted(y.unique())]
        
        prepared_data = {
            'X': X,
            'y': y,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'target_names': target_names,
            'label_encoders': label_encoders,
            'target_encoder': target_encoder,
            'stratified': use_stratify,
            'config': {
                'name': 'User Uploaded Dataset (Auto-Prepared)',
                'type': 'classification',
                'features': len(feature_names),
                'classes': len(target_names),
                'description': f'Auto-prepared dataset with {len(feature_names)} features and {len(target_names)} classes'
            }
        }
        
        logger.info(f"Dataset safely prepared for ML: {X.shape[0]} samples, {X.shape[1]} features, {len(target_names)} classes, stratified: {use_stratify}")
        return prepared_data


# Global instance
data_prep_tools = DataPreparationTools()
