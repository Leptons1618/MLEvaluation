"""
Create enhanced sample datasets for testing new features
"""

import pandas as pd
import numpy as np
import os

def create_enhanced_test_dataset():
    """Create a comprehensive dataset for testing all enhanced features"""
    
    np.random.seed(42)
    n_samples = 500
    
    # Create diverse feature types
    data = {
        # Numeric features with different scales
        'small_numbers': np.random.normal(0, 1, n_samples),
        'large_numbers': np.random.normal(10000, 2000, n_samples),
        'integers': np.random.randint(1, 100, n_samples),
        'percentages': np.random.uniform(0, 100, n_samples),
        
        # Features with different distributions
        'skewed_feature': np.random.exponential(2, n_samples),
        'uniform_feature': np.random.uniform(-10, 10, n_samples),
        'binary_numeric': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        
        # Categorical features
        'low_cardinality_cat': np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'medium_cardinality_cat': np.random.choice([f'Category_{i}' for i in range(20)], n_samples),
        'high_cardinality_cat': [f'Item_{i}' for i in np.random.randint(0, 100, n_samples)],
        
        # Text-like features
        'text_feature': np.random.choice(['Option1', 'Option2', 'Option3', 'Special_Case'], n_samples),
        'mixed_case_text': np.random.choice(['good', 'bad', 'EXCELLENT', 'Poor', 'Average'], n_samples),
        
        # Date-like strings (to test type conversion)
        'date_strings': [f'2023-{month:02d}-{day:02d}' for month, day in 
                        zip(np.random.randint(1, 13, n_samples), np.random.randint(1, 29, n_samples))],
        
        # Boolean-like strings
        'boolean_strings': np.random.choice(['True', 'False', 'yes', 'no'], n_samples),
        
        # Features with correlations
        'correlated_feature_1': np.random.normal(0, 1, n_samples),
    }
    
    # Create correlated feature
    data['correlated_feature_2'] = data['correlated_feature_1'] * 0.8 + np.random.normal(0, 0.5, n_samples)
    
    # Create target based on some features
    target_score = (data['small_numbers'] * 0.5 + 
                   data['correlated_feature_1'] * 0.3 + 
                   np.random.normal(0, 0.2, n_samples))
    
    # Create categorical target
    data['target'] = ['High' if score > 0.5 else 'Medium' if score > -0.5 else 'Low' 
                     for score in target_score]
    
    # Add some class imbalance
    imbalance_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    for idx in imbalance_indices:
        data['target'][idx] = 'Rare_Class'
    
    df = pd.DataFrame(data)
    
    # Introduce missing values strategically
    # Light missing (5%)
    light_missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    df.loc[light_missing_indices, 'percentages'] = np.nan
    
    # Moderate missing (15%)
    moderate_missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
    df.loc[moderate_missing_indices, 'text_feature'] = np.nan
    
    # Heavy missing (40%)
    heavy_missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.4), replace=False)
    df.loc[heavy_missing_indices, 'mixed_case_text'] = np.nan
    
    # Add some duplicates
    duplicate_rows = df.sample(n=20, random_state=42)
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    # Add some outliers
    outlier_indices = np.random.choice(len(df), size=10, replace=False)
    df.loc[outlier_indices, 'small_numbers'] = df['small_numbers'].quantile(0.95) + np.random.normal(5, 1, 10)
    df.loc[outlier_indices, 'large_numbers'] = df['large_numbers'].quantile(0.05) - np.random.normal(5000, 1000, 10)
    
    return df

def create_feature_engineering_dataset():
    """Create a dataset specifically for testing feature engineering"""
    
    np.random.seed(123)
    n_samples = 300
    
    data = {
        # Features that can be combined
        'height_cm': np.random.normal(170, 10, n_samples),
        'weight_kg': np.random.normal(70, 15, n_samples),
        'age_years': np.random.randint(18, 80, n_samples),
        
        # Financial features
        'income': np.random.normal(50000, 20000, n_samples),
        'expenses': np.random.normal(30000, 15000, n_samples),
        'savings': np.random.normal(10000, 8000, n_samples),
        
        # Time-related features
        'hours_worked': np.random.normal(40, 8, n_samples),
        'years_experience': np.random.randint(0, 30, n_samples),
        
        # Categorical features for encoding
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
        'job_category': np.random.choice(['Tech', 'Finance', 'Healthcare', 'Education', 'Other'], n_samples),
        'location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples, p=[0.5, 0.3, 0.2]),
        
        # Target for classification
        'performance_rating': np.random.choice(['Poor', 'Average', 'Good', 'Excellent'], n_samples, p=[0.1, 0.3, 0.4, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    # Create logical relationships
    # BMI calculation opportunity
    df['bmi_opportunity'] = df['weight_kg'] / (df['height_cm'] / 100) ** 2
    
    # Savings rate opportunity  
    df['savings_rate_opportunity'] = (df['income'] - df['expenses']) / df['income'] * 100
    
    # Experience vs age relationship
    df.loc[df['years_experience'] > df['age_years'] - 16, 'years_experience'] = df['age_years'] - 16
    
    return df

def create_correlation_test_dataset():
    """Create dataset for testing correlation analysis"""
    
    np.random.seed(789)
    n_samples = 400
    
    # Create base feature
    base_feature = np.random.normal(0, 1, n_samples)
    
    data = {
        'base_feature': base_feature,
        'highly_correlated': base_feature * 0.9 + np.random.normal(0, 0.1, n_samples),
        'moderately_correlated': base_feature * 0.5 + np.random.normal(0, 0.8, n_samples),
        'weakly_correlated': base_feature * 0.2 + np.random.normal(0, 0.9, n_samples),
        'uncorrelated': np.random.normal(0, 1, n_samples),
        'negatively_correlated': -base_feature * 0.7 + np.random.normal(0, 0.5, n_samples),
        
        # Additional features
        'feature_1': np.random.normal(5, 2, n_samples),
        'feature_2': np.random.exponential(1, n_samples),
        'feature_3': np.random.uniform(-5, 5, n_samples),
        
        # Categorical features
        'category_A': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'category_B': np.random.choice(['P', 'Q', 'R', 'S'], n_samples),
        
        # Target based on multiple features
        'target': np.where(
            (base_feature > 0.5) & (np.random.random(n_samples) > 0.3), 'Positive',
            np.where(base_feature < -0.5, 'Negative', 'Neutral')
        )
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Create sample datasets directory
    os.makedirs("sample_datasets", exist_ok=True)
    
    print("📁 Creating Enhanced Sample Datasets...")
    
    # Dataset 1: Comprehensive testing dataset
    df1 = create_enhanced_test_dataset()
    df1.to_csv("sample_datasets/comprehensive_test_dataset.csv", index=False)
    print(f"✅ Created comprehensive_test_dataset.csv ({df1.shape[0]} rows, {df1.shape[1]} columns)")
    print(f"   - Features: {df1.shape[1] - 1}")
    print(f"   - Missing values: {df1.isnull().sum().sum()}")
    print(f"   - Duplicates: {df1.duplicated().sum()}")
    print(f"   - Target distribution: {df1['target'].value_counts().to_dict()}")
    print(f"   - Data types: {df1.dtypes.value_counts().to_dict()}")
    
    # Dataset 2: Feature engineering dataset
    df2 = create_feature_engineering_dataset()
    df2.to_csv("sample_datasets/feature_engineering_dataset.csv", index=False)
    print(f"\n✅ Created feature_engineering_dataset.csv ({df2.shape[0]} rows, {df2.shape[1]} columns)")
    print(f"   - Designed for feature combination testing")
    print(f"   - BMI calculation opportunity available")
    print(f"   - Savings rate calculation opportunity available")
    
    # Dataset 3: Correlation analysis dataset
    df3 = create_correlation_test_dataset()
    df3.to_csv("sample_datasets/correlation_analysis_dataset.csv", index=False)
    print(f"\n✅ Created correlation_analysis_dataset.csv ({df3.shape[0]} rows, {df3.shape[1]} columns)")
    print(f"   - Features with known correlation patterns")
    print(f"   - Perfect for testing correlation analysis")
    
    print(f"\n🎉 Enhanced sample datasets created successfully!")
    print(f"\n💡 Test these datasets to explore new features:")
    print(f"   1. comprehensive_test_dataset.csv - Test all enhanced features")
    print(f"   2. feature_engineering_dataset.csv - Test feature combinations")
    print(f"   3. correlation_analysis_dataset.csv - Test correlation analysis")
    print(f"   4. Use existing problematic datasets for error handling")
    
    print(f"\n🔧 New Features to Test:")
    print(f"   • Editable column types (convert date_strings to datetime)")
    print(f"   • Enhanced quality report with imputation options")
    print(f"   • Operation tracking and dataset versioning")
    print(f"   • Advanced feature selection with correlation analysis")
    print(f"   • Statistical feature selection methods")
    print(f"   • Smart feature engineering suggestions")
    print(f"   • Interactive feature combination tools")
