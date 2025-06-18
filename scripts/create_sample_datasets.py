"""
Create sample problematic datasets for testing the enhanced data preparation features
"""

import pandas as pd
import numpy as np
import os

def create_problematic_dataset():
    """Create a dataset with multiple issues for testing auto-preparation"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create base data
    n_samples = 100
    
    data = {
        # Numeric features with different scales
        'feature_small_scale': np.random.normal(0, 1, n_samples),  # Scale 0-3
        'feature_large_scale': np.random.normal(1000, 500, n_samples),  # Scale 0-3000
        'feature_with_missing': np.random.normal(50, 10, n_samples),
        
        # Categorical features
        'category_normal': np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'category_high_cardinality': [f'Cat_{i%60}' for i in range(n_samples)],  # 60 unique categories
        'category_with_missing': np.random.choice(['X', 'Y', 'Z'], n_samples, p=[0.5, 0.3, 0.2]),
        
        # Target with class imbalance and small classes
        'target': (['Majority'] * 60 + 
                  ['Minority'] * 25 + 
                  ['Small_Class_1'] * 10 + 
                  ['Small_Class_2'] * 3 + 
                  ['Tiny_Class'] * 1 + 
                  ['Single_Sample'])  # Single sample class
    }
    
    df = pd.DataFrame(data)
    
    # Introduce missing values
    missing_indices = np.random.choice(n_samples, size=20, replace=False)
    df.loc[missing_indices, 'feature_with_missing'] = np.nan
    
    missing_indices_cat = np.random.choice(n_samples, size=15, replace=False)
    df.loc[missing_indices_cat, 'category_with_missing'] = np.nan
    
    # Add some duplicate rows
    duplicate_rows = df.iloc[:5].copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    return df

def create_missing_values_dataset():
    """Create a dataset with heavy missing values"""
    
    np.random.seed(42)
    n_samples = 80
    
    data = {
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2_heavy_missing': np.random.normal(10, 2, n_samples),
        'feature_3_moderate_missing': np.random.normal(5, 1, n_samples),
        'feature_4_light_missing': np.random.normal(20, 3, n_samples),
        'target': np.random.choice(['Class_A', 'Class_B'], n_samples, p=[0.6, 0.4])
    }
    
    df = pd.DataFrame(data)
    
    # Create different levels of missing data
    # Heavy missing (>50%)
    heavy_missing_indices = np.random.choice(n_samples, size=50, replace=False)
    df.loc[heavy_missing_indices, 'feature_2_heavy_missing'] = np.nan
    
    # Moderate missing (~30%)
    moderate_missing_indices = np.random.choice(n_samples, size=25, replace=False)
    df.loc[moderate_missing_indices, 'feature_3_moderate_missing'] = np.nan
    
    # Light missing (~10%)
    light_missing_indices = np.random.choice(n_samples, size=8, replace=False)
    df.loc[light_missing_indices, 'feature_4_light_missing'] = np.nan
    
    return df

def create_single_class_dataset():
    """Create a dataset with classes having only 1 sample - classic stratification error"""
    
    np.random.seed(42)
    
    # Create arrays of the same length
    n_samples = 12
    
    data = {
        'feature_1': list(range(1, n_samples + 1)),
        'feature_2': [i * 10 for i in range(1, n_samples + 1)],
        'feature_3': np.random.normal(0, 1, n_samples).tolist(),
        'target': (['Main_Class'] * 5 + 
                  ['Secondary'] * 3 + 
                  ['Single_1', 'Single_2', 'Single_3', 'Single_4'])
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Create sample datasets directory
    os.makedirs("sample_datasets", exist_ok=True)
    
    print("📁 Creating sample problematic datasets...")
    
    # Dataset 1: Multiple issues
    df1 = create_problematic_dataset()
    df1.to_csv("sample_datasets/problematic_dataset.csv", index=False)
    print(f"✅ Created problematic_dataset.csv ({df1.shape[0]} rows, {df1.shape[1]} columns)")
    print(f"   - Missing values: {df1.isnull().sum().sum()}")
    print(f"   - Duplicates: {df1.duplicated().sum()}")
    print(f"   - Target distribution: {df1['target'].value_counts().to_dict()}")
    
    # Dataset 2: Heavy missing values
    df2 = create_missing_values_dataset()
    df2.to_csv("sample_datasets/missing_values_dataset.csv", index=False)
    print(f"\n✅ Created missing_values_dataset.csv ({df2.shape[0]} rows, {df2.shape[1]} columns)")
    print(f"   - Missing values: {df2.isnull().sum().to_dict()}")
    
    # Dataset 3: Single sample classes
    df3 = create_single_class_dataset()
    df3.to_csv("sample_datasets/single_class_dataset.csv", index=False)
    print(f"\n✅ Created single_class_dataset.csv ({df3.shape[0]} rows, {df3.shape[1]} columns)")
    print(f"   - Target distribution: {df3['target'].value_counts().to_dict()}")
    
    print(f"\n🎉 Sample datasets created in 'sample_datasets/' directory!")
    print(f"\n💡 You can upload these files in the 'Data Upload & Preparation' page to test:")
    print(f"   1. problematic_dataset.csv - Multiple issues (missing data, duplicates, class imbalance, scaling)")
    print(f"   2. missing_values_dataset.csv - Different levels of missing data")
    print(f"   3. single_class_dataset.csv - Classes with single samples (stratification error)")
