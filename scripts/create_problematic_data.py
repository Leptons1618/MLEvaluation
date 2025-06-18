"""
Create sample datasets that demonstrate data preparation issues including stratification errors
"""

import pandas as pd
import numpy as np
import os

def create_stratification_error_dataset():
    """Create a dataset that would cause stratification errors"""
    np.random.seed(42)
    
    # Create dataset with some classes having very few samples
    data = {
        'customer_age': np.random.randint(18, 80, 100),
        'income': np.random.uniform(20000, 100000, 100),
        'spending_score': np.random.randint(1, 100, 100),
        'membership_years': np.random.randint(1, 20, 100),
        'purchase_frequency': np.random.uniform(1, 50, 100),
        # Problematic target with single-sample classes
        'customer_segment': (['Regular'] * 60 + ['Premium'] * 25 + ['VIP'] * 10 + 
                           ['Elite'] * 3 + ['Diamond'] * 1 + ['Platinum'] * 1)
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    df.loc[5:10, 'income'] = np.nan
    df.loc[20:22, 'spending_score'] = np.nan
    
    # Add duplicates
    df = pd.concat([df, df.iloc[0:3]], ignore_index=True)
    
    return df

def create_data_quality_issues_dataset():
    """Create a dataset with various data quality issues"""
    np.random.seed(123)
    
    # Create a messy dataset
    data = {
        'product_id': range(1, 301),
        'price': np.random.uniform(10, 1000, 300),
        'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Sports', 'Home'] + 
                                   [f'Category_{i}' for i in range(20)], 300),  # High cardinality
        'rating': np.random.uniform(1, 5, 300),
        'review_count': np.random.randint(0, 1000, 300),
        'availability': np.random.choice(['In Stock', 'Out of Stock', 'Limited'], 300),
        'brand': np.random.choice(['BrandA', 'BrandB', 'BrandC', 'BrandD'], 300),
        'weight_kg': np.random.uniform(0.1, 50, 300),
        'quality_score': np.random.choice(['Excellent', 'Good', 'Fair', 'Poor', 'Defective'], 
                                        300, p=[0.4, 0.3, 0.2, 0.08, 0.02])  # Imbalanced target
    }
    
    df = pd.DataFrame(data)
    
    # Add significant missing values
    df.loc[50:100, 'rating'] = np.nan  # 50 missing values
    df.loc[150:200, 'brand'] = None    # 50 missing values
    df.loc[250:270, 'weight_kg'] = np.nan  # 20 missing values
    
    # Add duplicates
    df = pd.concat([df, df.iloc[0:10]], ignore_index=True)
    
    # Add some extreme outliers in price (different scale issue)
    df.loc[0:5, 'price'] = df.loc[0:5, 'price'] * 1000  # Make some prices extremely high
    
    return df

def create_imbalanced_dataset():
    """Create a heavily imbalanced dataset"""
    np.random.seed(456)
    
    # Medical diagnosis dataset with extreme class imbalance
    data = {
        'patient_age': np.random.randint(20, 90, 1000),
        'blood_pressure_systolic': np.random.randint(90, 180, 1000),
        'blood_pressure_diastolic': np.random.randint(60, 120, 1000),
        'cholesterol': np.random.randint(150, 300, 1000),
        'bmi': np.random.uniform(18, 40, 1000),
        'smoking': np.random.choice(['Yes', 'No'], 1000, p=[0.3, 0.7]),
        'exercise_hours_week': np.random.uniform(0, 15, 1000),
        'family_history': np.random.choice(['Yes', 'No'], 1000, p=[0.4, 0.6]),
        # Extremely imbalanced diagnosis
        'diagnosis': (['Healthy'] * 950 + ['At Risk'] * 35 + ['Disease'] * 10 + 
                     ['Critical'] * 3 + ['Severe'] * 2)
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    df.loc[100:110, 'cholesterol'] = np.nan
    df.loc[200:205, 'bmi'] = np.nan
    
    return df

def save_sample_datasets():
    """Save all sample datasets to files"""
    print("📁 Creating Sample Datasets with Data Preparation Issues")
    print("=" * 60)
    
    # Dataset 1: Stratification Error Dataset
    df1 = create_stratification_error_dataset()
    filename1 = 'sample_stratification_error.csv'
    df1.to_csv(filename1, index=False)
    
    print(f"✅ Created {filename1}")
    print(f"   📊 Shape: {df1.shape}")
    print(f"   🎯 Target distribution: {df1['customer_segment'].value_counts().to_dict()}")
    print(f"   ❌ Missing values: {df1.isnull().sum().sum()}")
    print(f"   🔄 Duplicates: {df1.duplicated().sum()}")
    print(f"   ⚠️  Issues: Classes with 1 sample (will cause stratification error)")
    
    # Dataset 2: Data Quality Issues Dataset
    df2 = create_data_quality_issues_dataset()
    filename2 = 'sample_data_quality_issues.csv'
    df2.to_csv(filename2, index=False)
    
    print(f"\n✅ Created {filename2}")
    print(f"   📊 Shape: {df2.shape}")
    print(f"   🎯 Target distribution: {df2['quality_score'].value_counts().to_dict()}")
    print(f"   ❌ Missing values: {df2.isnull().sum().sum()}")
    print(f"   🔄 Duplicates: {df2.duplicated().sum()}")
    print(f"   ⚠️  Issues: High missing values, duplicates, scale differences")
    
    # Dataset 3: Imbalanced Dataset
    df3 = create_imbalanced_dataset()
    filename3 = 'sample_imbalanced_medical.csv'
    df3.to_csv(filename3, index=False)
    
    print(f"\n✅ Created {filename3}")
    print(f"   📊 Shape: {df3.shape}")
    print(f"   🎯 Target distribution: {df3['diagnosis'].value_counts().to_dict()}")
    print(f"   ❌ Missing values: {df3.isnull().sum().sum()}")
    print(f"   ⚠️  Issues: Extreme class imbalance (95% one class)")
    
    print(f"\n🎯 How to Use These Datasets:")
    print("1. Run the Streamlit app: python run_app.py")
    print("2. Navigate to 'Data Upload & Prep' page")
    print("3. Upload one of the created CSV files:")
    print(f"   • {filename1} - Test stratification error handling")
    print(f"   • {filename2} - Test comprehensive data quality fixes")
    print(f"   • {filename3} - Test class imbalance detection")
    print("4. Click 'Analyze Preparation Needs' to see detected issues")
    print("5. Try different auto-fix options:")
    print("   • 'Auto-Fix Critical Issues' for essential fixes")
    print("   • 'Auto-Fix All Recommended' for comprehensive preparation")
    print("   • Custom fix selection for specific issues")
    
    print(f"\n💡 Expected Behavior:")
    print("• The stratification error should be automatically handled")
    print("• Missing values should be detected and fixable")
    print("• Class imbalance should be identified with recommendations")
    print("• Duplicates should be detected and removable")
    print("• All datasets should prepare successfully after fixes")

if __name__ == "__main__":
    save_sample_datasets()
