"""
Demo: How to Use the New Data Upload Feature
"""

import pandas as pd
import numpy as np

def create_sample_datasets():
    """Create sample datasets for testing the upload feature"""
    
    print("📁 Creating Sample Datasets for Upload Testing")
    print("=" * 50)
    
    # Dataset 1: Customer Churn Dataset
    np.random.seed(42)
    n_customers = 500
    
    customer_data = {
        'customer_id': range(1, n_customers + 1),
        'age': np.random.randint(18, 80, n_customers),
        'monthly_charges': np.random.uniform(20, 120, n_customers),
        'total_charges': np.random.uniform(200, 8000, n_customers),
        'contract_length': np.random.choice(['Month-to-month', '1 year', '2 year'], n_customers),
        'internet_service': np.random.choice(['DSL', 'Fiber', 'No'], n_customers),
        'payment_method': np.random.choice(['Credit card', 'Bank transfer', 'Electronic check', 'Mailed check'], n_customers),
        'senior_citizen': np.random.choice([0, 1], n_customers, p=[0.8, 0.2]),
        'partner': np.random.choice(['Yes', 'No'], n_customers),
        'dependents': np.random.choice(['Yes', 'No'], n_customers),
        'churn': np.random.choice(['Yes', 'No'], n_customers, p=[0.3, 0.7])  # Target
    }
    
    # Add some missing values to make it realistic
    customer_data['monthly_charges'][10:15] = np.nan
    customer_data['contract_length'][50:52] = None
    
    customer_df = pd.DataFrame(customer_data)
    customer_df.to_csv('sample_customer_churn.csv', index=False)
    print(f"✅ Created customer_churn.csv: {customer_df.shape[0]} rows, {customer_df.shape[1]} columns")
    
    # Dataset 2: Product Quality Dataset
    n_products = 300
    
    product_data = {
        'product_id': range(1, n_products + 1),
        'weight': np.random.uniform(0.5, 5.0, n_products),
        'length': np.random.uniform(10, 100, n_products),
        'width': np.random.uniform(5, 50, n_products),
        'height': np.random.uniform(2, 20, n_products),
        'material': np.random.choice(['Plastic', 'Metal', 'Glass', 'Wood'], n_products),
        'manufacturer': np.random.choice(['CompanyA', 'CompanyB', 'CompanyC'], n_products),
        'batch_number': np.random.randint(1000, 9999, n_products),
        'temperature_test': np.random.uniform(-10, 50, n_products),
        'pressure_test': np.random.uniform(0, 100, n_products),
        'quality_grade': np.random.choice(['A', 'B', 'C', 'D'], n_products, p=[0.4, 0.3, 0.2, 0.1])  # Target
    }
    
    product_df = pd.DataFrame(product_data)
    product_df.to_csv('sample_product_quality.csv', index=False)
    print(f"✅ Created product_quality.csv: {product_df.shape[0]} rows, {product_df.shape[1]} columns")
    
    # Dataset 3: Employee Performance Dataset
    n_employees = 400
    
    employee_data = {
        'employee_id': range(1, n_employees + 1),
        'age': np.random.randint(22, 65, n_employees),
        'years_experience': np.random.randint(0, 30, n_employees),
        'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_employees),
        'department': np.random.choice(['Sales', 'Marketing', 'Engineering', 'HR', 'Finance'], n_employees),
        'salary': np.random.uniform(30000, 150000, n_employees),
        'hours_per_week': np.random.uniform(35, 55, n_employees),
        'projects_completed': np.random.randint(0, 50, n_employees),
        'training_hours': np.random.randint(0, 200, n_employees),
        'remote_work': np.random.choice(['Yes', 'No'], n_employees),
        'performance_rating': np.random.choice(['Excellent', 'Good', 'Satisfactory', 'Needs Improvement'], 
                                            n_employees, p=[0.25, 0.35, 0.3, 0.1])  # Target
    }
    
    employee_df = pd.DataFrame(employee_data)
    employee_df.to_csv('sample_employee_performance.csv', index=False)
    print(f"✅ Created employee_performance.csv: {employee_df.shape[0]} rows, {employee_df.shape[1]} columns")
    
    print("\n🎯 How to Use These Datasets:")
    print("1. Run the Streamlit app: python run_app.py")
    print("2. Navigate to 'Data Upload' page")
    print("3. Upload one of the created CSV files")
    print("4. Explore the data with pagination")
    print("5. Review the data quality report")
    print("6. Select target column and prepare for ML")
    print("7. Switch to 'Model Analysis' page to use your uploaded dataset")
    
    print("\n💡 Target Columns for Each Dataset:")
    print("- customer_churn.csv: 'churn' (binary classification)")
    print("- product_quality.csv: 'quality_grade' (multi-class classification)")
    print("- employee_performance.csv: 'performance_rating' (multi-class classification)")

if __name__ == "__main__":
    create_sample_datasets()
