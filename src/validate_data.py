import great_expectations as gx
import sys

def validate_data(filepath):
    print(f"Validating data from {filepath}...")
    
    # Try the simple V2 API which returns a PandasDataset
    try:
        df = gx.read_csv(filepath)
    except AttributeError:
        # Fallback for very new versions if read_csv is gone
        import pandas as pd
        from great_expectations.dataset import PandasDataset
        pd_df = pd.read_csv(filepath)
        df = PandasDataset(pd_df)
    
    # Define Expectations
    print("Defining expectations...")
    
    # 1. Critical columns must exist
    critical_columns = ['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Gender', 'Married']
    for col in critical_columns:
        df.expect_column_to_exist(col)
        
    # 2. Income must be non-negative
    df.expect_column_values_to_be_between(column='ApplicantIncome', min_value=0)
    
    # 3. Loan Amount must be non-negative (if present)
    df.expect_column_values_to_be_between(column='LoanAmount', min_value=0)
    
    # 4. Credit History should be 0 or 1 (or null)
    df.expect_column_values_to_be_in_set(column='Credit_History', value_set=[0, 1, 0.0, 1.0])
    
    # 5. Gender should be Male or Female (or null)
    df.expect_column_values_to_be_in_set(column='Gender', value_set=['Male', 'Female'])
    
    # 6. Loan_ID should be unique and not null
    df.expect_column_values_to_be_unique(column='Loan_ID')
    df.expect_column_values_to_not_be_null(column='Loan_ID')

    # Validate
    print("Running validation...")
    results = df.validate()
    
    if results["success"]:
        print("✅ Data Validation Passed!")
        return True
    else:
        print("❌ Data Validation Failed!")
        return False

if __name__ == "__main__":
    # Validate Training Data
    success = validate_data('data/loan-train.csv')
    if not success:
        sys.exit(1)
