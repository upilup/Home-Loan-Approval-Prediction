import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Create Total_Income
        # Handle cases where columns might be missing if running on subset
        if 'ApplicantIncome' in X.columns and 'CoapplicantIncome' in X.columns:
            X['Total_Income'] = X['ApplicantIncome'] + X['CoapplicantIncome']
            
        # Map Credit_History to Categorical (Y/N)
        if 'Credit_History' in X.columns:
            X['Credit_History'] = X['Credit_History'].map({1.0: 'Y', 0.0: 'N', 1: 'Y', 0: 'N'})

        
        # Log Transformations (using log1p for safety)
        if 'LoanAmount' in X.columns:
            X['LoanAmount_Log'] = np.log1p(X['LoanAmount'])
        
        if 'Total_Income' in X.columns:
            X['Total_Income_Log'] = np.log1p(X['Total_Income'])
            
        # Drop original columns
        cols_to_drop = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Total_Income', 'Loan_ID']
        X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])
        
        return X

def get_pipeline():
    # Define features that will exist AFTER FeatureEngineering
    # Note: We need to know which columns are numerical/categorical after engineering
    # This is a bit tricky in a pure pipeline without data, but we can define the selectors
    
    # We will assume the standard set of columns after engineering:
    # Num: LoanAmount_Log, Total_Income_Log, Loan_Amount_Term, Credit_History
    # Cat: Gender, Married, Dependents, Education, Self_Employed, Property_Area
    
    # Since ColumnTransformer requires knowing column names or indices, 
    # and our FeatureEngineer changes the columns, we usually put FeatureEngineer OUTSIDE 
    # the ColumnTransformer, or use a pipeline where the first step is FeatureEngineer.
    
    # However, ColumnTransformer needs to know which columns to apply to.
    # A robust way is to use `make_column_selector` but that applies to the output of the previous step.
    # Let's define the pipeline structure:
    
    # Step 1: Feature Engineering (Pandas in, Pandas out)
    # Step 2: Column Transformer (Pandas in, Array out)
    
    from sklearn.compose import make_column_selector
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, make_column_selector(dtype_include=np.number)),
            ('cat', categorical_transformer, make_column_selector(dtype_include=object))
        ],
        verbose_feature_names_out=False
    )
    
    full_pipeline = Pipeline(steps=[
        ('feature_engineering', FeatureEngineer()),
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline
