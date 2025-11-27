import joblib
import pandas as pd
import numpy as np

# Load the model once when the script is imported
try:
    model = joblib.load('src/model.pkl')
except FileNotFoundError:
    # Fallback if running from root directory
    try:
        model = joblib.load('model.pkl')
    except:
        model = None

def predict_loan_approval(data):
    """
    Predicts loan approval status for a single applicant or a batch.
    
    Args:
        data (dict or pd.DataFrame): Applicant details.
        
    Returns:
        dict: Prediction result ('Approved'/'Rejected') and probability.
    """
    if model is None:
        return {"error": "Model not found. Please train the model first."}
    
    # Convert dict to DataFrame if necessary
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data
        
    # Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    status = "Approved" if prediction == 1 else "Rejected"
    
    return {
        "status": status,
        "probability": float(probability)
    }

if __name__ == "__main__":
    # Test Prediction
    sample_data = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '0',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 5000,
        'CoapplicantIncome': 0,
        'LoanAmount': 150,
        'Loan_Amount_Term': 360,
        'Credit_History': 1.0,
        'Property_Area': 'Urban'
    }
    
    print("Test Prediction:")
    print(predict_loan_approval(sample_data))
