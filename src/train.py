import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from src.preprocessing import get_pipeline

def train_model():
    # Load Data
    print("Loading data...")
    try:
        train_df = pd.read_csv('data/loan-train.csv')
    except FileNotFoundError:
        # Fallback if running from src directory
        train_df = pd.read_csv('../data/loan-train.csv')
    
    # Separate Target and Features
    X = train_df.drop(['Loan_Status'], axis=1)
    y = train_df['Loan_Status'].map({'Y': 1, 'N': 0})
    
    # Split Data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Get Pipeline
    print("Building pipeline...")
    pipeline_steps = get_pipeline()
    
    # Add Model to Pipeline
    # We append the model as the final step
    full_pipeline = Pipeline(steps=[
        ('preprocessor', pipeline_steps),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # Train
    print("Training model...")
    full_pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = full_pipeline.predict(X_val)
    y_prob = full_pipeline.predict_proba(X_val)[:, 1]
    
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_val, y_prob):.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_val, y_pred))
    
    # Save Model
    print("Saving model to src/model.pkl...")
    joblib.dump(full_pipeline, 'src/model.pkl')
    print("Done!")

if __name__ == "__main__":
    train_model()
