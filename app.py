import streamlit as st
import pandas as pd
import sys
import os

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.predict import predict_loan_approval

# Page Config
st.set_page_config(page_title="Home Loan Approval", page_icon="🏠", layout="wide")

# Title
st.title("🏠 Home Loan Approval Prediction")
st.markdown("Enter applicant details to check loan eligibility.")

# Sidebar - RAG Chatbot
st.sidebar.title("🤖 Loan Assistant")
st.sidebar.info("Ask questions about our loan policies!")

# Simple RAG Implementation (Keyword Search)
def simple_rag(query):
    try:
        with open('data/loan_policy.txt', 'r') as f:
            policy_text = f.read()
            
        # Split into sections
        sections = policy_text.split('\n\n')
        
        # Find most relevant section
        best_section = ""
        max_overlap = 0
        
        query_words = set(query.lower().split())
        
        for section in sections:
            section_words = set(section.lower().split())
            overlap = len(query_words.intersection(section_words))
            if overlap > max_overlap:
                max_overlap = overlap
                best_section = section
                
        if max_overlap > 0:
            return best_section
        else:
            return "I couldn't find specific information about that in our policy. Please contact support."
            
    except FileNotFoundError:
        return "Policy document not found."

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.sidebar.chat_message(message["role"]):
        st.sidebar.markdown(message["content"])

if prompt := st.sidebar.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.sidebar.chat_message("user"):
        st.sidebar.markdown(prompt)

    response = simple_rag(prompt)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.sidebar.chat_message("assistant"):
        st.sidebar.markdown(response)

# Main Form
col1, col2 = st.columns(2)

with col1:
    st.subheader("Applicant Details")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    
with col2:
    st.subheader("Financial Details")
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=100)
    loan_term = st.selectbox("Loan Amount Term (Months)", [360, 180, 120, 84, 60])
    credit_history = st.selectbox("Credit History", [1.0, 0.0], format_func=lambda x: "Good (1.0)" if x == 1.0 else "Bad (0.0)")
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Predict Button
if st.button("Check Eligibility", type="primary"):
    # Prepare Data
    input_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }
    
    # Call Prediction
    result = predict_loan_approval(input_data)
    
    # Display Result
    if "error" in result:
        st.error(result["error"])
    else:
        status = result["status"]
        prob = result["probability"]
        
        if status == "Approved":
            st.success(f"🎉 Congratulations! Loan Approved.")
            st.metric("Approval Probability", f"{prob:.2%}")
        else:
            st.error(f"❌ Sorry, Loan Rejected.")
            st.metric("Approval Probability", f"{prob:.2%}")
            
            # Simple Explanation
            if credit_history == 0.0:
                st.warning("Reason: Poor Credit History is a major factor.")
            elif (applicant_income + coapplicant_income) < 3000:
                st.warning("Reason: Total Income might be too low.")
