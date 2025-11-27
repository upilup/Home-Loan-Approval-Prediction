# 🏠 Home Loan Approval Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)

## 📌 Project Overview
This project automates the home loan approval process using Machine Learning. By analyzing applicant details (income, credit history, etc.), the model predicts whether a loan should be **Approved** or **Rejected**.

The project includes a full pipeline:
- **Data Analysis**: In-depth EDA to understand trends.
- **Preprocessing**: Handling missing values and outliers.
- **Modeling**: Comparing Logistic Regression, Random Forest, and XGBoost.
- **Deployment**: An interactive web application built with Streamlit.

## 📂 Project Structure
```
├── data/               # Dataset files
├── src/                # Source code for preprocessing and training
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── app.py              # Streamlit Web Application
├── loan_approval.ipynb # Jupyter Notebook for experimentation
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## 🚀 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Home Loan Approval Prediction"
   ```

2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Run the Web App
Launch the interactive dashboard to test predictions in real-time:
```bash
streamlit run app.py
```

### Run the Pipeline
You can also run the modular scripts directly:
```bash
python src/train.py   # Train the model
python src/predict.py # Generate predictions
```

### Explore the Notebook
Open `loan_approval.ipynb` to see the step-by-step analysis and model experiments.

## 📊 Model Performance
After experimenting with multiple models, **Logistic Regression** was selected as the final model due to its balance of accuracy and interpretability.

## 🔮 Future Improvements
- Integrate a **RAG-based Chatbot** for loan policy queries.
- Deploy the application to the cloud (e.g., Streamlit Cloud, AWS).
- Improve model accuracy with advanced ensemble techniques.

## 👤 Author
[Your Name]