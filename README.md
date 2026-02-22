# AI_ML Internship - Mini Project
# Credit Card Fraud Detection System
📌 Project Overview
Credit card fraud costs financial institutions billions of dollars annually. This project implements a machine learning solution to identify fraudulent transactions in real-time. Given the highly imbalanced nature of the dataset (where fraud accounts for less than 0.2% of transactions), I utilized advanced sampling techniques and gradient-boosting algorithms to ensure high recall and precision.

🚀 Key Features
Imbalance Handling: Utilized SMOTE (Synthetic Minority Over-sampling Technique) to balance the training data.

Anomaly Detection: Integrated Isolation Forest to identify outliers and potential new fraud patterns.

High-Performance Classifier: Developed using XGBoost, tuned to prioritize detecting fraud (Recall) while minimizing false alarms.

Interactive Dashboard: A full-stack Streamlit web application for manual transaction checking and batch processing.

## 🛠️ Tech Stack
Language: Python

Libraries: Scikit-Learn, XGBoost, Pandas, NumPy, Imbalanced-learn

Deployment: Streamlit

## 📊 Dataset Information
The project uses the Kaggle Credit Card Fraud Detection dataset.

PCA Transformation: Features V1-V28 are principal components obtained with PCA for privacy reasons.

Class Imbalance: 492 frauds out of 284,807 transactions.

## ⚙️ How to Run

Install Dependencies:

Bash
pip install -r requirements.txt
Run the Training Notebook:
Ensure the creditcard.csv is in the data/ folder and run the Fraud_Detection.ipynb to generate the model files.

Launch the Dashboard:

Bash
streamlit run src/app.py
📈 Results
Recall: ~92% (Successfully identifying the vast majority of fraud).

ROC-AUC Score: 0.98.

Confusion Matrix: Minimizes False Negatives to ensure financial security.

## Project Structure:
```text
Credit-Card-Fraud-Detection/
├── data/
│   └── creditcard.csv          # The Kaggle dataset (not uploaded to GitHub due to size)
├── models/
│   ├── fraud_detection_model.pkl # The trained XGBoost classifier
│   └── scaler.pkl                # The StandardScaler for Time and Amount
├── notebooks/
│   └── Fraud_Detection.ipynb    # Full data analysis and model training
├── src/
│   ├── app.py                   # Streamlit web application dashboard
│   └── preprocess.py            # Reusable data cleaning and scaling logic
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
