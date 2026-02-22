##  AI/ML Internship – Mini Project
## Credit Card Fraud Detection System💳

## 📌 Project Overview

Credit card fraud costs financial institutions billions of dollars annually. This project implements a machine learning-based fraud detection system to identify fraudulent transactions in real-time.

Since the dataset is highly imbalanced (fraud accounts for less than 0.2% of transactions), advanced sampling techniques and gradient-boosting algorithms were used to ensure high recall and precision.

## 🚀 Key Features

- **Imbalance Handling** 
Utilized SMOTE (Synthetic Minority Over-sampling Technique) to balance the training data.

- **Anomaly Detection**
Integrated Isolation Forest to detect outliers and uncover potential new fraud patterns.

- **High-Performance Classifier**
Built using XGBoost, tuned to prioritize high Recall (detecting fraud) while minimizing false alarms.

- **Interactive Dashboard**
Developed a full-stack Streamlit web application for:

  - Manual transaction fraud checking

  - Batch transaction processing

## 🛠️ Tech Stack

## Language:

- Python

## Libraries:

- Scikit-Learn

- XGBoost

- Pandas

- NumPy

- Imbalanced-learn

## Deployment:

- Streamlit

## 📊 Dataset Information

The project uses the Kaggle Credit Card Fraud Detection Dataset.

- PCA Transformation:
Features V1–V28 are principal components obtained using PCA for privacy protection.

- Class Imbalance:

  - Total Transactions: 284,807

  - Fraud Cases: 492

  - Fraud Percentage: ~0.17%

## ⚙️ How to Run
- Install Dependencies
pip install -r requirements.txt
- Run the Training Notebook  (Ensure creditcard.csv is placed inside the data/ folder.)

   - **Open and run:**

      - **notebooks/Fraud_Detection.ipynb**  (This will generate the trained model files inside the models/ directory.)

- Launch the Streamlit Dashboard
  
  - **streamlit run src/app.py**

## 📈 Results

- Recall: ~92%
(Successfully detects the majority of fraudulent transactions)

- ROC-AUC Score: 0.98

- Confusion Matrix:
Optimized to minimize False Negatives, ensuring stronger financial security.

## 📂 Project Structure
```text
Credit-Card-Fraud-Detection/
│
├── data/
│   └── creditcard.csv                # Kaggle dataset (not uploaded due to size)
│
├── models/
│   ├── fraud_detection_model.pkl     # Trained XGBoost model
│   └── scaler.pkl                    # StandardScaler for Time & Amount
│
├── notebooks/
│   └── Fraud_Detection.ipynb         # Data analysis & model training
│
├── src/
│   ├── app.py                        # Streamlit dashboard
│   └── preprocess.py                 # Data preprocessing logic
│
├── requirements.txt                  # Project dependencies
│
└── README.md                         # Project documentation
