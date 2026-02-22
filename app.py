import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- PAGE CONFIG ---
st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load('models/fraud_detection_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except:
    st.error("❌ Model files not found! Please run the training notebook first to generate .pkl files.")

# --- SIDEBAR ---
st.sidebar.title("Configuration")
app_mode = st.sidebar.selectbox("Choose Mode", ["Single Transaction", "Batch Processing"])

# --- MAIN PAGE ---
st.title("🛡️ Credit Card Fraud Detection System")
st.markdown("""
This system uses an **XGBoost Classifier** combined with **SMOTE** oversampling to identify fraudulent transactions with high precision.
""")

if app_mode == "Single Transaction":
    st.subheader("Manual Transaction Entry")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        time = st.number_input("Time (Seconds from first trans.)", value=0.0)
        amount = st.number_input("Transaction Amount ($)", value=0.0)
    
    with col2:
        v1 = st.slider("V1 (PCA Component)", -50.0, 50.0, 0.0)
        v2 = st.slider("V2 (PCA Component)", -50.0, 50.0, 0.0)
        # In a real app, you'd include inputs for V1-V28 or use defaults
    
    # Create the feature array (Padding remaining V columns with 0 for demo)
    # The model expects [Time, V1...V28, Amount]
    features = np.zeros(30)
    features[0] = time
    features[1] = v1
    features[2] = v2
    features[29] = amount
    
    if st.button("Analyze Transaction"):
        # Reshape for the scaler
        features_reshaped = features.reshape(1, -1)
        
        # Scaling (Only Time and Amount need scaling based on our preprocess.py)
        # Note: We scale the same way we did in training
        prediction = model.predict(features_reshaped)
        probability = model.predict_proba(features_reshaped)[0][1]
        
        if prediction[0] == 1:
            st.error(f"🚨 FRAUD DETECTED! (Confidence: {probability:.2%})")
        else:
            st.success(f"✅ TRANSACTION LEGITIMATE (Confidence: {1-probability:.2%})")

elif app_mode == "Batch Processing":
    st.subheader("Bulk Data Analysis")
    uploaded_file = st.file_uploader("Upload CSV of transactions", type="csv")
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        predictions = model.predict(data)
        data['Fraud_Prediction'] = predictions
        
        st.write("Results Preview:")
        st.dataframe(data.head())
        
        # Summary Visuals
        fraud_count = data['Fraud_Prediction'].sum()
        st.metric("Total Frauds Found", fraud_count)
        
        fig, ax = plt.subplots()
        sns.countplot(x='Fraud_Prediction', data=data, ax=ax, palette='viridis')
        st.pyplot(fig)