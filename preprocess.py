import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

class FraudPreprocessor:
    def __init__(self, scaler_path='models/scaler.pkl'):
        self.scaler_path = scaler_path
        self.scaler = StandardScaler()

    def clean_data(self, df):
        """Removes duplicates and handles missing values."""
        df = df.drop_duplicates()
        df = df.fillna(method='ffill')
        return df

    def scale_features(self, df, is_training=True):
        """
        Scales 'Time' and 'Amount'. 
        V1-V28 are usually already scaled via PCA in this dataset.
        """
        # Ensure we are only scaling the columns that need it
        target_cols = ['Time', 'Amount']
        
        if is_training:
            df[target_cols] = self.scaler.fit_transform(df[target_cols])
            # Save the scaler to use later in the Streamlit app
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.scaler, self.scaler_path)
        else:
            # Load the saved scaler for production inference
            self.scaler = joblib.load(self.scaler_path)
            df[target_cols] = self.scaler.transform(df[target_cols])
            
        return df

    def prepare_features(self, df):
        """Separates features from the target variable."""
        if 'Class' in df.columns:
            X = df.drop(['Class'], axis=1)
            y = df['Class']
            return X, y
        return df

def get_processed_data(file_path):
    """Orchestrator function to run the full pipeline."""
    preprocessor = FraudPreprocessor()
    
    # 1. Load
    df = pd.read_csv(file_path)
    
    # 2. Clean
    df = preprocessor.clean_data(df)
    
    # 3. Scale
    df = preprocessor.scale_features(df, is_training=True)
    
    # 4. Split
    X, y = preprocessor.prepare_features(df)
    
    return X, y

if __name__ == "__main__":
    # Test the preprocessor
    print("Running preprocessing pipeline...")
    try:
        X, y = get_processed_data('creditcard.csv')
        print(f"Success! Features shape: {X.shape}, Target shape: {y.shape}")
    except Exception as e:
        print(f"Error: {e}. Please ensure 'creditcard.csv' exists in the working directory.")