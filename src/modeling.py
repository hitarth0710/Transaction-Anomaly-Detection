import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

def prepare_features(df, feature_cols=None):
    """
    Prepare feature set for modeling
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Transaction data
    feature_cols : list
        List of column names to use as features
        
    Returns:
    --------
    X : pandas.DataFrame
        Features for model training
    y : pandas.Series
        Target variable (anomaly flags)
    """
    if feature_cols is None:
        feature_cols = ['Transaction_Amount',
                       'Average_Transaction_Amount',
                       'Frequency_of_Transactions']
    
    X = df[feature_cols]
    y = df['Is_Anomaly']
    
    return X, y

def train_isolation_forest(X_train, contamination=0.02, random_state=42):
    """
    Train Isolation Forest model
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    contamination : float
        Expected proportion of anomalies
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    sklearn.ensemble.IsolationForest
        Trained model
    """
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(X_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Parameters:
    -----------
    model : sklearn.ensemble.IsolationForest
        Trained model
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        True labels
        
    Returns:
    --------
    str
        Classification report
    """
    # Predict anomalies on the test set
    y_pred = model.predict(X_test)
    
    # Convert predictions to binary values (0: normal, 1: anomaly)
    y_pred_binary = [1 if pred == -1 else 0 for pred in y_pred]
    
    # Generate classification report
    report = classification_report(y_test, y_pred_binary, target_names=['Normal', 'Anomaly'])
    return report

def save_model(model, model_path):
    """
    Save trained model to disk
    
    Parameters:
    -----------
    model : sklearn.ensemble.IsolationForest
        Trained model
    model_path : str
        Path to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def load_model(model_path):
    """
    Load trained model from disk
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
        
    Returns:
    --------
    sklearn.ensemble.IsolationForest
        Loaded model
    """
    model = joblib.load(model_path)
    return model

def predict_anomaly(model, inputs, feature_cols=None):
    """
    Predict whether a transaction is an anomaly
    
    Parameters:
    -----------
    model : sklearn.ensemble.IsolationForest
        Trained model
    inputs : list or dict
        Input feature values
    feature_cols : list
        Feature column names
        
    Returns:
    --------
    bool
        True if anomaly, False if normal
    """
    if feature_cols is None:
        feature_cols = ['Transaction_Amount',
                       'Average_Transaction_Amount',
                       'Frequency_of_Transactions']
    
    if isinstance(inputs, dict):
        # Convert dict to DataFrame
        user_df = pd.DataFrame([inputs])
    else:
        # Convert list to DataFrame
        user_df = pd.DataFrame([inputs], columns=feature_cols)
    
    # Make prediction
    prediction = model.predict(user_df)
    
    # Convert to binary (True if anomaly)
    is_anomaly = prediction[0] == -1
    
    return is_anomaly