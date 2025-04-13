import pandas as pd
import os

def load_data(data_path):
    """
    Load transaction dataset from the specified path
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded transaction data
    """
    return pd.read_csv(data_path)

def clean_data(df):
    """
    Clean and preprocess the transaction data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw transaction data
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned transaction data
    """
    # Check for missing values
    print("Missing values in each column:")
    print(df.isnull().sum())
    
    # Handle missing values if any
    # df = df.fillna(...) - add specific logic if needed
    
    return df

def calculate_statistical_anomalies(df, col='Transaction_Amount', threshold=2):
    """
    Calculate statistical anomalies based on standard deviations
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Transaction data
    col : str
        Column name to analyze for anomalies
    threshold : float
        Number of standard deviations to consider as anomaly threshold
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with anomaly flag
    """
    # Calculate mean and standard deviation
    mean_value = df[col].mean()
    std_value = df[col].std()
    
    # Define the anomaly threshold
    anomaly_threshold = mean_value + threshold * std_value
    
    # Flag anomalies
    df['Is_Anomaly'] = df[col] > anomaly_threshold
    
    # Calculate anomaly ratio
    anomaly_ratio = df['Is_Anomaly'].sum() / df.shape[0]
    print(f"Anomaly ratio: {anomaly_ratio:.4f}")
    
    return df