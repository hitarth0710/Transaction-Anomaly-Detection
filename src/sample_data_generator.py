import pandas as pd
import numpy as np
import os

def generate_transaction_data(n_samples=1000, anomaly_ratio=0.02, seed=42):
    """
    Generate synthetic transaction data with anomalies
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    anomaly_ratio : float
        Ratio of anomalies to include
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        Generated transaction data
    """
    np.random.seed(seed)
    
    # Calculate number of normal and anomaly samples
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies
    
    # Generate normal transactions
    normal_data = {
        'Transaction_ID': np.arange(1, n_normal + 1),
        'Customer_ID': np.random.randint(1000, 9999, n_normal),
        'Transaction_Amount': np.random.normal(100, 30, n_normal),
        'Average_Transaction_Amount': np.random.normal(100, 20, n_normal),
        'Frequency_of_Transactions': np.random.normal(5, 2, n_normal),
        'Age': np.random.randint(18, 75, n_normal),
        'Account_Type': np.random.choice(['Savings', 'Checking', 'Credit'], n_normal),
        'Day_of_Week': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], n_normal),
        'Is_Anomaly': False
    }
    
    # Generate anomaly transactions
    anomaly_data = {
        'Transaction_ID': np.arange(n_normal + 1, n_samples + 1),
        'Customer_ID': np.random.randint(1000, 9999, n_anomalies),
        'Transaction_Amount': np.random.normal(300, 50, n_anomalies),  # Higher amounts
        'Average_Transaction_Amount': np.random.normal(100, 20, n_anomalies),  # Similar to normal
        'Frequency_of_Transactions': np.random.normal(15, 5, n_anomalies),  # Higher frequency
        'Age': np.random.randint(18, 75, n_anomalies),
        'Account_Type': np.random.choice(['Savings', 'Checking', 'Credit'], n_anomalies),
        'Day_of_Week': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], n_anomalies),
        'Is_Anomaly': True
    }
    
    # Combine normal and anomaly data
    df_normal = pd.DataFrame(normal_data)
    df_anomaly = pd.DataFrame(anomaly_data)
    df = pd.concat([df_normal, df_anomaly], ignore_index=True)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # Generate sample data
    df = generate_transaction_data(n_samples=1000, anomaly_ratio=0.02)
    
    # Create data directory if it doesn't exist
    os.makedirs('../data', exist_ok=True)
    
    # Save to CSV
    output_path = '../data/transaction_anomalies_dataset.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} transactions ({df['Is_Anomaly'].sum()} anomalies) and saved to {output_path}")