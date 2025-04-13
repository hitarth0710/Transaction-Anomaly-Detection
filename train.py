from src import data_processing, modeling

# Load and prepare data
data = data_processing.load_data('dataset/transaction_anomalies_dataset.csv')
data = data_processing.clean_data(data)
data = data_processing.calculate_statistical_anomalies(data)

# Train model
X, y = modeling.prepare_features(data)
model = modeling.train_isolation_forest(X)

# Save model
modeling.save_model(model, 'models/isolation_forest_model.pkl')