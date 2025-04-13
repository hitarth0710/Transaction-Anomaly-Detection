# Transaction Anomaly Detection System

A machine learning application that detects anomalous financial transactions using the Isolation Forest algorithm with an intuitive graphical user interface.

## Features

- **Real-time anomaly detection** on transaction data
- **User-friendly GUI** for easy interaction
- **Interactive visualization** of transaction analysis
- **Pre-configured samples** to test normal and anomalous transactions
- **Visual feedback** with color-coded results

## Installation

### Prerequisites

- Python 3.8+
- Required packages:

```bash
pip install -r requirements.txt
git clone https://github.com/yourusername/ML-Project.git
cd ML-Project
pip install -r requirements.txt
python src/generate_sample_data.py
python detection.py
```

## Project Structure
```code
ML-Project/
├── data/                      # Dataset storage
│   └── transaction_anomalies_dataset.csv
├── models/                    # Saved models
│   └── isolation_forest_model.pkl
├── notebooks/                 # Jupyter notebooks
│   └── transaction.ipynb
├── src/                       # Source code
│   ├── __init__.py
│   ├── data_processing.py
│   ├── modeling.py
│   └── visualization.py
├── docs/                      # Documentation
│   └── project_report.md
├── detection.py               # GUI application
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Screenshots

Application Interface
![image](https://github.com/user-attachments/assets/4c1befcc-2d16-4438-8be3-6542f4fcb691)


Normal Transaction Analysis
![image](https://github.com/user-attachments/assets/efb6773a-5def-47e6-ae5a-5a7fab608ef8)

Anomaly Detection
![image](https://github.com/user-attachments/assets/fc3ec36a-44f5-49d1-a851-ccdb5f7d064b)
