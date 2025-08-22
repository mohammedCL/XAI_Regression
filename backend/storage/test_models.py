
# Quick test script for ExplainableAI regression models
# Run this in the backend directory

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Available datasets and models
datasets = [
    'simple_linear', 'complex_nonlinear', 'high_dimensional', 'noisy_data',
    'california_housing', 'diabetes', 'stock_price', 'energy_consumption', 'sales_prediction'
]

models = [
    'linear_regression', 'ridge_regression', 'lasso_regression', 'elastic_net',
    'random_forest', 'gradient_boosting', 'decision_tree', 'svr'
]

def test_model(dataset_name, model_name):
    storage_dir = Path("storage")
    
    # Load dataset
    df = pd.read_csv(storage_dir / f"{dataset_name}_regression.csv")
    
    # Load model
    model = joblib.load(storage_dir / f"{dataset_name}_{model_name}.joblib")
    
    # Get features (all columns except 'target')
    features = [col for col in df.columns if col != 'target']
    X = df[features].iloc[:5]  # Test with first 5 rows
    
    # Make predictions
    predictions = model.predict(X)
    
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Sample predictions: {predictions}")
    print("-" * 40)

# Example usage:
if __name__ == "__main__":
    # Test a few combinations
    test_model('california_housing', 'random_forest')
    test_model('sales_prediction', 'gradient_boosting')
    test_model('energy_consumption', 'linear_regression')
