"""
Create test regression assets for testing the ExplainableAI application.
This script generates:
1. A simple regression dataset (California housing-like)
2. A trained regression model (RandomForestRegressor)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
import joblib
import os

def create_regression_dataset():
    """Create a synthetic regression dataset similar to California housing."""
    
    # Generate synthetic regression data
    X, y = make_regression(
        n_samples=1000,
        n_features=8,
        n_informative=6,
        noise=0.1,
        random_state=42
    )
    
    # Create meaningful feature names
    feature_names = [
        'median_income',
        'house_age', 
        'avg_rooms',
        'avg_bedrooms',
        'population',
        'avg_occupancy',
        'latitude',
        'longitude'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # Scale the target to represent house prices (in 100k units)
    y_scaled = (y - y.min()) / (y.max() - y.min()) * 8 + 0.5  # Range: 0.5 to 8.5 (50k to 850k)
    df['median_house_value'] = y_scaled
    
    # Add some realistic ranges to features
    df['median_income'] = np.clip(df['median_income'] * 2 + 5, 0.5, 15)  # Income in 10k units
    df['house_age'] = np.clip(df['house_age'] * 10 + 25, 1, 52)  # Age in years
    df['avg_rooms'] = np.clip(df['avg_rooms'] * 2 + 6, 2, 15)  # Average rooms
    df['avg_bedrooms'] = np.clip(df['avg_bedrooms'] * 0.5 + 1.2, 0.5, 5)  # Average bedrooms
    df['population'] = np.clip(df['population'] * 1000 + 2000, 100, 8000)  # Population
    df['avg_occupancy'] = np.clip(df['avg_occupancy'] * 1 + 3, 1, 8)  # Average occupancy
    df['latitude'] = np.clip(df['latitude'] * 5 + 35, 32, 42)  # Latitude
    df['longitude'] = np.clip(df['longitude'] * 8 - 120, -125, -114)  # Longitude
    
    return df

def create_regression_model(df):
    """Train a RandomForestRegressor on the dataset."""
    
    # Prepare features and target
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Model R² Score:")
    print(f"  Training: {train_score:.4f}")
    print(f"  Test: {test_score:.4f}")
    
    return model

def main():
    """Create and save regression test assets."""
    
    # Create output directory
    storage_dir = "backend/storage"
    os.makedirs(storage_dir, exist_ok=True)
    
    print("Creating regression dataset...")
    df = create_regression_dataset()
    
    # Save dataset
    dataset_path = os.path.join(storage_dir, "california_housing_regression.csv")
    df.to_csv(dataset_path, index=False)
    print(f"Saved dataset to: {dataset_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Target range: {df['median_house_value'].min():.2f} - {df['median_house_value'].max():.2f}")
    
    print("\nCreating regression model...")
    model = create_regression_model(df)
    
    # Save model
    model_path = os.path.join(storage_dir, "california_housing_regressor.joblib")
    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path}")
    
    print("\nRegression test assets created successfully!")
    print(f"To test: Upload {model_path} and {dataset_path} with target 'median_house_value'")

if __name__ == "__main__":
    main()
