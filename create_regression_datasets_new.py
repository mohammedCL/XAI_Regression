"""
Script to create 5 different regression datasets with train/test splits and trained ML models.

This script generates various types of regression datasets with different characteristics:
- Simple Linear: Low noise, linear relationships
- Complex Nonlinear: High-dimensional with nonlinear patterns  
- High Dimensional: Many features with feature selection challenge
- Noisy Data: High noise to test robustness
- Real World: California housing dataset

Each dataset is saved in its own folder with:
- train.csv and test.csv files
- Multiple trained models in joblib format
- Scaler objects for models that need scaling
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Create main datasets directory
datasets_dir = Path("datasets")
datasets_dir.mkdir(exist_ok=True)

def create_dataset_folder(dataset_name):
    """Create a folder for the dataset"""
    folder_path = datasets_dir / dataset_name
    folder_path.mkdir(exist_ok=True)
    return folder_path

def train_and_save_models(X_train, y_train, X_test, y_test, folder_path, dataset_name):
    """Train multiple regression models and save them"""
    
    print(f"  Training models for {dataset_name}...")
    
    models = {
        'linear_regression': LinearRegression(),
        'ridge_regression': Ridge(alpha=1.0, random_state=42),
        'lasso_regression': Lasso(alpha=0.1, random_state=42),
        'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        'decision_tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'random_forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
    }
    
    # Models that need scaling
    scaling_models = {
        'svr': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'linear_scaled': LinearRegression(),
    }
    
    model_results = {}
    
    # Train models without scaling
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            model_results[name] = {
                'model': model,
                'r2': r2,
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse)
            }
            
            # Save model
            joblib.dump(model, folder_path / f"{name}.joblib")
            print(f"    ✓ {name}: R²={r2:.4f}, RMSE={np.sqrt(mse):.4f}")
            
        except Exception as e:
            print(f"    ❌ {name}: Failed - {str(e)}")
    
    # Train models with scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, folder_path / "scaler.joblib")
    
    for name, model in scaling_models.items():
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            model_results[name] = {
                'model': model,
                'r2': r2,
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse),
                'scaled': True
            }
            
            # Save model
            joblib.dump(model, folder_path / f"{name}.joblib")
            print(f"    ✓ {name} (scaled): R²={r2:.4f}, RMSE={np.sqrt(mse):.4f}")
            
        except Exception as e:
            print(f"    ❌ {name}: Failed - {str(e)}")
    
    return model_results

def save_train_test_data(X_train, y_train, X_test, y_test, feature_names, folder_path):
    """Save train and test datasets"""
    
    # Create train dataframe
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['target'] = y_train
    
    # Create test dataframe  
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['target'] = y_test
    
    # Save to CSV
    train_df.to_csv(folder_path / "train.csv", index=False)
    test_df.to_csv(folder_path / "test.csv", index=False)
    
    print(f"    ✓ Saved train.csv ({len(train_df)} samples)")
    print(f"    ✓ Saved test.csv ({len(test_df)} samples)")
    
    return train_df, test_df

def create_simple_linear_dataset():
    """Dataset 1: Simple Linear with low noise"""
    print("\n📊 Creating Simple Linear Dataset...")
    
    folder_path = create_dataset_folder("simple_linear")
    
    # Generate simple linear data
    X, y = make_regression(
        n_samples=1000,
        n_features=5,
        n_informative=4,
        noise=10,
        random_state=42
    )
    
    feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save datasets
    save_train_test_data(X_train, y_train, X_test, y_test, feature_names, folder_path)
    
    # Train and save models
    model_results = train_and_save_models(X_train, y_train, X_test, y_test, folder_path, "Simple Linear")
    
    return model_results

def create_complex_nonlinear_dataset():
    """Dataset 2: Complex Nonlinear with interactions"""
    print("\n📊 Creating Complex Nonlinear Dataset...")
    
    folder_path = create_dataset_folder("complex_nonlinear")
    
    # Generate base data
    X, y = make_regression(
        n_samples=1500,
        n_features=8,
        n_informative=6,
        noise=0.1,
        random_state=123
    )
    
    # Add nonlinear transformations
    X_nonlinear = X.copy()
    
    # Add polynomial features for first 3 features
    X_nonlinear[:, 0] = X[:, 0] ** 2
    X_nonlinear[:, 1] = np.sin(X[:, 1])
    X_nonlinear[:, 2] = np.log(np.abs(X[:, 2]) + 1)
    
    # Add interactions
    interaction_term = X[:, 0] * X[:, 1] * 0.5
    y = y + interaction_term + np.random.normal(0, 20, len(y))
    
    feature_names = [f'feature_{i+1}' for i in range(X_nonlinear.shape[1])]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_nonlinear, y, test_size=0.2, random_state=42
    )
    
    # Save datasets
    save_train_test_data(X_train, y_train, X_test, y_test, feature_names, folder_path)
    
    # Train and save models
    model_results = train_and_save_models(X_train, y_train, X_test, y_test, folder_path, "Complex Nonlinear")
    
    return model_results

def create_high_dimensional_dataset():
    """Dataset 3: High dimensional with many irrelevant features"""
    print("\n📊 Creating High Dimensional Dataset...")
    
    folder_path = create_dataset_folder("high_dimensional")
    
    # Generate high dimensional data with many irrelevant features
    X, y = make_regression(
        n_samples=800,
        n_features=25,
        n_informative=8,
        n_redundant=5,
        noise=15,
        random_state=456
    )
    
    feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save datasets
    save_train_test_data(X_train, y_train, X_test, y_test, feature_names, folder_path)
    
    # Train and save models
    model_results = train_and_save_models(X_train, y_train, X_test, y_test, folder_path, "High Dimensional")
    
    return model_results

def create_noisy_dataset():
    """Dataset 4: Noisy data to test robustness"""
    print("\n📊 Creating Noisy Dataset...")
    
    folder_path = create_dataset_folder("noisy_data")
    
    # Generate data with high noise
    X, y = make_regression(
        n_samples=1200,
        n_features=6,
        n_informative=4,
        noise=50,  # High noise
        random_state=789
    )
    
    # Add outliers
    n_outliers = int(0.05 * len(y))  # 5% outliers
    outlier_indices = np.random.choice(len(y), n_outliers, replace=False)
    y[outlier_indices] += np.random.normal(0, 200, n_outliers)
    
    feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save datasets
    save_train_test_data(X_train, y_train, X_test, y_test, feature_names, folder_path)
    
    # Train and save models
    model_results = train_and_save_models(X_train, y_train, X_test, y_test, folder_path, "Noisy Data")
    
    return model_results

def create_california_housing_dataset():
    """Dataset 5: Real-world California housing data"""
    print("\n📊 Creating California Housing Dataset...")
    
    folder_path = create_dataset_folder("california_housing")
    
    # Load California housing dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    feature_names = housing.feature_names.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save datasets
    save_train_test_data(X_train, y_train, X_test, y_test, feature_names, folder_path)
    
    # Train and save models
    model_results = train_and_save_models(X_train, y_train, X_test, y_test, folder_path, "California Housing")
    
    return model_results

def create_summary_report(all_results):
    """Create a summary report of all datasets and models"""
    
    print("\n📋 Creating Summary Report...")
    
    report_lines = [
        "REGRESSION DATASETS AND MODELS SUMMARY REPORT",
        "=" * 60,
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "DATASETS CREATED:",
        "1. Simple Linear - 1000 samples, 5 features, low noise",
        "2. Complex Nonlinear - 1500 samples, 8 features, nonlinear relationships", 
        "3. High Dimensional - 800 samples, 25 features, many irrelevant",
        "4. Noisy Data - 1200 samples, 6 features, high noise + outliers",
        "5. California Housing - 20640 samples, 8 features, real-world data",
        "",
        "MODELS TRAINED FOR EACH DATASET:",
        "- Linear Regression",
        "- Ridge Regression", 
        "- Lasso Regression",
        "- Elastic Net",
        "- Decision Tree",
        "- Random Forest",
        "- Gradient Boosting",
        "- SVR (with scaling)",
        "- Linear Regression (with scaling)",
        "",
        "PERFORMANCE SUMMARY:",
        "-" * 40
    ]
    
    for dataset_name, results in all_results.items():
        report_lines.append(f"\n{dataset_name.upper()}:")
        
        # Sort models by R² score
        sorted_models = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
        
        for model_name, metrics in sorted_models:
            scaled_info = " (scaled)" if metrics.get('scaled', False) else ""
            report_lines.append(
                f"  {model_name}{scaled_info:15} - R²: {metrics['r2']:.4f}, "
                f"RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}"
            )
    
    report_lines.extend([
        "",
        "FILES CREATED PER DATASET:",
        "- train.csv (training data)",
        "- test.csv (test data)", 
        "- *.joblib (trained models)",
        "- scaler.joblib (feature scaler for scaled models)",
        "",
        "USAGE:",
        "Load models with: model = joblib.load('path/to/model.joblib')",
        "Load scaler with: scaler = joblib.load('path/to/scaler.joblib')",
        "Load data with: df = pd.read_csv('path/to/train.csv')",
    ])
    
    # Save report
    report_path = datasets_dir / "SUMMARY_REPORT.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"    ✓ Summary report saved: {report_path}")
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("QUICK SUMMARY:")
    print("=" * 60)
    for dataset_name, results in all_results.items():
        best_model = max(results.items(), key=lambda x: x[1]['r2'])
        print(f"{dataset_name:20} - Best: {best_model[0]} (R²: {best_model[1]['r2']:.4f})")

def main():
    """Main function to create all datasets and models"""
    
    print("🚀 REGRESSION DATASETS AND MODELS GENERATOR")
    print("=" * 60)
    print("Creating 5 different regression datasets with train/test splits")
    print("and multiple trained ML models for each dataset...")
    
    all_results = {}
    
    # Create all datasets
    all_results['Simple Linear'] = create_simple_linear_dataset()
    all_results['Complex Nonlinear'] = create_complex_nonlinear_dataset() 
    all_results['High Dimensional'] = create_high_dimensional_dataset()
    all_results['Noisy Data'] = create_noisy_dataset()
    all_results['California Housing'] = create_california_housing_dataset()
    
    # Create summary report
    create_summary_report(all_results)
    
    print(f"\n✅ ALL COMPLETE!")
    print(f"📁 All datasets saved in: {datasets_dir.absolute()}")
    print(f"📋 Check SUMMARY_REPORT.txt for detailed results")
    print(f"\n📂 Directory structure:")
    print(f"{datasets_dir}/")
    for folder in sorted(datasets_dir.iterdir()):
        if folder.is_dir():
            print(f"├── {folder.name}/")
            for file in sorted(folder.iterdir()):
                print(f"│   ├── {file.name}")

if __name__ == "__main__":
    main()
