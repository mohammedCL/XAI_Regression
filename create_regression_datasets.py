"""
Script to create multiple regression datasets and trained models for testing
ExplainableAI regression functionality.

This script generates various types of regression datasets with different characteristics:
- Different numbers of features
- Different sample sizes
- Different noise levels
- Different complexity levels
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, fetch_california_housing, load_diabetes
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from pathlib import Path

# Create storage directory if it doesn't exist
storage_dir = Path("backend/storage")
storage_dir.mkdir(exist_ok=True)

def create_synthetic_dataset(name, n_samples=1000, n_features=10, noise=0.1, random_state=42):
    """Create a synthetic regression dataset"""
    print(f"Creating {name} dataset...")
    
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(1, n_features // 2),
        noise=noise,
        random_state=random_state
    )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df, feature_names

def create_real_world_datasets():
    """Create datasets based on real-world scenarios"""
    datasets = {}
    
    # 1. California Housing (already exists, but let's recreate for consistency)
    print("Creating California Housing dataset...")
    california = fetch_california_housing()
    df_california = pd.DataFrame(california.data, columns=california.feature_names)
    df_california['target'] = california.target
    datasets['california_housing'] = (df_california, california.feature_names)
    
    # 2. Diabetes dataset
    print("Creating Diabetes dataset...")
    diabetes = load_diabetes()
    df_diabetes = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df_diabetes['target'] = diabetes.target
    datasets['diabetes'] = (df_diabetes, diabetes.feature_names)
    
    # 3. Stock Price Simulation
    print("Creating Stock Price simulation dataset...")
    np.random.seed(42)
    n_samples = 2000
    
    # Economic indicators
    gdp_growth = np.random.normal(2.5, 1.5, n_samples)
    inflation_rate = np.random.normal(2.0, 0.8, n_samples)
    unemployment_rate = np.random.normal(5.0, 2.0, n_samples)
    interest_rate = np.random.normal(3.0, 1.2, n_samples)
    
    # Market indicators
    market_volatility = np.random.exponential(0.2, n_samples)
    trading_volume = np.random.lognormal(10, 1, n_samples)
    
    # Company metrics
    revenue_growth = np.random.normal(8.0, 5.0, n_samples)
    profit_margin = np.random.normal(15.0, 8.0, n_samples)
    debt_ratio = np.random.uniform(0.1, 0.8, n_samples)
    
    # Stock price based on these factors
    stock_price = (
        50 + 
        gdp_growth * 2.5 +
        inflation_rate * (-1.2) +
        unemployment_rate * (-0.8) +
        interest_rate * (-1.5) +
        market_volatility * (-10) +
        np.log(trading_volume) * 0.5 +
        revenue_growth * 0.3 +
        profit_margin * 0.2 +
        debt_ratio * (-15) +
        np.random.normal(0, 5, n_samples)  # Random noise
    )
    
    df_stock = pd.DataFrame({
        'gdp_growth': gdp_growth,
        'inflation_rate': inflation_rate,
        'unemployment_rate': unemployment_rate,
        'interest_rate': interest_rate,
        'market_volatility': market_volatility,
        'trading_volume': trading_volume,
        'revenue_growth': revenue_growth,
        'profit_margin': profit_margin,
        'debt_ratio': debt_ratio,
        'target': stock_price
    })
    
    stock_features = ['gdp_growth', 'inflation_rate', 'unemployment_rate', 'interest_rate',
                     'market_volatility', 'trading_volume', 'revenue_growth', 'profit_margin', 'debt_ratio']
    datasets['stock_price'] = (df_stock, stock_features)
    
    # 4. Energy Consumption Prediction
    print("Creating Energy Consumption dataset...")
    np.random.seed(123)
    n_samples = 1500
    
    # Weather factors
    temperature = np.random.normal(20, 10, n_samples)  # Celsius
    humidity = np.random.uniform(30, 90, n_samples)    # Percentage
    wind_speed = np.random.exponential(5, n_samples)   # km/h
    solar_radiation = np.random.gamma(2, 3, n_samples) # kWh/m²
    
    # Building characteristics
    building_size = np.random.uniform(100, 1000, n_samples)  # m²
    insulation_rating = np.random.uniform(1, 10, n_samples)  # 1-10 scale
    occupancy = np.random.poisson(20, n_samples)             # Number of people
    
    # Temporal factors
    hour_of_day = np.random.randint(0, 24, n_samples)
    day_of_week = np.random.randint(0, 7, n_samples)
    
    # Energy consumption based on these factors
    energy_consumption = (
        10 +  # Base consumption
        building_size * 0.05 +
        occupancy * 2 +
        np.where(temperature < 15, (15 - temperature) * 3, 0) +  # Heating
        np.where(temperature > 25, (temperature - 25) * 2, 0) +  # Cooling
        humidity * 0.1 +
        wind_speed * (-0.5) +
        solar_radiation * (-0.3) +
        insulation_rating * (-2) +
        np.where((hour_of_day >= 8) & (hour_of_day <= 18), 15, 5) +  # Business hours
        np.where(day_of_week < 5, 10, -5) +  # Weekday vs weekend
        np.random.normal(0, 5, n_samples)  # Random noise
    )
    
    df_energy = pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'solar_radiation': solar_radiation,
        'building_size': building_size,
        'insulation_rating': insulation_rating,
        'occupancy': occupancy,
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'target': energy_consumption
    })
    
    energy_features = ['temperature', 'humidity', 'wind_speed', 'solar_radiation',
                      'building_size', 'insulation_rating', 'occupancy', 'hour_of_day', 'day_of_week']
    datasets['energy_consumption'] = (df_energy, energy_features)
    
    # 5. Sales Prediction
    print("Creating Sales Prediction dataset...")
    np.random.seed(456)
    n_samples = 1200
    
    # Marketing factors
    advertising_spend = np.random.exponential(1000, n_samples)
    social_media_reach = np.random.lognormal(8, 1.5, n_samples)
    email_campaign_size = np.random.uniform(1000, 50000, n_samples)
    
    # Product factors
    price = np.random.uniform(10, 200, n_samples)
    discount_percent = np.random.uniform(0, 50, n_samples)
    product_rating = np.random.normal(4.0, 0.8, n_samples)
    
    # Market factors
    competitor_price = price * np.random.uniform(0.8, 1.2, n_samples)
    seasonal_factor = np.sin(np.random.uniform(0, 2*np.pi, n_samples)) * 20
    economic_index = np.random.normal(100, 15, n_samples)
    
    # Sales based on these factors
    sales = (
        100 +  # Base sales
        np.log(advertising_spend + 1) * 10 +
        np.log(social_media_reach + 1) * 5 +
        np.log(email_campaign_size + 1) * 2 +
        price * (-0.5) +
        discount_percent * 2 +
        product_rating * 30 +
        (price - competitor_price) * (-3) +
        seasonal_factor +
        economic_index * 0.3 +
        np.random.normal(0, 20, n_samples)  # Random noise
    )
    
    df_sales = pd.DataFrame({
        'advertising_spend': advertising_spend,
        'social_media_reach': social_media_reach,
        'email_campaign_size': email_campaign_size,
        'price': price,
        'discount_percent': discount_percent,
        'product_rating': product_rating,
        'competitor_price': competitor_price,
        'seasonal_factor': seasonal_factor,
        'economic_index': economic_index,
        'target': sales
    })
    
    sales_features = ['advertising_spend', 'social_media_reach', 'email_campaign_size',
                     'price', 'discount_percent', 'product_rating', 'competitor_price',
                     'seasonal_factor', 'economic_index']
    datasets['sales_prediction'] = (df_sales, sales_features)
    
    return datasets

def train_multiple_models(X_train, X_test, y_train, y_test, dataset_name):
    """Train multiple models on the same dataset"""
    models = {
        'linear_regression': LinearRegression(),
        'ridge_regression': Ridge(alpha=1.0),
        'lasso_regression': Lasso(alpha=1.0),
        'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'decision_tree': DecisionTreeRegressor(random_state=42),
        'svr': SVR(kernel='rbf', C=1.0)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"  Training {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Save model
        model_filename = f"{dataset_name}_{model_name}.joblib"
        joblib.dump(model, storage_dir / model_filename)
        
        results[model_name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'model_file': model_filename
        }
        
        print(f"    R² Score: {r2:.4f}, RMSE: {rmse:.4f}")
    
    return results

def main():
    """Main function to create all datasets and models"""
    print("🚀 Creating regression datasets and models for ExplainableAI testing...\n")
    
    all_results = {}
    
    # Create synthetic datasets
    synthetic_datasets = {
        'simple_linear': create_synthetic_dataset('Simple Linear', n_samples=500, n_features=3, noise=0.1),
        'complex_nonlinear': create_synthetic_dataset('Complex Nonlinear', n_samples=2000, n_features=15, noise=0.3),
        'high_dimensional': create_synthetic_dataset('High Dimensional', n_samples=1000, n_features=25, noise=0.2),
        'noisy_data': create_synthetic_dataset('Noisy Data', n_samples=800, n_features=8, noise=0.5)
    }
    
    # Create real-world datasets
    real_world_datasets = create_real_world_datasets()
    
    # Combine all datasets
    all_datasets = {**synthetic_datasets, **real_world_datasets}
    
    # Process each dataset
    for dataset_name, (df, feature_names) in all_datasets.items():
        print(f"\n📊 Processing {dataset_name} dataset...")
        print(f"   Shape: {df.shape}")
        print(f"   Features: {len(feature_names)}")
        
        # Save dataset
        dataset_filename = f"{dataset_name}_regression.csv"
        df.to_csv(storage_dir / dataset_filename, index=False)
        print(f"   Saved: {dataset_filename}")
        
        # Prepare data for training
        X = df[feature_names]
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features for some models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler
        scaler_filename = f"{dataset_name}_scaler.joblib"
        joblib.dump(scaler, storage_dir / scaler_filename)
        
        # Train models with original features
        print(f"   Training models...")
        model_results = train_multiple_models(X_train, X_test, y_train, y_test, dataset_name)
        
        # Train additional models with scaled features
        scaled_models = {
            'svr_scaled': SVR(kernel='rbf', C=1.0),
            'linear_scaled': LinearRegression()
        }
        
        for model_name, model in scaled_models.items():
            print(f"  Training {model_name}...")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            model_filename = f"{dataset_name}_{model_name}.joblib"
            joblib.dump(model, storage_dir / model_filename)
            
            model_results[model_name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'model_file': model_filename,
                'uses_scaler': True,
                'scaler_file': scaler_filename
            }
            
            print(f"    R² Score: {r2:.4f}, RMSE: {rmse:.4f}")
        
        all_results[dataset_name] = {
            'dataset_file': dataset_filename,
            'scaler_file': scaler_filename,
            'features': feature_names,
            'shape': df.shape,
            'models': model_results
        }
    
    # Create summary report
    create_summary_report(all_results)
    
    print(f"\n✅ Complete! Created {len(all_datasets)} datasets with multiple models each.")
    print(f"📁 All files saved to: {storage_dir}")
    print(f"📋 Check 'model_summary_report.txt' for detailed results.")

def create_summary_report(results):
    """Create a summary report of all models and their performance"""
    report_path = storage_dir / "model_summary_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("ExplainableAI Regression Models Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        for dataset_name, dataset_info in results.items():
            f.write(f"Dataset: {dataset_name}\n")
            f.write("-" * 30 + "\n")
            f.write(f"File: {dataset_info['dataset_file']}\n")
            f.write(f"Shape: {dataset_info['shape']}\n")
            f.write(f"Features: {len(dataset_info['features'])}\n")
            f.write(f"Feature names: {', '.join(dataset_info['features'])}\n\n")
            
            f.write("Model Performance:\n")
            for model_name, metrics in dataset_info['models'].items():
                f.write(f"  {model_name}:\n")
                f.write(f"    R² Score: {metrics['r2']:.4f}\n")
                f.write(f"    RMSE: {metrics['rmse']:.4f}\n")
                f.write(f"    MAE: {metrics['mae']:.4f}\n")
                f.write(f"    Model file: {metrics['model_file']}\n")
                if metrics.get('uses_scaler'):
                    f.write(f"    Scaler file: {metrics['scaler_file']}\n")
                f.write("\n")
            
            f.write("\n" + "="*50 + "\n\n")
    
    print(f"📋 Summary report saved: {report_path}")

def create_test_script():
    """Create a script to easily test different models"""
    test_script = """
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
"""
    
    with open(storage_dir / "test_models.py", 'w') as f:
        f.write(test_script)
    
    print(f"🧪 Test script created: {storage_dir / 'test_models.py'}")

if __name__ == "__main__":
    main()
    create_test_script()
