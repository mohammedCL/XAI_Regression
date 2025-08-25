#!/usr/bin/env python3
"""
Script to create separate train/test datasets and train a Linear Regression model
for the California Housing dataset.

This script will:
1. Load the California housing dataset
2. Split it into training and testing sets (80/20 split)
3. Train a Linear Regression model on the training data
4. Save the model as a .joblib file
5. Save the training and testing datasets as separate CSV files
6. Display model performance metrics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from datetime import datetime

def create_train_test_and_model():
    """Main function to create datasets and train model."""
    
    print("🏠 California Housing Dataset - Train/Test Split & Model Training")
    print("=" * 70)
    
    # Define paths
    data_path = "backend/storage/storage/california_housing_regression.csv"
    output_dir = "backend/storage"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. Load the dataset
        print("📊 Loading dataset...")
        df = pd.read_csv(data_path)
        print(f"✅ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Display basic info about the dataset
        print(f"\n📈 Dataset Info:")
        print(f"   • Features: {list(df.columns[:-1])}")
        print(f"   • Target: {df.columns[-1]}")
        print(f"   • Target range: {df['target'].min():.2f} - {df['target'].max():.2f}")
        print(f"   • Missing values: {df.isnull().sum().sum()}")
        
        # 2. Prepare features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        print(f"\n🔄 Splitting dataset...")
        print(f"   • Features shape: {X.shape}")
        print(f"   • Target shape: {y.shape}")
        
        # 3. Split the data (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42,
            shuffle=True
        )
        
        print(f"✅ Data split completed:")
        print(f"   • Training set: {X_train.shape[0]} samples")
        print(f"   • Testing set: {X_test.shape[0]} samples")
        
        # 4. Create and train the Linear Regression model
        print(f"\n🤖 Training Linear Regression model...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        print("✅ Model training completed!")
        
        # 5. Evaluate the model
        print(f"\n📊 Model Performance Evaluation:")
        
        # Training predictions
        y_train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Testing predictions
        y_test_pred = model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print(f"   📈 Training Metrics:")
        print(f"      • R² Score: {train_r2:.4f}")
        print(f"      • RMSE: {train_rmse:.4f}")
        print(f"      • MAE: {train_mae:.4f}")
        print(f"      • MSE: {train_mse:.4f}")
        
        print(f"   📉 Testing Metrics:")
        print(f"      • R² Score: {test_r2:.4f}")
        print(f"      • RMSE: {test_rmse:.4f}")
        print(f"      • MAE: {test_mae:.4f}")
        print(f"      • MSE: {test_mse:.4f}")
        
        # Check for overfitting
        r2_diff = train_r2 - test_r2
        if r2_diff > 0.1:
            print(f"   ⚠️  Warning: Possible overfitting (R² difference: {r2_diff:.4f})")
        else:
            print(f"   ✅ Good generalization (R² difference: {r2_diff:.4f})")
        
        # 6. Save the trained model
        model_path = os.path.join(output_dir, "california_housing_linear_regression.joblib")
        print(f"\n💾 Saving model...")
        joblib.dump(model, model_path)
        print(f"✅ Model saved to: {model_path}")
        
        # 7. Save training dataset
        train_data = pd.concat([X_train, y_train], axis=1)
        train_path = os.path.join(output_dir, "california_housing_train.csv")
        train_data.to_csv(train_path, index=False)
        print(f"✅ Training dataset saved to: {train_path}")
        print(f"   • Shape: {train_data.shape}")
        
        # 8. Save testing dataset
        test_data = pd.concat([X_test, y_test], axis=1)
        test_path = os.path.join(output_dir, "california_housing_test.csv")
        test_data.to_csv(test_path, index=False)
        print(f"✅ Testing dataset saved to: {test_path}")
        print(f"   • Shape: {test_data.shape}")
        
        # 9. Create a summary report
        report_path = os.path.join(output_dir, "model_training_report.txt")
        with open(report_path, 'w') as f:
            f.write("California Housing Linear Regression Model Training Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Dataset Information:\n")
            f.write(f"• Total samples: {df.shape[0]}\n")
            f.write(f"• Features: {df.shape[1] - 1}\n")
            f.write(f"• Feature names: {', '.join(X.columns)}\n")
            f.write(f"• Target variable: target\n")
            f.write(f"• Target range: {y.min():.2f} - {y.max():.2f}\n\n")
            
            f.write("Data Split:\n")
            f.write(f"• Training samples: {X_train.shape[0]} ({X_train.shape[0]/df.shape[0]*100:.1f}%)\n")
            f.write(f"• Testing samples: {X_test.shape[0]} ({X_test.shape[0]/df.shape[0]*100:.1f}%)\n\n")
            
            f.write("Model Performance:\n")
            f.write(f"• Training R²: {train_r2:.4f}\n")
            f.write(f"• Testing R²: {test_r2:.4f}\n")
            f.write(f"• Training RMSE: {train_rmse:.4f}\n")
            f.write(f"• Testing RMSE: {test_rmse:.4f}\n")
            f.write(f"• Generalization gap: {r2_diff:.4f}\n\n")
            
            f.write("Feature Coefficients:\n")
            for feature, coef in zip(X.columns, model.coef_):
                f.write(f"• {feature}: {coef:.6f}\n")
            f.write(f"• Intercept: {model.intercept_:.6f}\n\n")
            
            f.write("Files Generated:\n")
            f.write(f"• Model: {model_path}\n")
            f.write(f"• Training data: {train_path}\n")
            f.write(f"• Testing data: {test_path}\n")
        
        print(f"✅ Training report saved to: {report_path}")
        
        # 10. Display feature importance (coefficients)
        print(f"\n🎯 Feature Importance (Linear Regression Coefficients):")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_,
            'Abs_Coefficient': np.abs(model.coef_)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        for _, row in feature_importance.iterrows():
            print(f"   • {row['Feature']:<12}: {row['Coefficient']:>8.4f}")
        print(f"   • {'Intercept':<12}: {model.intercept_:>8.4f}")
        
        print(f"\n🎉 All files have been created successfully!")
        print(f"📂 Output directory: {os.path.abspath(output_dir)}")
        
        return {
            'model': model,
            'train_data': train_data,
            'test_data': test_data,
            'metrics': {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse
            }
        }
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def verify_files():
    """Verify that all generated files exist and are valid."""
    print(f"\n🔍 Verifying generated files...")
    
    files_to_check = [
        "backend/storage/california_housing_linear_regression.joblib",
        "backend/storage/california_housing_train.csv",
        "backend/storage/california_housing_test.csv",
        "backend/storage/model_training_report.txt"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} - NOT FOUND")
    
    # Test loading the model
    try:
        model_path = "backend/storage/california_housing_linear_regression.joblib"
        if os.path.exists(model_path):
            loaded_model = joblib.load(model_path)
            print(f"✅ Model loads successfully: {type(loaded_model).__name__}")
            
            # Test prediction with sample data
            if hasattr(loaded_model, 'coef_'):
                print(f"✅ Model has coefficients: {len(loaded_model.coef_)} features")
        else:
            print(f"❌ Cannot test model loading - file not found")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting California Housing dataset preparation and model training...\n")
    
    # Create the model and datasets
    result = create_train_test_and_model()
    
    if result is not None:
        # Verify all files were created correctly
        verify_files()
        print(f"\n✨ Script completed successfully!")
    else:
        print(f"\n💥 Script failed. Please check the error messages above.")
