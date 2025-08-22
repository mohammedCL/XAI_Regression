"""
Easy Model Switcher for ExplainableAI Regression Testing
Copy model files to the standard names used by the backend API
"""

import shutil
from pathlib import Path
import sys

def list_available_datasets():
    """List all available datasets"""
    storage_dir = Path("backend/storage")
    datasets = set()
    
    for file in storage_dir.glob("*_regression.csv"):
        dataset_name = file.stem.replace("_regression", "")
        datasets.add(dataset_name)
    
    return sorted(list(datasets))

def list_available_models(dataset_name):
    """List all available models for a dataset"""
    storage_dir = Path("backend/storage")
    models = []
    
    for file in storage_dir.glob(f"{dataset_name}_*.joblib"):
        if "scaler" not in file.name:
            model_name = file.stem.replace(f"{dataset_name}_", "")
            models.append(model_name)
    
    return sorted(models)

def switch_model(dataset_name, model_name):
    """Switch to a specific dataset and model"""
    storage_dir = Path("backend/storage")
    
    # Check if dataset exists
    dataset_file = storage_dir / f"{dataset_name}_regression.csv"
    if not dataset_file.exists():
        print(f"❌ Dataset '{dataset_name}' not found!")
        return False
    
    # Check if model exists
    model_file = storage_dir / f"{dataset_name}_{model_name}.joblib"
    if not model_file.exists():
        print(f"❌ Model '{model_name}' not found for dataset '{dataset_name}'!")
        return False
    
    # Copy dataset to standard name
    target_dataset = storage_dir / "california_housing_regression.csv"
    shutil.copy2(dataset_file, target_dataset)
    print(f"✅ Copied {dataset_file.name} → {target_dataset.name}")
    
    # Copy model to standard name
    target_model = storage_dir / "california_housing_regressor.joblib"
    shutil.copy2(model_file, target_model)
    print(f"✅ Copied {model_file.name} → {target_model.name}")
    
    # Copy scaler if it exists and model needs it
    scaler_file = storage_dir / f"{dataset_name}_scaler.joblib"
    if scaler_file.exists():
        target_scaler = storage_dir / "california_housing_scaler.joblib"
        shutil.copy2(scaler_file, target_scaler)
        print(f"✅ Copied {scaler_file.name} → {target_scaler.name}")
    
    print(f"\n🎯 Successfully switched to:")
    print(f"   Dataset: {dataset_name}")
    print(f"   Model: {model_name}")
    print(f"   Backend will now use this configuration!")
    
    return True

def show_model_info(dataset_name, model_name):
    """Show information about a specific model"""
    # Read from the summary report
    report_file = Path("backend/storage/model_summary_report.txt")
    if not report_file.exists():
        print("❌ Summary report not found!")
        return
    
    with open(report_file, 'r') as f:
        content = f.read()
    
    # Find the dataset section
    dataset_section = f"Dataset: {dataset_name}"
    if dataset_section not in content:
        print(f"❌ Dataset '{dataset_name}' not found in report!")
        return
    
    # Find the model section
    model_section = f"  {model_name}:"
    if model_section not in content:
        print(f"❌ Model '{model_name}' not found for dataset '{dataset_name}'!")
        return
    
    # Extract and display the model info
    lines = content.split('\n')
    dataset_idx = next(i for i, line in enumerate(lines) if dataset_section in line)
    model_idx = next(i for i, line in enumerate(lines[dataset_idx:]) if model_section in line) + dataset_idx
    
    print(f"\n📊 {dataset_name} - {model_name}")
    print("=" * 40)
    
    # Show model metrics
    for i in range(model_idx, len(lines)):
        line = lines[i].strip()
        if line.startswith("R² Score:") or line.startswith("RMSE:") or line.startswith("MAE:"):
            print(f"  {line}")
        elif line == "" or line.startswith("  ") and not line.startswith("    "):
            break

def main():
    if len(sys.argv) < 2:
        print("🚀 ExplainableAI Model Switcher")
        print("=" * 40)
        print("\nUsage:")
        print("  python switch_model.py list")
        print("  python switch_model.py info <dataset> <model>")
        print("  python switch_model.py switch <dataset> <model>")
        print("\nExamples:")
        print("  python switch_model.py list")
        print("  python switch_model.py info sales_prediction random_forest")
        print("  python switch_model.py switch energy_consumption gradient_boosting")
        return
    
    command = sys.argv[1]
    
    if command == "list":
        print("📋 Available Datasets and Models:")
        print("=" * 40)
        
        datasets = list_available_datasets()
        for dataset in datasets:
            models = list_available_models(dataset)
            print(f"\n🗂️  {dataset}:")
            for model in models:
                print(f"     • {model}")
    
    elif command == "info":
        if len(sys.argv) < 4:
            print("❌ Usage: python switch_model.py info <dataset> <model>")
            return
        
        dataset_name = sys.argv[2]
        model_name = sys.argv[3]
        show_model_info(dataset_name, model_name)
    
    elif command == "switch":
        if len(sys.argv) < 4:
            print("❌ Usage: python switch_model.py switch <dataset> <model>")
            return
        
        dataset_name = sys.argv[2]
        model_name = sys.argv[3]
        
        if switch_model(dataset_name, model_name):
            print(f"\n🔄 You can now test the API with:")
            print(f"   • Dataset: {dataset_name}")
            print(f"   • Model: {model_name}")
            print(f"\n💡 The backend API will automatically use the new model!")
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: list, info, switch")

if __name__ == "__main__":
    main()
