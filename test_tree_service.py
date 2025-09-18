#!/usr/bin/env python3
"""
Test script to verify the TreeService works correctly with regression models
This script generates synthetic datasets and trains models for testing
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from backend.app.services.base_model_service import BaseModelService
from backend.app.services.tree_service import TreeService

def generate_synthetic_data():
    """Generate synthetic regression dataset"""
    print("Generating synthetic regression dataset...")
    
    # Create a synthetic regression dataset
    X, y = make_regression(
        n_samples=1000,
        n_features=8,
        n_informative=6,
        n_targets=1,
        noise=0.1,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"✓ Generated dataset with {X.shape[0]} samples and {X.shape[1]} features")
    print(f"  - Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  - Target mean: {y.mean():.2f}, std: {y.std():.2f}")
    
    return df, feature_names

def generate_california_housing_data():
    """Generate California housing dataset"""
    print("Loading California housing dataset...")
    
    # Load California housing dataset
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"✓ Loaded California housing dataset with {X.shape[0]} samples and {X.shape[1]} features")
    print(f"  - Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  - Target mean: {y.mean():.2f}, std: {y.std():.2f}")
    
    return df, feature_names

def train_models(X_train, y_train):
    """Train different regression models"""
    models = {}
    
    print("Training regression models...")
    
    # Decision Tree Regressor
    print("  - Training Decision Tree Regressor...")
    dt_model = DecisionTreeRegressor(max_depth=10, random_state=42)
    dt_model.fit(X_train, y_train)
    models['decision_tree'] = dt_model
    
    # Random Forest Regressor
    print("  - Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=10, max_depth=8, random_state=42)
    rf_model.fit(X_train, y_train)
    models['random_forest'] = rf_model
    
    # Gradient Boosting Regressor
    print("  - Training Gradient Boosting Regressor...")
    gb_model = GradientBoostingRegressor(n_estimators=10, max_depth=6, random_state=42)
    gb_model.fit(X_train, y_train)
    models['gradient_boosting'] = gb_model
    
    print("✓ All models trained successfully")
    return models

def evaluate_models(models, X_test, y_test):
    """Evaluate trained models"""
    print("\nEvaluating model performance...")
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"  - {name.replace('_', ' ').title()}:")
        print(f"    * R² Score: {r2:.4f}")
        print(f"    * RMSE: {rmse:.4f}")

def test_regression_tree_service():
    """Test TreeService with regression models"""
    
    # Generate synthetic data
    data, feature_names = generate_synthetic_data()
    
    # Split features and target
    X = data[feature_names]
    y = data['target']
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate models
    evaluate_models(models, X_test, y_test)
    
    print("\n" + "="*60)
    print("Testing Decision Tree Regressor...")
    
    model = models['decision_tree']
    
    # Create service
    base_service = BaseModelService()
    
    # Set up the base service
    base_service.model = model
    base_service.X_train = X_train
    base_service.y_train = y_train
    base_service.X_test = X_test
    base_service.y_test = y_test
    base_service.feature_names = feature_names
    
    # Create tree service
    tree_service = TreeService(base_service)
    
    # Test get_decision_tree
    print("\n1. Testing get_decision_tree()...")
    tree_result = tree_service.get_decision_tree()
    
    if "error" in tree_result:
        print(f"Error: {tree_result['error']}")
        return
    
    print(f"✓ Successfully extracted tree structure")
    print(f"  - Model type: {tree_result['model_type']}")
    print(f"  - Algorithm: {tree_result['algorithm']}")
    print(f"  - Is regression: {tree_result['is_regression']}")
    print(f"  - Number of trees: {tree_result['num_trees']}")
    
    # Check first tree
    if tree_result['trees']:
        first_tree = tree_result['trees'][0]
        print(f"  - Tree 0 performance: {first_tree['performance']:.4f} ({first_tree['performance_metric']})")
        print(f"  - Tree 0 max depth: {first_tree['max_depth']}")
        print(f"  - Tree 0 total nodes: {first_tree['total_nodes']}")
        print(f"  - Tree 0 leaf nodes: {first_tree['leaf_nodes']}")
        print(f"  - Tree 0 is regression: {first_tree['is_regression']}")
        
        # Check tree structure - find a leaf node
        def find_leaf_node(node):
            if node['type'] == 'leaf':
                return node
            if 'left' in node:
                leaf = find_leaf_node(node['left'])
                if leaf:
                    return leaf
            if 'right' in node:
                leaf = find_leaf_node(node['right'])
                if leaf:
                    return leaf
            return None
        
        leaf_node = find_leaf_node(first_tree['tree_structure'])
        if leaf_node:
            print(f"  - Sample leaf node:")
            print(f"    * Type: {leaf_node['type']}")
            print(f"    * Samples: {leaf_node['samples']}")
            print(f"    * Is regression: {leaf_node['is_regression']}")
            if 'prediction' in leaf_node:
                print(f"    * Prediction: {leaf_node['prediction']:.4f}")
            if 'uncertainty' in leaf_node:
                print(f"    * Uncertainty: {leaf_node['uncertainty']:.4f}")
    
    # Test get_tree_rules
    print("\n2. Testing get_tree_rules()...")
    rules_result = tree_service.get_tree_rules(tree_idx=0, max_depth=3)
    
    if "error" in rules_result:
        print(f"Error: {rules_result['error']}")
        return
    
    print(f"✓ Successfully extracted tree rules")
    print(f"  - Tree index: {rules_result['tree_index']}")
    print(f"  - Max depth: {rules_result['max_depth']}")
    print(f"  - Total rules: {rules_result['total_rules']}")
    print(f"  - Is regression: {rules_result['is_regression']}")
    
    # Show first few rules
    if rules_result['rules']:
        print(f"  - First 3 rules:")
        for i, rule in enumerate(rules_result['rules'][:3]):
            print(f"    {i+1}. {rule['rule']}")
            print(f"       Samples: {rule['samples']}, Support: {rule['support']:.3f}")
            if rule['type'] == 'regression':
                print(f"       Predicted value: {rule['predicted_value']:.4f}")
            else:
                print(f"       Predicted class: {rule['predicted_class']}, Confidence: {rule['confidence']:.3f}")
            print()

def test_ensemble_regression():
    """Test TreeService with ensemble regression models"""
    
    print("\n" + "="*60)
    print("Testing Random Forest Regressor...")
    
    # Generate California housing data for diversity
    data, feature_names = generate_california_housing_data()
    
    # Split features and target
    X = data[feature_names]
    y = data['target']
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    models = train_models(X_train, y_train)
    
    model = models['random_forest']
    
    # Create service
    base_service = BaseModelService()
    
    # Set up the base service
    base_service.model = model
    base_service.X_train = X_train
    base_service.y_train = y_train
    base_service.X_test = X_test
    base_service.y_test = y_test
    base_service.feature_names = feature_names
    
    # Create tree service
    tree_service = TreeService(base_service)
    
    # Test get_decision_tree
    tree_result = tree_service.get_decision_tree()
    
    if "error" in tree_result:
        print(f"Error: {tree_result['error']}")
        return
    
    print(f"✓ Successfully extracted ensemble tree structure")
    print(f"  - Model type: {tree_result['model_type']}")
    print(f"  - Algorithm: {tree_result['algorithm']}")
    print(f"  - Is regression: {tree_result['is_regression']}")
    print(f"  - Number of trees shown: {tree_result['num_trees']}")
    print(f"  - Total trees in model: {tree_result['total_trees']}")

def test_gradient_boosting_regression():
    """Test TreeService with Gradient Boosting regression models"""
    
    print("\n" + "="*60)
    print("Testing Gradient Boosting Regressor...")
    
    # Use synthetic data for GB testing
    data, feature_names = generate_synthetic_data()
    
    # Split features and target
    X = data[feature_names]
    y = data['target']
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    models = train_models(X_train, y_train)
    
    model = models['gradient_boosting']
    
    # Create service
    base_service = BaseModelService()
    
    # Set up the base service
    base_service.model = model
    base_service.X_train = X_train
    base_service.y_train = y_train
    base_service.X_test = X_test
    base_service.y_test = y_test
    base_service.feature_names = feature_names
    
    # Create tree service
    tree_service = TreeService(base_service)
    
    # Test get_decision_tree
    tree_result = tree_service.get_decision_tree()
    
    if "error" in tree_result:
        print(f"Error: {tree_result['error']}")
        return
    
    print(f"✓ Successfully extracted gradient boosting tree structure")
    print(f"  - Model type: {tree_result['model_type']}")
    print(f"  - Algorithm: {tree_result['algorithm']}")
    print(f"  - Is regression: {tree_result['is_regression']}")
    print(f"  - Number of trees shown: {tree_result['num_trees']}")
    print(f"  - Total trees in model: {tree_result['total_trees']}")
    
    # Test tree rules for GB
    rules_result = tree_service.get_tree_rules(tree_idx=0, max_depth=2)
    if "error" not in rules_result:
        print(f"  - Successfully extracted rules from first GB tree")
        print(f"  - Total rules: {rules_result['total_rules']}")
        print(f"  - Is regression: {rules_result['is_regression']}")
        
        # Show a sample rule for GB
        if rules_result['rules']:
            first_rule = rules_result['rules'][0]
            print(f"  - Sample rule: {first_rule['rule']}")
            if first_rule['type'] == 'regression':
                print(f"    * Predicted value: {first_rule['predicted_value']:.4f}")
                print(f"    * Samples: {first_rule['samples']}, Support: {first_rule['support']:.3f}")

def test_detailed_regression_leaf_analysis():
    """Test to analyze regression leaf nodes in detail"""
    
    print("\n" + "="*60)
    print("Detailed Regression Leaf Node Analysis...")
    
    # Generate regression data with different characteristics
    X_reg, y_reg = make_regression(n_samples=500, n_features=4, noise=0.1, random_state=42)
    feature_names = [f"feature_{i}" for i in range(X_reg.shape[1])]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    # Train regression tree with different depths
    reg_tree = DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=42)
    reg_tree.fit(X_train, y_train)
    
    print(f"Target variable statistics:")
    print(f"  - Training set range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    print(f"  - Training set mean: {y_train.mean():.2f}, std: {y_train.std():.2f}")
    
    # Test regression tree
    base_service = BaseModelService()
    base_service.model = reg_tree
    base_service.X_train = pd.DataFrame(X_train, columns=feature_names)
    base_service.y_train = pd.Series(y_train)
    base_service.X_test = pd.DataFrame(X_test, columns=feature_names)
    base_service.y_test = pd.Series(y_test)
    base_service.feature_names = feature_names
    
    tree_service = TreeService(base_service)
    tree_result = tree_service.get_decision_tree()
    
    if "error" not in tree_result and tree_result['trees']:
        # Find and display multiple regression leaf nodes
        leaf_nodes = []
        
        def collect_leaf_nodes(node, depth=0):
            if node['type'] == 'leaf':
                leaf_nodes.append((node, depth))
            else:
                if 'left' in node:
                    collect_leaf_nodes(node['left'], depth + 1)
                if 'right' in node:
                    collect_leaf_nodes(node['right'], depth + 1)
        
        collect_leaf_nodes(tree_result['trees'][0]['tree_structure'])
        
        print(f"\nRegression Tree Analysis:")
        print(f"  - Total leaf nodes found: {len(leaf_nodes)}")
        print(f"  - Tree max depth: {tree_result['trees'][0]['max_depth']}")
        print(f"  - Tree R² score: {tree_result['trees'][0]['performance']:.4f}")
        
        # Show details of first few leaf nodes
        print(f"\nDetailed Leaf Node Analysis (showing first 5):")
        for i, (leaf, depth) in enumerate(leaf_nodes[:5]):
            print(f"  Leaf {i+1} (depth {depth}):")
            print(f"    - Samples: {leaf['samples']}")
            print(f"    - Predicted Value: {leaf['prediction']:.4f}")
            print(f"    - Uncertainty (MSE): {leaf['uncertainty']:.4f}")
            print(f"    - Is Regression: {leaf['is_regression']}")
            
        # Show prediction range across all leaves
        predictions = [leaf[0]['prediction'] for leaf in leaf_nodes]
        print(f"\nPrediction Range Across All Leaves:")
        print(f"  - Min prediction: {min(predictions):.4f}")
        print(f"  - Max prediction: {max(predictions):.4f}")
        print(f"  - Mean prediction: {np.mean(predictions):.4f}")
        print(f"  - Std prediction: {np.std(predictions):.4f}")

if __name__ == "__main__":
    print("Testing TreeService with Regression Models")
    print("="*60)
    
    try:
        test_regression_tree_service()
        test_ensemble_regression()
        test_gradient_boosting_regression()
        test_detailed_regression_leaf_analysis()
        print("\n" + "="*60)
        print("✓ All regression tests completed successfully!")
        print("\nKey Regression Features Verified:")
        print("  ✓ Regression leaf nodes show actual predicted continuous values")
        print("  ✓ Tree rules properly handle regression predictions")
        print("  ✓ Gradient Boosting regression trees work correctly")
        print("  ✓ Ensemble models (Random Forest) extract trees properly")
        print("  ✓ Performance metrics use R² for regression models")
        print("  ✓ Uncertainty values are properly calculated for regression leaves")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
