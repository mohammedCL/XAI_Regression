import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from .base_model_service import BaseModelService


class PredictionService:
    """
    Service for individual prediction analysis and what-if scenarios.
    """
    
    def __init__(self, base_service: BaseModelService):
        self.base = base_service

    def individual_prediction(self, instance_idx: int) -> Dict[str, Any]:
        """Get detailed prediction analysis for a single instance."""
        self.base._is_ready()
        
        if not (0 <= instance_idx < len(self.base.X_df)):
            raise ValueError(f"Instance index {instance_idx} is out of range. Dataset has {len(self.base.X_df)} instances.")
        
        instance_data = self.base.X_df.iloc[instance_idx]
        instance_df = instance_data.to_frame().T
        
        # Regression model
        prediction_value = float(self.base.model.predict(instance_df)[0])
        actual_value = float(self.base.y_s.iloc[instance_idx])
        prediction_error = abs(prediction_value - actual_value)
        
        # For regression, confidence is based on prediction uncertainty
        # Simple heuristic: smaller residuals suggest higher confidence
        all_predictions = self.base.safe_predict(self.base.X_df)
        all_errors = np.abs(all_predictions - self.base.y_s)
        max_error = np.max(all_errors) if len(all_errors) > 0 else 1.0
        confidence = 1.0 - (prediction_error / max_error) if max_error > 0 else 1.0
        
        shap_vals_for_instance = self.base._get_instance_shap_vector(instance_idx)
        
        # Handle case where explainer might not be available
        base_value = 0.0
        if self.base.explainer and hasattr(self.base.explainer, 'expected_value'):
            if isinstance(self.base.explainer.expected_value, (list, np.ndarray)):
                base_value = float(self.base.explainer.expected_value[0] if len(self.base.explainer.expected_value) > 0 else 0.0)
            else:
                base_value = float(self.base.explainer.expected_value)

        contributions = [
            {
                "name": name,
                "value": self.base._safe_float(instance_data[name]),
                "shap": float(shap_vals_for_instance[i])
            }
            for i, name in enumerate(self.base.feature_names)
        ]
        contributions.sort(key=lambda x: abs(x["shap"]), reverse=True)

        return {
            "prediction_value": prediction_value,
            "actual_value": actual_value,
            "prediction_error": prediction_error,
            "confidence_score": confidence,
            "base_value": base_value,
            "shap_values": [float(v) for v in shap_vals_for_instance],
            "feature_contributions": contributions,
            "model_type": "regression"
        }

    def explain_instance(self, instance_idx: int) -> Dict[str, Any]:
        """Explain a single instance prediction with detailed SHAP analysis."""
        self.base._is_ready()
        
        if not (0 <= instance_idx < len(self.base.X_df)):
            raise ValueError(f"Instance index {instance_idx} is out of range. Dataset has {len(self.base.X_df)} instances.")
            
        instance_data = self.base.X_df.iloc[instance_idx]
        shap_vals_for_instance = self.base._get_instance_shap_vector(instance_idx)
        
        # Handle case where explainer might not be available
        base_value = 0.0
        if self.base.explainer and hasattr(self.base.explainer, 'expected_value'):
            if isinstance(self.base.explainer.expected_value, (list, np.ndarray)):
                base_value = float(self.base.explainer.expected_value[0] if len(self.base.explainer.expected_value) > 0 else 0.0)
            else:
                base_value = float(self.base.explainer.expected_value)

        # Create single-row DataFrame for prediction to maintain feature names
        instance_df = pd.DataFrame([instance_data], columns=self.base.feature_names)
        
        # Regression model
        prediction_value = float(self.base.safe_predict(instance_df)[0])
        actual_value = float(self.base.y_s.iloc[instance_idx])

        # Prepare both mapping and ordered arrays for convenience on the frontend
        shap_mapping = dict(zip(self.base.feature_names, shap_vals_for_instance))
        ordered = sorted(shap_mapping.items(), key=lambda kv: abs(kv[1]), reverse=True)
        ordered_features = [name for name, _ in ordered]
        ordered_values = [float(val) for _, val in ordered]
        ordered_feature_values = [instance_data[name] for name in ordered_features]

        return {
            "instance_id": instance_idx,
            "features": instance_data.to_dict(),
            "prediction": prediction_value,
            "actual_value": actual_value,
            "base_value": float(base_value),
            "shap_values_map": shap_mapping,
            "ordered_contributions": {
                "feature_names": ordered_features,
                "feature_values": [self.base._safe_float(v) for v in ordered_feature_values],
                "shap_values": ordered_values
            },
            "model_type": "regression"
        }

    def perform_what_if(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Perform what-if analysis by modifying feature values."""
        self.base._is_ready()
        
        try:
            # Start with the first instance as base
            base_instance = self.base.X_df.iloc[0].copy()
            
            # Update with provided feature values
            for feature_name, value in features.items():
                if feature_name in base_instance.index:
                    base_instance[feature_name] = value
                else:
                    raise ValueError(f"Feature '{feature_name}' not found in dataset.")
            
            # Make prediction
            instance_df = pd.DataFrame([base_instance], columns=self.base.feature_names)
            
            # Regression model
            prediction_value = float(self.base.safe_predict(instance_df)[0])
            
            # Get SHAP explanation if available
            shap_explanation = {}
            if self.base.explainer:
                try:
                    shap_values = self.base.explainer.shap_values(instance_df)
                    if isinstance(shap_values, list):
                        shap_vals = shap_values[0] if len(shap_values) > 0 else []
                    else:
                        if len(shap_values.shape) == 2:
                            shap_vals = shap_values[0]
                        else:
                            shap_vals = shap_values
                    
                    shap_explanation = dict(zip(self.base.feature_names, [float(v) for v in shap_vals]))
                except Exception as e:
                    print(f"SHAP explanation failed: {e}")
            
            return {
                "modified_features": features,
                "prediction_value": prediction_value,
                "feature_values": base_instance.to_dict(),
                "shap_explanations": shap_explanation,
                "model_type": "regression"
            }
            
        except Exception as e:
            raise ValueError(f"What-if analysis failed: {str(e)}")
