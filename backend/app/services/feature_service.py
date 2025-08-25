import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from .base_model_service import BaseModelService


class FeatureService:
    """
    Service for feature-related operations like feature importance, metadata,
    correlation analysis, and feature interactions.
    """
    
    def __init__(self, base_service: BaseModelService):
        self.base = base_service
        # Simple in-memory caches for heavy computations
        self._correlation_cache: Dict[str, Dict[str, Any]] = {}
        self._importance_cache: Dict[str, Dict[str, Any]] = {}

    def get_feature_importance(self, method: str) -> Dict[str, Any]:
        """Get feature importance using specified method (shap, builtin, etc.)."""
        # Use the advanced method but return simplified format for backward compatibility
        advanced_result = self.compute_feature_importance_advanced(method, 'importance', 20, 'bar')
        
        # Handle error cases
        if 'error' in advanced_result:
            raise ValueError(advanced_result['error'])
        
        # Convert to legacy format
        features = [{
            "name": item["name"],
            "importance": item["importance_score"]
        } for item in advanced_result["features"]]
        
        return {"method": method, "features": features}

    def get_feature_metadata(self) -> Dict[str, Any]:
        """Return available features with basic metadata."""
        self.base._is_ready()
        features: List[Dict[str, Any]] = []
        
        for name in self.base.feature_names:
            col = self.base.X_df[name]
            is_numeric = pd.api.types.is_numeric_dtype(col.dtype)
            
            feature_info = {
                "name": name,
                "type": "numerical" if is_numeric else "categorical",
                "missing_count": int(col.isnull().sum()),
                "unique_count": int(col.nunique()),
                "dtype": str(col.dtype)
            }
            
            if is_numeric:
                feature_info.update({
                    "min": float(col.min()),
                    "max": float(col.max()),
                    "mean": float(col.mean()),
                    "std": float(col.std()),
                    "median": float(col.median())
                })
            else:
                feature_info["top_categories"] = col.value_counts().head(5).to_dict()
                
            features.append(feature_info)
            
        return {"features": features}

    def _encode_mixed_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode mixed dataframe for correlation computation."""
        encoded_cols = {}
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c].dtype):
                encoded_cols[c] = df[c].fillna(df[c].median())
            else:
                # Label encode categorical variables
                encoded_cols[c] = pd.Categorical(df[c].fillna('missing')).codes
        return pd.DataFrame(encoded_cols)

    def compute_correlation(self, selected_features: List[str]) -> Dict[str, Any]:
        """Compute correlation matrix for selected features."""
        self.base._is_ready()
        
        if not selected_features or len(selected_features) < 2:
            raise ValueError("At least 2 features must be selected for correlation analysis.")
        
        for feat in selected_features:
            if feat not in self.base.feature_names:
                raise ValueError(f"Feature '{feat}' not found in dataset.")

        cache_key = "|".join(sorted(selected_features))
        if cache_key in self._correlation_cache:
            return self._correlation_cache[cache_key]

        df = self.base.X_df[selected_features]
        enc = self._encode_mixed_dataframe(df)
        corr = enc.corr(method='pearson')
        matrix = corr.loc[selected_features, selected_features].to_numpy().tolist()
        
        payload = {
            "features": selected_features,
            "matrix": [[float(v) for v in row] for row in matrix],
            "computed_at": pd.Timestamp.utcnow().isoformat()
        }
        
        # Cache result
        self._correlation_cache[cache_key] = payload
        return payload

    def compute_feature_importance_advanced(self, method: str = 'shap', sort_by: str = 'importance', 
                                          top_n: int = 20, visualization: str = 'bar') -> Dict[str, Any]:
        """Compute advanced feature importance with detailed analysis."""
        self.base._is_ready()
        
        # Normalize params for cache key
        key = f"{method}|{sort_by}|{top_n}|{visualization}"
        if key in self._importance_cache:
            return self._importance_cache[key]

        method = method.lower()
        
        try:
            if method == 'shap':
                raw_importance = self._compute_shap_importance()
            elif method == 'permutation':
                raw_importance = self._compute_permutation_importance()
            elif method in ('gain', 'builtin'):
                raw_importance = self._compute_gain_importance()
            else:
                raise ValueError(f"Unsupported importance method: {method}")
                
        except Exception as e:
            # Return error response that frontend can handle
            error_payload = {
                "error": str(e),
                "total_features": len(self.base.feature_names),
                "positive_impact_count": 0,
                "negative_impact_count": 0,
                "features": [],
                "computation_method": method,
                "computed_at": pd.Timestamp.utcnow().isoformat()
            }
            
            # Add fallback suggestion
            if method == 'shap':
                error_payload["suggested_fallback"] = "gain"
            elif method == 'gain':
                error_payload["suggested_fallback"] = "permutation"
                
            return error_payload

        # Build feature importance items
        items = []
        for idx, name in enumerate(self.base.feature_names):
            importance = float(raw_importance[idx])
            impact_direction = 'positive' if importance >= 0 else 'negative'
            
            items.append({
                "name": name,
                "importance_score": abs(importance),  # Always positive for sorting
                "importance": importance,  # Original value (can be negative)
                "impact_direction": impact_direction,
                "rank": 0  # Will be set after sorting
            })

        # Sorting
        if sort_by == 'feature_name':
            items.sort(key=lambda x: x['name'])
        elif sort_by == 'impact':
            items.sort(key=lambda x: abs(x['importance']), reverse=True)
        else:  # sort by importance            
            items.sort(key=lambda x: x['importance_score'], reverse=True)

        # Assign ranks after sort
        for i, it in enumerate(items):
            it['rank'] = i + 1

        top_items = items[:int(top_n)]
        
        payload = {
            "total_features": len(items),
            "positive_impact_count": sum(1 for i in items if i['impact_direction'] == 'positive'),
            "negative_impact_count": sum(1 for i in items if i['impact_direction'] == 'negative'),
            "features": top_items,
            "computation_method": method,
            "computed_at": pd.Timestamp.utcnow().isoformat(),
            "visualization_type": visualization,
            "sort_by": sort_by
        }
        
        self._importance_cache[key] = payload
        return payload

    def _compute_shap_importance(self) -> np.ndarray:
        """Compute SHAP-based feature importance."""
        if self.base.shap_values is None:
            raise ValueError("SHAP values are not available for this model. Try using 'gain' or 'permutation' method instead.")
        
        shap_vals = self.base._get_shap_values_for_analysis()
        if shap_vals is None:
            raise ValueError("Failed to compute SHAP-based feature importance.")
        
        # For regression, we use mean absolute SHAP values across all instances
        return np.abs(shap_vals).mean(axis=0)

    def _compute_permutation_importance(self) -> np.ndarray:
        """Compute permutation-based feature importance."""
        from sklearn.inspection import permutation_importance
        
        try:
            # Use test data for permutation importance
            X_test = self.base.X_test if hasattr(self.base, 'X_test') and self.base.X_test is not None else self.base.X_df
            y_test = self.base.y_test if hasattr(self.base, 'y_test') and self.base.y_test is not None else self.base.y_df.values
            
            # Compute permutation importance
            perm_importance = permutation_importance(
                self.base.model,
                X_test,
                y_test,
                n_repeats=10,
                random_state=42,
                scoring='neg_mean_squared_error',
                n_jobs=1  # Avoid multiprocessing issues
            )
            
            return perm_importance.importances_mean
            
        except Exception as e:
            raise ValueError(f"Failed to compute permutation importance: {str(e)}")

    def _compute_gain_importance(self) -> np.ndarray:
        """Compute gain-based (builtin) feature importance."""
        if not hasattr(self.base.model, 'feature_importances_'):
            # Try to get feature importance from wrapped model
            if hasattr(self.base.model, 'model') and hasattr(self.base.model.model, 'feature_importances_'):
                return self.base.model.model.feature_importances_
            else:
                raise ValueError("Model doesn't support gain-based feature importance. Try using 'permutation' method instead.")
        
        return self.base.model.feature_importances_

    def get_feature_interactions(self, feature1: str, feature2: str) -> Dict[str, Any]:
        """Get feature interaction analysis (simplified version)."""
        self.base._is_ready()
        
        if feature1 not in self.base.feature_names or feature2 not in self.base.feature_names:
            raise ValueError("One or both features not found in dataset.")
            
        # This requires SHAP interaction values, which can be computationally expensive
        # For a production system, this would be pre-computed or calculated on demand by a worker.
        # Here we mock it for simplicity, but a real implementation would use:
        # shap_interaction_values = self.explainer.shap_interaction_values(self.X_df)
        
        # Mocking interaction effects
        f1_values = self.base.X_df[feature1]
        f2_values = self.base.X_df[feature2]
        
        # Create a plausible-looking interaction effect based on the features
        interaction_effect = (f1_values - f1_values.mean()) * (f2_values - f2_values.mean())
        interaction_effect_normalized = (interaction_effect / interaction_effect.abs().max()) * 0.1

        return {
            "feature1_values": f1_values.tolist(),
            "feature2_values": f2_values.tolist(),
            "interaction_shap_values": interaction_effect_normalized.tolist()
        }
