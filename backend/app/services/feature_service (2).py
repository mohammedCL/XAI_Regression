import pandas as pd
import numpy as np
from typing import Dict, Any, List
from .base_model_service import BaseModelService
import hashlib
import time


class FeatureService:
    """
    Service for feature-related operations like feature importance, metadata,
    correlation analysis, and feature interactions.
    """

    def __init__(self, base_service: BaseModelService):
        self.base = base_service
        self._importance_cache: Dict[str, Dict[str, Any]] = {}
        self._correlation_cache: Dict[str, Dict[str, Any]] = {}

    # ---------- Fingerprinting ----------
    def generate_data_fingerprint(self) -> str:
        try:
            data_structure = f"{self.base.X_train.shape}|{list(self.base.X_train.columns)}"
            data_sample = pd.util.hash_pandas_object(self.base.X_train.head(100)).values
            fingerprint_input = f"{data_structure}|{data_sample.tobytes()}"
            return hashlib.md5(fingerprint_input.encode()).hexdigest()[:12]
        except Exception:
            return hashlib.md5(f"{self.base.X_train.shape}".encode()).hexdigest()[:12]

    def generate_model_fingerprint(self) -> str:
        try:
            model_params = str(sorted(self.base.model.get_params().items()))
            model_type = type(self.base.model).__name__
            fingerprint_input = f"{model_type}|{model_params}"
            return hashlib.md5(fingerprint_input.encode()).hexdigest()[:12]
        except Exception:
            model_type = type(self.base.model).__name__
            return hashlib.md5(model_type.encode()).hexdigest()[:12]

    def _is_cache_valid(self, cache_entry: Dict[str, Any], method: str) -> bool:
        try:
            ttl_config = {'shap': 1800, 'gain': 300, 'builtin': 300, 'permutation': 900}
            elapsed = time.time() - cache_entry['timestamp']
            ttl = ttl_config.get(method, 600)
            if elapsed >= ttl:
                return False
            return (cache_entry['data_fingerprint'] == self.generate_data_fingerprint() and
                    cache_entry.get('model_fingerprint', '') == self.generate_model_fingerprint())
        except Exception:
            return False

    def _cleanup_stale_cache(self):
        try:
            current_data_fp = self.generate_data_fingerprint()
            current_model_fp = self.generate_model_fingerprint()
            keys_to_remove = [
                key for key, entry in self._importance_cache.items()
                if (entry.get('data_fingerprint') != current_data_fp or
                    entry.get('model_fingerprint') != current_model_fp)
            ]
            for key in keys_to_remove:
                del self._importance_cache[key]
        except Exception:
            pass

    # ---------- Feature Importance ----------
    def get_feature_importance(self, method: str) -> Dict[str, Any]:
        advanced_result = self.compute_feature_importance_advanced(method, 'importance', 20, 'bar')
        if 'error' in advanced_result:
            raise ValueError(advanced_result['error'])
        features = [{"name": item["name"], "importance": item["importance_score"]}
                    for item in advanced_result["features"]]
        return {"method": method, "features": features}

    def compute_feature_importance_advanced(self, method: str = 'shap',
                                            sort_by: str = 'importance',
                                            top_n: int = 20,
                                            visualization: str = 'bar') -> Dict[str, Any]:
        self.base._is_ready()

        data_fp = self.generate_data_fingerprint()
        model_fp = self.generate_model_fingerprint()
        cache_key = f"{method}|{sort_by}|{top_n}|{visualization}|{data_fp}|{model_fp}"

        # Cache check
        if cache_key in self._importance_cache:
            cache_entry = self._importance_cache[cache_key]
            if self._is_cache_valid(cache_entry, method):
                return cache_entry['result']
            else:
                del self._importance_cache[cache_key]

        if len(self._importance_cache) > 10:
            self._cleanup_stale_cache()

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
            error_payload = {
                "error": str(e),
                "total_features": len(self.base.feature_names),
                "positive_impact_count": 0,
                "negative_impact_count": 0,
                "features": [],
                "computation_method": method,
                "computed_at": pd.Timestamp.utcnow().isoformat()
            }
            if method == 'shap':
                error_payload["suggested_fallback"] = "gain"
            elif method == 'gain':
                error_payload["suggested_fallback"] = "permutation"
            return error_payload

        # Build items
        items = []
        for idx, name in enumerate(self.base.feature_names):
            importance = float(raw_importance[idx])
            impact_direction = 'positive' if importance >= 0 else 'negative'
            items.append({
                "name": name,
                "importance_score": abs(importance),
                "importance": importance,
                "impact_direction": impact_direction,
                "rank": 0
            })

        if sort_by == 'feature_name':
            items.sort(key=lambda x: x['name'])
        elif sort_by == 'impact':
            items.sort(key=lambda x: abs(x['importance']), reverse=True)
        else:
            items.sort(key=lambda x: x['importance_score'], reverse=True)

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

        self._importance_cache[cache_key] = {
            'result': payload,
            'timestamp': time.time(),
            'method': method,
            'data_fingerprint': data_fp,
            'model_fingerprint': model_fp
        }
        return payload

    # ---------- Importance Helpers ----------
    def _compute_shap_importance(self) -> np.ndarray:
        if self.base.shap_values is None:
            raise ValueError("SHAP values not available.")
        shap_vals = self.base._get_shap_values_for_analysis()
        if shap_vals is None:
            raise ValueError("Failed to compute SHAP importance.")
        return np.abs(shap_vals).mean(axis=0)

    def _compute_permutation_importance(self) -> np.ndarray:
        from sklearn.inspection import permutation_importance
        X_test = getattr(self.base, 'X_test', self.base.X_train)
        y_test = getattr(self.base, 'y_test', self.base.y_train)
        perm = permutation_importance(
            self.base.model, X_test, y_test,
            n_repeats=10, random_state=42,
            scoring='neg_mean_squared_error', n_jobs=1
        )
        return perm.importances_mean

    def _compute_gain_importance(self) -> np.ndarray:
        if hasattr(self.base.model, 'feature_importances_'):
            return self.base.model.feature_importances_
        if hasattr(self.base.model, 'model') and hasattr(self.base.model.model, 'feature_importances_'):
            return self.base.model.model.feature_importances_
        raise ValueError("Model doesn't support gain-based feature importance.")

    # ---------- Feature Metadata / Controls ----------
    def get_feature_metadata(self) -> Dict[str, Any]:
        self.base._is_ready()
        return {
            "total_features": len(self.base.feature_names),
            "features": [
                {"name": name, "dtype": str(self.base.X_train[name].dtype)}
                for name in self.base.feature_names
            ]
        }

    # ---------- Correlation Matrix ----------
    def _encode_mixed_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode mixed dataframe for correlation computation."""
        encoded_cols = {}
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c].dtype):
                encoded_cols[c] = df[c].fillna(df[c].median())
            else:
                encoded_cols[c] = pd.Categorical(df[c].fillna('missing')).codes
        return pd.DataFrame(encoded_cols)

    def compute_correlation(self, selected_features: List[str]) -> Dict[str, Any]:
        """Compute correlation matrix for selected features with TTL caching."""
        self.base._is_ready()

        if not selected_features or len(selected_features) < 2:
            raise ValueError("At least 2 features must be selected for correlation analysis.")

        for feat in selected_features:
            if feat not in self.base.feature_names:
                raise ValueError(f"Feature '{feat}' not found in dataset.")

        data_fp = self.generate_data_fingerprint()
        cache_key = "|".join(sorted(selected_features))

        # Check cache with TTL validation (default 300s)
        if cache_key in self._correlation_cache:
            cache_entry = self._correlation_cache[cache_key]
            elapsed = time.time() - cache_entry['timestamp']
            ttl = 300
            if elapsed < ttl and cache_entry['data_fingerprint'] == data_fp:
                return cache_entry['result']
            else:
                del self._correlation_cache[cache_key]

        # Recompute correlation
        df = self.base.X_train[selected_features]
        enc = self._encode_mixed_dataframe(df)
        corr = enc.corr(method='pearson')
        matrix = corr.loc[selected_features, selected_features].to_numpy().tolist()

        payload = {
            "features": selected_features,
            "matrix": [[float(v) for v in row] for row in matrix],
            "computed_at": pd.Timestamp.utcnow().isoformat()
        }

        # Store in cache with timestamp + fingerprint
        self._correlation_cache[cache_key] = {
            'result': payload,
            'timestamp': time.time(),
            'data_fingerprint': data_fp
        }

        return payload
