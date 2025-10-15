# --- API Documentation: Stateless S3-based Endpoints ---
#
# All analysis, prediction, and feature endpoints now require S3 URLs for model, train dataset, test dataset, and target column in the request payload.
# Endpoints load model/data from S3 for every request (stateless pattern).
#
# Example payload (for POST endpoints):
# {
#   "model": "<S3 URL to model>",
#   "train_dataset": "<S3 URL to train dataset>",
#   "test_dataset": "<S3 URL to test dataset>",
#   "target_column": "target",
#   ...additional parameters...
# }
#
# Updated endpoints:
# - /analysis/overview
# - /analysis/regression-stats
# - /analysis/feature-importance
# - /analysis/explain-instance
# - /analysis/what-if
# - /analysis/feature-dependence
# - /analysis/instances
# - /analysis/dataset-comparison
# - /analysis/feature-interactions
# - /analysis/decision-tree
# - /api/correlation
# - /api/feature-importance
# - /api/individual-prediction
# - /api/partial-dependence
# - /api/shap-dependence
# - /api/ice-plot
# - /api/interaction-network
# - /api/pairwise-analysis
#
# All endpoints are now stateless and robust to backend restarts, scaling, and multi-instance deployments.
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Depends, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from typing import Dict
from typing import List

from app.core.config import settings
from app.core.auth import verify_token
from app.services.model_service import ModelService
from app.services.ai_explanation_service import AIExplanationService
import pandas as pd
import joblib
import requests
import tempfile
import os
import math
from typing import Dict
from fastapi import APIRouter, Depends, Body, HTTPException
from pydantic import BaseModel
app = FastAPI(title=settings.PROJECT_NAME, version=settings.PROJECT_VERSION)

# --- S3 Utility Functions ---

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service = ModelService()
ai_explanation_service = AIExplanationService()


# Auto-load functionality disabled as requested by user
# User will upload model and dataset through the frontend interface

# --- Utility Function for Error Handling ---
def handle_request(service_func, *args, **kwargs):
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        else:
            return obj

    try:
        result = service_func(*args, **kwargs)
        # # Debug: print the result before returning as JSON
        # import pprint
        # print("[DEBUG] handle_request result:")
        # pprint.pprint(result)
        sanitized_result = sanitize(result)
        return JSONResponse(status_code=200, content=sanitized_result)
    except ValueError as e:
        import traceback
        print("[DEBUG] ValueError in handle_request:")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # For debugging, print the full error
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# --- API Endpoints ---

@app.get("/", tags=["Root"])
def read_root():
    return {"message": f"Welcome to the {settings.PROJECT_NAME}"}

@app.get("/api/files")
def get_s3_file_metadata():
    """
    Lists files and models from the external S3 API and returns their metadata (name, URL, folder).
    Separates files and models based on the folder field.
    """
    file_api = "http://xailoadbalancer-579761463.ap-south-1.elb.amazonaws.com/api/files_download"
    token = ""
    EXTERNAL_S3_API_URL = f"{file_api}/Regression"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    try:
        response = requests.get(EXTERNAL_S3_API_URL, headers=headers)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        json_data = response.json()
        all_items = json_data.get("files", [])
        
        # Separate files and models based on folder
        files = [item for item in all_items if item.get("folder") == "files"]
        models = [item for item in all_items if item.get("folder") == "models"]
        
        return {
            "success": True,
            "files": files,
            "models": models,
            "total_files": len(files),
            "total_models": len(models)
        }
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to external S3 API: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to connect to external S3 API: {str(e)}")
    except Exception as e:
        print(f"Error processing external S3 API response: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing S3 API response: {str(e)}")



class LoadDataRequest(BaseModel):
    model: str
    train_dataset: str = None
    test_dataset: str = None
    target_column: str = "target"


@app.post("/load")
async def load_data(payload: LoadDataRequest):
    
    try:
        model_name = payload.model
        train_dataset = payload.train_dataset
        test_dataset = payload.test_dataset
        target_column = payload.target_column

        if not model_name:
            raise HTTPException(status_code=400, detail="Missing model name")

        return handle_request(model_service.load_model_and_datasets, model_path=model_name, train_data_path=train_dataset, test_data_path=test_dataset, target_column=target_column)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/analysis/overview", tags=["Analysis"])
async def get_overview(payload: Dict = Body(...), token: str = Depends(verify_token)):
    model_url = payload.get("model")
    train_data_url = payload.get("train_dataset")
    test_data_url = payload.get("test_dataset")
    target_column = payload.get("target_column", "target")
    if not model_url or not train_data_url or not test_data_url:
        raise HTTPException(status_code=400, detail="Missing S3 URLs for model or datasets")
    try:
        model_service.load_model_and_datasets(
            model_path=model_url,
            train_data_path=train_data_url,
            test_data_path=test_data_url,
            target_column=target_column
        )
        return handle_request(model_service.get_model_overview)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model/data from S3: {e}")

@app.post("/analysis/regression-stats", tags=["Analysis"])
async def get_regression_statistics(payload: Dict = Body(...), token: str = Depends(verify_token)):
    model_url = payload.get("model")
    train_data_url = payload.get("train_dataset")
    test_data_url = payload.get("test_dataset")
    target_column = payload.get("target_column", "target")
    if not model_url or not train_data_url or not test_data_url:
        raise HTTPException(status_code=400, detail="Missing S3 URLs for model or datasets")
    try:
        model_service.load_model_and_datasets(
            model_path=model_url,
            train_data_path=train_data_url,
            test_data_path=test_data_url,
            target_column=target_column
        )
        return handle_request(model_service.get_regression_stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model/data from S3: {e}")

@app.post("/analysis/feature-importance", tags=["Analysis"])
async def get_feature_importance(payload: Dict = Body(...), token: str = Depends(verify_token)):
    """
    Expects payload:
    {
        "model": "<S3 URL to model>",
        "train_dataset": "<S3 URL to train dataset>",
        "test_dataset": "<S3 URL to test dataset>",
        "target_column": "target",
        "method": "shap"
    }
    """
    model_url = payload.get("model")
    train_data_url = payload.get("train_dataset")
    test_data_url = payload.get("test_dataset")
    target_column = payload.get("target_column", "target")
    method = payload.get("method", "shap")
    if not model_url or not train_data_url or not test_data_url:
        raise HTTPException(status_code=400, detail="Missing S3 URLs for model or datasets")
    try:
        # Use the existing load_model_and_datasets method
        model_service.load_model_and_datasets(
            model_path=model_url,
            train_data_path=train_data_url,
            test_data_path=test_data_url,
            target_column=target_column
        )
        return handle_request(model_service.get_feature_importance, method)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model/data from S3: {e}")

@app.post("/analysis/explain-instance", tags=["Analysis"])
async def explain_instance(payload: Dict = Body(...), token: str = Depends(verify_token)):
    model_url = payload.get("model")
    train_data_url = payload.get("train_dataset")
    test_data_url = payload.get("test_dataset")
    target_column = payload.get("target_column", "target")
    instance_idx = int(payload.get("instance_idx", 0))
    if not model_url or not train_data_url or not test_data_url:
        raise HTTPException(status_code=400, detail="Missing S3 URLs for model or datasets")
    try:
        model_service.load_model_and_datasets(
            model_path=model_url,
            train_data_path=train_data_url,
            test_data_path=test_data_url,
            target_column=target_column
        )
        return handle_request(model_service.explain_instance, instance_idx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model/data from S3: {e}")

@app.post("/analysis/what-if", tags=["Analysis"])
async def perform_what_if(payload: Dict = Body(...), token: str = Depends(verify_token)):
    model_url = payload.get("model")
    train_data_url = payload.get("train_dataset")
    test_data_url = payload.get("test_dataset")
    target_column = payload.get("target_column", "target")
    features = payload.get("features")
    if not model_url or not train_data_url or not test_data_url:
        raise HTTPException(status_code=400, detail="Missing S3 URLs for model or datasets")
    try:
        model_service.load_model_and_datasets(
            model_path=model_url,
            train_data_path=train_data_url,
            test_data_path=test_data_url,
            target_column=target_column
        )
        return handle_request(model_service.perform_what_if, features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model/data from S3: {e}")

@app.post("/analysis/feature-dependence", tags=["Analysis"])
async def get_feature_dependence(payload: Dict = Body(...), token: str = Depends(verify_token)):
    model_url = payload.get("model")
    train_data_url = payload.get("train_dataset")
    test_data_url = payload.get("test_dataset")
    target_column = payload.get("target_column", "target")
    feature_name = payload.get("feature_name")
    if not model_url or not train_data_url or not test_data_url or not feature_name:
        raise HTTPException(status_code=400, detail="Missing S3 URLs or feature_name")
    try:
        model_service.load_model_and_datasets(
            model_path=model_url,
            train_data_path=train_data_url,
            test_data_path=test_data_url,
            target_column=target_column
        )
        return handle_request(model_service.get_feature_dependence, feature_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model/data from S3: {e}")

@app.post("/analysis/instances", tags=["Analysis"])
async def list_instances(payload: Dict = Body(...), token: str = Depends(verify_token)):
    model_url = payload.get("model")
    train_data_url = payload.get("train_dataset")
    test_data_url = payload.get("test_dataset")
    target_column = payload.get("target_column", "target")
    sort_by = payload.get("sort_by", "prediction")
    limit = int(payload.get("limit", 100))
    if not model_url or not train_data_url or not test_data_url:
        raise HTTPException(status_code=400, detail="Missing S3 URLs for model or datasets")
    try:
        model_service.load_model_and_datasets(
            model_path=model_url,
            train_data_path=train_data_url,
            test_data_path=test_data_url,
            target_column=target_column
        )
        return handle_request(model_service.list_instances, sort_by, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model/data from S3: {e}")

@app.post("/analysis/dataset-comparison", tags=["Analysis"])
async def get_dataset_comparison(payload: Dict = Body(...), token: str = Depends(verify_token)):
    model_url = payload.get("model")
    train_data_url = payload.get("train_dataset")
    test_data_url = payload.get("test_dataset")
    target_column = payload.get("target_column", "target")
    if not model_url or not train_data_url or not test_data_url:
        raise HTTPException(status_code=400, detail="Missing S3 URLs for model or datasets")
    try:
        model_service.load_model_and_datasets(
            model_path=model_url,
            train_data_path=train_data_url,
            test_data_path=test_data_url,
            target_column=target_column
        )
        return handle_request(model_service.get_dataset_comparison)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model/data from S3: {e}")

# --- New enterprise feature endpoints ---
@app.post("/api/features", tags=["Features"])
async def get_features_metadata(payload: Dict = Body(...), token: str = Depends(verify_token)):
    model_url = payload.get("model")
    train_data_url = payload.get("train_dataset")
    test_data_url = payload.get("test_dataset")
    target_column = payload.get("target_column", "target")
    if not model_url or not train_data_url or not test_data_url:
        raise HTTPException(status_code=400, detail="Missing S3 URLs for model or datasets")
    try:
        model_service.load_model_and_datasets(
            model_path=model_url,
            train_data_path=train_data_url,
            test_data_path=test_data_url,
            target_column=target_column
        )
        return handle_request(model_service.get_feature_metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model/data from S3: {e}")

@app.post("/api/correlation", tags=["Features"])
async def post_correlation(payload: Dict = Body(...), token: str = Depends(verify_token)):
    model_url = payload.get("model")
    train_data_url = payload.get("train_dataset")
    test_data_url = payload.get("test_dataset")
    target_column = payload.get("target_column", "target")
    selected: List[str] = payload.get("features") or []
    if not model_url or not train_data_url or not test_data_url:
        raise HTTPException(status_code=400, detail="Missing S3 URLs for model or datasets")
    try:
        model_service.load_model_and_datasets(
            model_path=model_url,
            train_data_path=train_data_url,
            test_data_path=test_data_url,
            target_column=target_column
        )
        return handle_request(model_service.compute_correlation, selected)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model/data from S3: {e}")

@app.post("/api/feature-importance", tags=["Features"])
async def post_feature_importance(payload: Dict = Body(...), token: str = Depends(verify_token)):
    model_url = payload.get("model")
    train_data_url = payload.get("train_dataset")
    test_data_url = payload.get("test_dataset")
    target_column = payload.get("target_column", "target")
    method = payload.get("method", "shap")
    sort_by = payload.get("sort_by", "importance")
    top_n = int(payload.get("top_n", 20))
    visualization = payload.get("visualization", "bar")
    if not model_url or not train_data_url or not test_data_url:
        raise HTTPException(status_code=400, detail="Missing S3 URLs for model or datasets")
    try:
        model_service.load_model_and_datasets(
            model_path=model_url,
            train_data_path=train_data_url,
            test_data_path=test_data_url,
            target_column=target_column
        )
        return handle_request(model_service.compute_feature_importance_advanced, method, sort_by, top_n, visualization)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model/data from S3: {e}")


# Updated to stateless POST endpoints
@app.post("/analysis/feature-interactions", tags=["Analysis"])
async def post_feature_interactions(payload: Dict = Body(...), token: str = Depends(verify_token)):
    model_url = payload.get("model")
    train_data_url = payload.get("train_dataset")
    test_data_url = payload.get("test_dataset")
    target_column = payload.get("target_column", "target")
    feature1 = payload.get("feature1")
    feature2 = payload.get("feature2")
    if not model_url or not train_data_url or not test_data_url or not feature1 or not feature2:
        raise HTTPException(status_code=400, detail="Missing S3 URLs or feature names")
    try:
        model_service.load_model_and_datasets(
            model_path=model_url,
            train_data_path=train_data_url,
            test_data_path=test_data_url,
            target_column=target_column
        )
        return handle_request(model_service.get_feature_interactions, feature1, feature2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model/data from S3: {e}")

@app.post("/analysis/decision-tree", tags=["Analysis"])
async def post_decision_tree(payload: Dict = Body(...), token: str = Depends(verify_token)):
    model_url = payload.get("model")
    train_data_url = payload.get("train_dataset")
    test_data_url = payload.get("test_dataset")
    target_column = payload.get("target_column", "target")
    if not model_url or not train_data_url or not test_data_url:
        raise HTTPException(status_code=400, detail="Missing S3 URLs for model or datasets")
    try:
        model_service.load_model_and_datasets(
            model_path=model_url,
            train_data_path=train_data_url,
            test_data_path=test_data_url,
            target_column=target_column
        )
        return handle_request(model_service.get_decision_tree)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model/data from S3: {e}")

# --- Individual Prediction API ---  
@app.post("/api/individual-prediction", tags=["Prediction"])
async def post_individual_prediction(payload: Dict = Body(...), token: str = Depends(verify_token)):
    model_url = payload.get("model")
    train_data_url = payload.get("train_dataset")
    test_data_url = payload.get("test_dataset")
    target_column = payload.get("target_column", "target")
    instance_idx = int(payload.get("instance_idx", 0))
    if not model_url or not train_data_url or not test_data_url:
        raise HTTPException(status_code=400, detail="Missing S3 URLs for model or datasets")
    try:
        model_service.load_model_and_datasets(
            model_path=model_url,
            train_data_path=train_data_url,
            test_data_path=test_data_url,
            target_column=target_column
        )
        return handle_request(model_service.individual_prediction, instance_idx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model/data from S3: {e}")

# --- Regression Analysis Endpoints ---
@app.post("/api/partial-dependence", tags=["Dependence"])
async def post_partial_dependence(payload: Dict = Body(...), token: str = Depends(verify_token)):
    model_url = payload.get("model")
    train_data_url = payload.get("train_dataset")
    test_data_url = payload.get("test_dataset")
    target_column = payload.get("target_column", "target")
    feature_name = payload.get("feature_name")
    num_points = int(payload.get("num_points", 20))
    if not model_url or not train_data_url or not test_data_url or not feature_name:
        raise HTTPException(status_code=400, detail="Missing S3 URLs for model/datasets or 'feature'")
    try:
        model_service.load_model_and_datasets(
            model_path=model_url,
            train_data_path=train_data_url,
            test_data_path=test_data_url,
            target_column=target_column
        )
        return handle_request(model_service.partial_dependence, feature_name, num_points)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model/data from S3: {e}")

@app.post("/api/shap-dependence", tags=["Dependence"])
async def post_shap_dependence(payload: Dict = Body(...), token: str = Depends(verify_token)):
    model_url = payload.get("model")
    train_data_url = payload.get("train_dataset")
    test_data_url = payload.get("test_dataset")
    target_column = payload.get("target_column", "target")
    feature_name = payload.get("feature_name")
    color_by = payload.get("color_by")
    if not model_url or not train_data_url or not test_data_url or not feature_name:
        raise HTTPException(status_code=400, detail="Missing S3 URLs for model/datasets or 'feature'")
    try:
        model_service.load_model_and_datasets(
            model_path=model_url,
            train_data_path=train_data_url,
            test_data_path=test_data_url,
            target_column=target_column
        )
        return handle_request(model_service.shap_dependence, feature_name, color_by)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model/data from S3: {e}")

@app.post("/api/ice-plot", tags=["Dependence"])
async def post_ice_plot(payload: Dict = Body(...), token: str = Depends(verify_token)):
    model_url = payload.get("model")
    train_data_url = payload.get("train_dataset")
    test_data_url = payload.get("test_dataset")
    target_column = payload.get("target_column", "target")
    feature_name = payload.get("feature_name")
    num_points = int(payload.get("num_points", 20))
    num_instances = int(payload.get("num_instances", 20))
    if not model_url or not train_data_url or not test_data_url or not feature_name:
        raise HTTPException(status_code=400, detail="Missing S3 URLs for model/datasets or 'feature_name'")
    try:
        model_service.load_model_and_datasets(
            model_path=model_url,
            train_data_path=train_data_url,
            test_data_path=test_data_url,
            target_column=target_column
        )
        return handle_request(model_service.ice_plot, feature_name, num_points, num_instances)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model/data from S3: {e}")

# --- Section 5 APIs ---
@app.post("/api/interaction-network", tags=["Interactions"])
async def post_interaction_network(payload: Dict = Body(...), token: str = Depends(verify_token)):
    model_url = payload.get("model")
    train_data_url = payload.get("train_dataset")
    test_data_url = payload.get("test_dataset")
    target_column = payload.get("target_column", "target")
    top_k = int(payload.get("top_k", 30))
    sample_rows = int(payload.get("sample_rows", 200))
    if not model_url or not train_data_url or not test_data_url:
        raise HTTPException(status_code=400, detail="Missing S3 URLs for model or datasets")
    try:
        model_service.load_model_and_datasets(
            model_path=model_url,
            train_data_path=train_data_url,
            test_data_path=test_data_url,
            target_column=target_column
        )
        return handle_request(model_service.interaction_network, top_k, sample_rows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model/data from S3: {e}")

@app.post("/api/pairwise-analysis", tags=["Interactions"])
async def post_pairwise_analysis(payload: Dict = Body(...), token: str = Depends(verify_token)):
    model_url = payload.get("model")
    train_data_url = payload.get("train_dataset")
    test_data_url = payload.get("test_dataset")
    target_column = payload.get("target_column", "target")
    f1 = payload.get("feature1")
    f2 = payload.get("feature2")
    color_by = payload.get("color_by")
    sample_size = int(payload.get("sample_size", 1000))
    if not model_url or not train_data_url or not test_data_url or not f1 or not f2:
        raise HTTPException(status_code=400, detail="Missing S3 URLs for model/datasets or 'feature1'/'feature2'")
    try:
        model_service.load_model_and_datasets(
            model_path=model_url,
            train_data_path=train_data_url,
            test_data_path=test_data_url,
            target_column=target_column
        )
        return handle_request(model_service.pairwise_analysis, f1, f2, color_by, sample_size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model/data from S3: {e}")

# --- AI Explanation Endpoint ---
@app.post("/analysis/explain-with-ai", tags=["AI Analysis"])
async def explain_with_ai(
    payload: Dict = Body(...),
    token: str = Depends(verify_token)
):
    """
    Generate an AI-powered explanation of the current analysis results.
    
    Expected payload:
    {
        "analysis_type": "overview|feature_importance|classification_stats|...",
        "analysis_data": {...}  # The data to be explained
    }
    """
    try:
        analysis_type = payload.get("analysis_type")
        analysis_data = payload.get("analysis_data", {})
        
        if not analysis_type:
            raise HTTPException(status_code=400, detail="Missing 'analysis_type' in payload")
        
        # Generate AI explanation
        explanation = ai_explanation_service.generate_explanation(analysis_data, analysis_type)
        
        return JSONResponse(status_code=200, content=explanation)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating AI explanation: {str(e)}")
