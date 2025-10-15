import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000';

const apiClient = axios.create({
    baseURL: API_BASE_URL,
});

// Token management
const TOKEN_KEY = 'authToken';
const DEFAULT_TOKEN = 'test_token';

const getAuthToken = () => {
    try {
        let token = localStorage.getItem(TOKEN_KEY);
        if (!token) {
            token = DEFAULT_TOKEN;
            localStorage.setItem(TOKEN_KEY, token);
        }
        return token;
    } catch (error) {
        return DEFAULT_TOKEN;
    }
};

getAuthToken();

apiClient.interceptors.request.use((config) => {
    const token = getAuthToken();
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

// --- Stateless API Functions ---
// All endpoints require S3 URLs and target_column in payload

export const postModelOverview = (payload) => apiClient.post('/analysis/overview', payload).then(res => res.data);
export const postRegressionStats = (payload) => apiClient.post('/analysis/regression-stats', payload).then(res => res.data);
export const postFeatureImportance = (payload) => apiClient.post('/analysis/feature-importance', payload).then(res => res.data);
export const postExplainInstance = (payload) => apiClient.post('/analysis/explain-instance', payload).then(res => res.data);
export const postWhatIf = (payload) => apiClient.post('/analysis/what-if', payload).then(res => res.data);
export const postFeatureDependence = (payload) => apiClient.post('/analysis/feature-dependence', payload).then(res => res.data);
export const postListInstances = (payload) => apiClient.post('/analysis/instances', payload).then(res => res.data);
export const postDatasetComparison = (payload) => apiClient.post('/analysis/dataset-comparison', payload).then(res => res.data);
export const postFeatureInteractions = (payload) => apiClient.post('/analysis/feature-interactions', payload).then(res => res.data);
export const postDecisionTree = (payload) => apiClient.post('/analysis/decision-tree', payload).then(res => res.data);

// Enterprise feature APIs
export const postFeaturesMetadata = (payload) => apiClient.post('/api/features', payload).then(res => res.data);
export const postCorrelation = (payload) => apiClient.post('/api/correlation', payload).then(res => res.data);
export const postAdvancedImportance = (payload) => apiClient.post('/api/feature-importance', payload).then(res => res.data);
export const postIndividualPrediction = (payload) => apiClient.post('/api/individual-prediction', payload).then(res => res.data);
export const postPartialDependence = (payload) => apiClient.post('/api/partial-dependence', payload).then(res => res.data);
export const postShapDependence = (payload) => apiClient.post('/api/shap-dependence', payload).then(res => res.data);
export const postIcePlot = (payload) => apiClient.post('/api/ice-plot', payload).then(res => res.data);
export const postInteractionNetwork = (payload) => apiClient.post('/api/interaction-network', payload).then(res => res.data);
export const postPairwiseAnalysis = (payload) => apiClient.post('/api/pairwise-analysis', payload).then(res => res.data);

// AI Explanation API
export const explainWithAI = (payload) => apiClient.post('/analysis/explain-with-ai', payload).then(res => res.data);

// Migration Checklist:
// - Update all components to use these new POST functions
// - Ensure S3 URLs and target_column are collected and sent in every payload
// - Remove any GET requests for analysis endpoints
// - Remove any session/local file upload logic for backend analysis
// - Update frontend docs to reflect new API usage
