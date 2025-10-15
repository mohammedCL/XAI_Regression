# API Update Summary: Stateless S3-based Endpoints

## Overview
All analysis, prediction, and feature endpoints have been refactored to follow a stateless pattern. Each request must provide S3 URLs for the model, train dataset, test dataset, and target column in the payload. The backend loads model/data from S3 for every request, ensuring robustness and scalability.

## Example Payload (for POST endpoints)
```json
{
  "model": "<S3 URL to model>",
  "train_dataset": "<S3 URL to train dataset>",
  "test_dataset": "<S3 URL to test dataset>",
  "target_column": "target",
  // ...additional parameters...
}
```

## Updated Endpoints
- `/analysis/overview`
- `/analysis/regression-stats`
- `/analysis/feature-importance`
- `/analysis/explain-instance`
- `/analysis/what-if`
- `/analysis/feature-dependence`
- `/analysis/instances`
- `/analysis/dataset-comparison`
- `/analysis/feature-interactions`
- `/analysis/decision-tree`
- `/api/correlation`
- `/api/feature-importance`
- `/api/individual-prediction`
- `/api/partial-dependence`
- `/api/shap-dependence`
- `/api/ice-plot`
- `/api/interaction-network`
- `/api/pairwise-analysis`

## Key Changes
- All endpoints now require S3 URLs and target column in the request payload.
- Endpoints load model/data from S3 for every request (stateless, no in-memory session).
- Improved reliability for scaling, multi-instance deployments, and backend restarts.
- Legacy local file logic removed or refactored.

## Benefits
- Stateless design supports microservices, serverless, and cloud-native architectures.
- No dependency on server memory or session state between requests.
- Easier scaling, load balancing, and fault tolerance.

## Migration Notes
- Update frontend and client code to send S3 URLs and target column in every request.
- Remove any logic that assumes persistent backend state or local file uploads.

## Contact
For questions or further changes, contact the backend maintainers.
