# ExplainableAI - Regression Model Analysis Dashboard

A comprehensive web-based platform for analyzing, visualizing, and understanding regression machine learning models through advanced explainability techniques powered by SHAP, AI explanations, and interactive visualizations.

## 🎯 Overview

ExplainableAI provides an intuitive interface for data scientists and machine learning practitioners to:

- **Upload and analyze** sklearn regression models (.joblib, .pkl)
- **Generate comprehensive insights** using SHAP (SHapley Additive exPlanations)
- **Visualize feature importance** and model behavior through interactive charts
- **Perform what-if analysis** to understand prediction changes
- **Get AI-powered explanations** using AWS Bedrock (Claude 3 Sonnet)
- **Analyze model performance** with regression metrics, residual analysis, and diagnostic plots
- **Explore feature dependencies** with partial dependence plots and SHAP dependence plots

## 🏗️ Architecture

### Backend (FastAPI)
- **Framework**: FastAPI with async support
- **ML Support**: scikit-learn regression models
- **Explainability**: SHAP library for model interpretability
- **AI Explanations**: AWS Bedrock integration with Claude 3 Sonnet
- **Authentication**: Token-based authentication system
- **Data Storage**: Local file storage for models and datasets

### Frontend (React + TypeScript)
- **Framework**: React 19 with TypeScript
- **Build Tool**: Vite for fast development and building
- **Styling**: Tailwind CSS for responsive design
- **Visualizations**: 
  - Plotly.js for interactive charts
  - Recharts for data visualization
  - Matplotlib for diagnostic plots
- **Navigation**: React Router for SPA routing

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 18+
- npm or yarn

### Backend Setup

1. **Navigate to backend directory**
   ```bash
   cd backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables** (Optional - for AI explanations)
   ```bash
   export AWS_ACCESS_KEY_ID_LLM="your_aws_access_key"
   export AWS_SECRET_ACCESS_KEY_LLM="your_aws_secret_key"
   export AWS_SESSION_TOKEN_LLM="your_aws_session_token"  # Optional
   export REGION_LLM="us-east-1"  # Optional
   ```

5. **Start the server**
   ```bash
   uvicorn main:app --reload
   ```
   The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```
   The frontend will be available at `http://localhost:5173`

## 📊 Features

### Model Analysis
- **Model Overview**: Comprehensive statistics, performance metrics, and metadata
- **Regression Stats**: R², RMSE, MAE, MAPE, explained variance, and residual analysis
- **Feature Importance**: SHAP-based and built-in importance rankings
- **Diagnostic Plots**: Residuals vs fitted, predicted vs actual, Q-Q plots
- **Performance Summary**: Model quality assessment and overfitting detection

### Explainability & Interpretability
- **Instance Explanations**: SHAP values for individual predictions
- **Feature Dependence**: Partial dependence plots and SHAP dependence plots
- **What-If Analysis**: Real-time prediction changes with feature modifications
- **Feature Interactions**: Pairwise feature interaction analysis
- **Decision Tree Visualization**: Explore ensemble tree structures

### Data Analysis
- **Dataset Comparison**: Training vs test dataset statistics and drift detection
- **Feature Correlations**: Correlation analysis between selected features
- **Data Quality**: Missing values, duplicates, and health scores
- **Interactive Visualizations**: Scatter plots, heatmaps, and residual plots

### AI-Powered Insights
- **Natural Language Explanations**: AI-generated interpretations of analysis results
- **Context-Aware Descriptions**: Explanations tailored to different analysis types
- **Business Impact**: Translation of technical metrics into business insights

## 🛠️ API Endpoints

### Authentication
All endpoints require a token parameter. For development, use `token=dev_token`.

### Core Endpoints
- `POST /upload/model-and-data` - Upload model and dataset
- `POST /upload/model-and-separate-datasets` - Upload model with separate train/test data
- `GET /analysis/overview` - Get model overview and performance metrics
- `GET /analysis/regression-stats` - Get detailed regression statistics
- `GET /analysis/feature-importance` - Get feature importance rankings

### Explainability Endpoints
- `GET /analysis/explain-instance/{instance_idx}` - Explain individual prediction
- `POST /analysis/what-if` - Perform what-if analysis
- `GET /analysis/feature-dependence/{feature_name}` - Get feature dependence
- `POST /analysis/explain-with-ai` - Get AI-powered explanations

### Advanced Analysis
- `POST /api/correlation` - Feature correlation analysis
- `POST /api/residual-analysis` - Residual analysis and diagnostics
- `POST /api/partial-dependence` - Partial dependence plots
- `POST /api/interaction-network` - Feature interaction network

## 🧪 Supported Models

### Model Formats
- **scikit-learn**: `.joblib`, `.pkl`, `.pickle` files

### Model Types
- **Linear Regression**: Linear models, Ridge, Lasso, Elastic Net
- **Tree-based Models**: Random Forest, Gradient Boosting, Decision Trees
- **Support Vector Regression**: SVR with various kernels
- **Neural Networks**: MLPRegressor and compatible models

### Data Formats
- **CSV Files**: Training and test datasets
- **Features**: Numeric and categorical features
- **Target**: Continuous numeric values

## 🔧 Configuration

### Environment Variables
```bash
# AWS Bedrock Configuration (Optional)
AWS_ACCESS_KEY_ID_LLM=your_access_key
AWS_SECRET_ACCESS_KEY_LLM=your_secret_key
AWS_SESSION_TOKEN_LLM=your_session_token
REGION_LLM=us-east-1

# Storage Configuration
STORAGE_DIR=./storage  # Default: backend/storage
```

### Model Requirements
- Models must be trained and saved using supported formats
- Feature names should be consistent between training and inference
- Regression models should output continuous numeric predictions

## 📁 Project Structure

```
ExplainableAI/
├── backend/                    # FastAPI Backend
│   ├── main.py                # Main application entry point
│   ├── requirements.txt       # Python dependencies
│   ├── app/
│   │   ├── core/             # Core configuration and auth
│   │   ├── services/         # Business logic services
│   │   │   ├── model_service.py
│   │   │   └── ai_explanation_service.py
│   │   └── storage/          # File storage
│   └── storage/              # Uploaded models and datasets
├── frontend/                 # React Frontend
│   ├── package.json         # Node.js dependencies
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── services/        # API services
│   │   └── App.tsx         # Main application component
│   └── public/             # Static assets
├── *.csv                    # Sample datasets
├── *.joblib                 # Sample models
└── test_*.py               # Test scripts
```

## 🧪 Testing

### Sample Data
The repository includes comprehensive datasets and models for testing:

#### Available Datasets
- `california_housing_regression.csv` - California housing prices dataset
- `diabetes_regression.csv` - Diabetes progression dataset  
- `complex_nonlinear_regression.csv` - Synthetic nonlinear regression dataset
- `synthetic_polynomial_regression.csv` - Polynomial features dataset
- `boston_housing_regression.csv` - Boston housing market analysis
- `energy_efficiency_regression.csv` - Building energy efficiency prediction
- `concrete_strength_regression.csv` - Concrete compressive strength
- `wine_quality_regression.csv` - Wine quality scoring
- `insurance_cost_regression.csv` - Medical insurance cost prediction

#### Pre-trained Models
Each dataset includes 10 trained models:
- Linear Regression
- Ridge Regression  
- Lasso Regression
- Elastic Net
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Regression (SVR)
- Scaled Linear Regression
- Scaled SVR

### Test Scripts
- `test_upload.py` - Test file upload functionality
- `create_regression_datasets.py` - Generate multiple regression datasets
- `switch_model.py` - Switch between different datasets and models
- `validate_ui_data_display.py` - Validate frontend integration

### Running Tests
```bash
# Test backend functionality
python test_upload.py

# Create multiple regression datasets
python create_regression_datasets.py

# Switch between models for testing
python switch_model.py

# Validate API endpoints
python validate_ui_data_display.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Common Issues
- **SHAP Errors**: Ensure your model is compatible with SHAP explainers
- **Memory Issues**: Use smaller datasets or sample sizes for large models
- **AWS Credentials**: Set up proper AWS credentials for AI explanations

### Documentation
- **FastAPI Docs**: Available at `http://localhost:8000/docs` when running
- **SHAP Documentation**: https://shap.readthedocs.io/
- **AWS Bedrock**: https://docs.aws.amazon.com/bedrock/

## 🔄 Recent Updates

- ✅ Comprehensive regression model support
- ✅ AI-powered explanations with AWS Bedrock integration
- ✅ Advanced residual analysis and diagnostic plots
- ✅ Multiple dataset generation for testing
- ✅ Feature interaction analysis
- ✅ Data drift detection
- ✅ Comprehensive test coverage
- ✅ Model switching capabilities

## 🚧 Roadmap

- [ ] Model comparison dashboard
- [ ] Automated report generation
- [ ] Model monitoring and alerting
- [ ] Integration with MLflow/Weights & Biases
- [ ] Time series regression support
- [ ] Advanced feature engineering insights
- [ ] Model fairness and bias detection
- [ ] Ensemble model analysis
