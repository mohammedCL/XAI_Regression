import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import MainLayout from './components/layout/MainLayout';
import ModelOverview from './components/analysis/ModelOverview';
import RegressionStats from './components/analysis/RegressionStats';
import FeatureImportance from './components/analysis/FeatureImportance';
import IndividualPredictions from './components/analysis/IndividualPredictions';
import WhatIfAnalysis from './components/analysis/WhatIfAnalysis';
import FeatureDependence from './components/analysis/FeatureDependence';
import FeatureInteractions from './components/analysis/FeatureInteractions';
import UploadPage from './components/analysis/UploadPage';
import DecisionTreesWrapper from './components/analysis/DecisionTreesWrapper';
import { S3ConfigProvider } from './context/S3ConfigContext';

function App() {
  return (
    <S3ConfigProvider>
      <Router>
        <Routes>
          <Route path="/upload" element={<UploadPage />} /> {/* New upload route */}
          <Route path="/" element={<MainLayout />}>
            {/* Default route redirects to model overview */}
            <Route index element={<Navigate to="/overview" replace />} />
            <Route path="overview" element={<ModelOverview modelType="regression" />} />
            <Route path="regression-stats" element={<RegressionStats />} />
            <Route path="feature-importance" element={<FeatureImportance modelType="regression" />} />
            <Route path="individual-predictions" element={<IndividualPredictions modelType="regression" />} />
            <Route path="what-if" element={<WhatIfAnalysis modelType="regression" />} />
            <Route path="feature-dependence" element={<FeatureDependence modelType="regression" />} />
            <Route path="feature-interactions" element={<FeatureInteractions modelType="regression" />} />
            <Route path="decision-trees" element={<DecisionTreesWrapper />} />
          </Route>
        </Routes>
      </Router>
    </S3ConfigProvider>
  );
}

export default App;