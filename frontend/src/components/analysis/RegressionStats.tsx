import React, { useState, useEffect } from 'react';
import Card from '../common/Card';
import MetricCard from '../common/MetricCard';
import AIExplanationPanel from '../common/AIExplanationPanel';
import ExplainWithAIButton from '../common/ExplainWithAIButton';
import { getRegressionStats } from '../../services/api';

interface RegressionMetrics {
    r2_score: number;
    rmse: number;
    mse: number;
    mae: number;
    mape: number;
    adjusted_r2: number;
    explained_variance: number;
}

interface PerformanceSummary {
    model_quality: {
        explained_variance: number;
        model_fit: string;
        overfitting_risk: string;
    };
    error_analysis: {
        average_error: number;
        prediction_spread: number;
        relative_error_pct: number;
    };
}

interface DiagnosticPlots {
    residual_plot: string;
    qq_plot: string;
    predicted_vs_actual: string;
    residual_distribution: string;
}

interface ResidualStats {
    mean: number;
    std: number;
    min: number;
    max: number;
    median: number;
}

interface RegressionStatsData {
    metrics: RegressionMetrics;
    data_source: string;
    diagnostic_plots: DiagnosticPlots;
    performance_summary: PerformanceSummary;
    residual_stats: ResidualStats;
}

const RegressionStats: React.FC = () => {
    const [data, setData] = useState<RegressionStatsData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [showAIExplanation, setShowAIExplanation] = useState(false);

    useEffect(() => {
        fetchRegressionStats();
    }, []);

    const fetchRegressionStats = async () => {
        try {
            setLoading(true);
            const response = await getRegressionStats();
            setData(response);
            setError(null);
        } catch (err) {
            setError('Failed to load regression statistics');
            console.error('Error fetching regression stats:', err);
        } finally {
            setLoading(false);
        }
    };

    const formatPercentage = (value: number): string =>
        (value && !isNaN(value)) ? `${(value * 100).toFixed(1)}%` : 'N/A';
    const formatNumber = (value: number, decimals: number = 3): string =>
        (value && !isNaN(value)) ? value.toFixed(decimals) : 'N/A';

    const getFitColor = (fit: string): string => {
        switch (fit.toLowerCase()) {
            case 'excellent': return 'text-green-600';
            case 'good': return 'text-blue-600';
            case 'moderate': return 'text-yellow-600';
            default: return 'text-red-600';
        }
    }; if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            </div>
        );
    }

    if (error) {
        return (
            <Card>
                <div className="text-red-600 text-center">
                    <p>{error}</p>
                    <button
                        onClick={fetchRegressionStats}
                        className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                    >
                        Retry
                    </button>
                </div>
            </Card>
        );
    }

    if (!data) return null;

    // Add comprehensive null checks for nested data structures
    if (!data.metrics || !data.performance_summary || !data.performance_summary.model_quality || !data.performance_summary.error_analysis) {
        return (
            <Card>
                <div className="text-red-600 text-center">
                    <p>Invalid data structure received from API</p>
                    <button
                        onClick={fetchRegressionStats}
                        className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                    >
                        Retry
                    </button>
                </div>
            </Card>
        );
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-gray-900">Regression Analysis</h1>
                    <p className="text-gray-600 mt-2">
                        Comprehensive regression model performance analysis and diagnostics
                    </p>
                </div>
                <ExplainWithAIButton
                    onClick={() => setShowAIExplanation(true)}
                />
            </div>

            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <MetricCard
                    title="R² Score"
                    value={data.metrics.r2_score}
                    format="percentage"
                />
                <MetricCard
                    title="RMSE"
                    value={data.metrics.rmse}
                    format="number"
                />
                <MetricCard
                    title="MAE"
                    value={data.metrics.mae}
                    format="number"
                />
                <MetricCard
                    title="MAPE"
                    value={data.metrics.mape ? data.metrics.mape / 100 : undefined}
                    format="percentage"
                />
            </div>            {/* Performance Summary */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                    <h3 className="text-lg font-semibold mb-4">Model Quality</h3>
                    <div className="space-y-3">
                        <div className="flex justify-between">
                            <span className="text-gray-600">Explained Variance:</span>
                            <span className="font-medium">{formatPercentage(data.performance_summary.model_quality.explained_variance)}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-600">Model Fit:</span>
                            <span className={`font-medium ${getFitColor(data.performance_summary.model_quality.model_fit || '')}`}>
                                {data.performance_summary.model_quality.model_fit || 'N/A'}
                            </span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-600">Overfitting Risk:</span>
                            <span className={`font-medium ${data.performance_summary.model_quality.overfitting_risk === 'Low' ? 'text-green-600' :
                                data.performance_summary.model_quality.overfitting_risk === 'Medium' ? 'text-yellow-600' : 'text-red-600'
                                }`}>
                                {data.performance_summary.model_quality.overfitting_risk || 'N/A'}
                            </span>
                        </div>
                    </div>
                </Card>

                <Card>
                    <h3 className="text-lg font-semibold mb-4">Error Analysis</h3>
                    <div className="space-y-3">
                        <div className="flex justify-between">
                            <span className="text-gray-600">Average Error:</span>
                            <span className="font-medium">{formatNumber(data.performance_summary.error_analysis.average_error)}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-600">Prediction Spread:</span>
                            <span className="font-medium">{formatNumber(data.performance_summary.error_analysis.prediction_spread)}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-600">Relative Error:</span>
                            <span className="font-medium">{formatPercentage(data.performance_summary.error_analysis.relative_error_pct / 100)}</span>
                        </div>
                    </div>
                </Card>
            </div>

            {/* Detailed Metrics */}
            <Card>
                <h3 className="text-lg font-semibold mb-4">Detailed Regression Metrics</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    <div className="text-center p-4 bg-gray-50 rounded-lg">
                        <div className="text-2xl font-bold text-blue-600">{formatNumber(data.metrics.mse)}</div>
                        <div className="text-sm text-gray-600">Mean Squared Error</div>
                    </div>
                    <div className="text-center p-4 bg-gray-50 rounded-lg">
                        <div className="text-2xl font-bold text-green-600">{formatPercentage(data.metrics.adjusted_r2)}</div>
                        <div className="text-sm text-gray-600">Adjusted R²</div>
                    </div>
                    <div className="text-center p-4 bg-gray-50 rounded-lg">
                        <div className="text-2xl font-bold text-purple-600">{formatPercentage(data.metrics.explained_variance)}</div>
                        <div className="text-sm text-gray-600">Explained Variance</div>
                    </div>
                </div>
            </Card>

            {/* Residual Statistics */}
            {data.residual_stats ? (
                <Card>
                    <h3 className="text-lg font-semibold mb-4">Residual Statistics</h3>
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                        <div className="text-center">
                            <div className="text-lg font-bold text-gray-900">{formatNumber(data.residual_stats.mean)}</div>
                            <div className="text-sm text-gray-600">Mean</div>
                        </div>
                        <div className="text-center">
                            <div className="text-lg font-bold text-gray-900">{formatNumber(data.residual_stats.std)}</div>
                            <div className="text-sm text-gray-600">Std Dev</div>
                        </div>
                        <div className="text-center">
                            <div className="text-lg font-bold text-gray-900">{formatNumber(data.residual_stats.min)}</div>
                            <div className="text-sm text-gray-600">Min</div>
                        </div>
                        <div className="text-center">
                            <div className="text-lg font-bold text-gray-900">{formatNumber(data.residual_stats.max)}</div>
                            <div className="text-sm text-gray-600">Max</div>
                        </div>
                        <div className="text-center">
                            <div className="text-lg font-bold text-gray-900">{formatNumber(data.residual_stats.median)}</div>
                            <div className="text-sm text-gray-600">Median</div>
                        </div>
                    </div>
                </Card>
            ) : null}      {/* Diagnostic Plots */}
            <Card>
                <h3 className="text-lg font-semibold mb-4">Diagnostic Plots</h3>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div>
                        <h4 className="text-md font-medium mb-2">Residuals vs Fitted Values</h4>
                        {data.diagnostic_plots?.residual_plot ? (
                            <img
                                src={`data:image/png;base64,${data.diagnostic_plots.residual_plot}`}
                                alt="Residual Plot"
                                className="w-full rounded-lg border"
                                onError={(e) => {
                                    const target = e.target as HTMLImageElement;
                                    target.style.display = 'none';
                                    target.nextElementSibling!.classList.remove('hidden');
                                }}
                            />
                        ) : null}
                        <div className="hidden p-8 bg-gray-100 rounded-lg text-center text-gray-500">
                            Plot unavailable
                        </div>
                    </div>
                    <div>
                        <h4 className="text-md font-medium mb-2">Q-Q Plot (Normality Check)</h4>
                        {data.diagnostic_plots?.qq_plot ? (
                            <img
                                src={`data:image/png;base64,${data.diagnostic_plots.qq_plot}`}
                                alt="Q-Q Plot"
                                className="w-full rounded-lg border"
                                onError={(e) => {
                                    const target = e.target as HTMLImageElement;
                                    target.style.display = 'none';
                                    target.nextElementSibling!.classList.remove('hidden');
                                }}
                            />
                        ) : null}
                        <div className="hidden p-8 bg-gray-100 rounded-lg text-center text-gray-500">
                            Plot unavailable
                        </div>
                    </div>
                    <div>
                        <h4 className="text-md font-medium mb-2">Predicted vs Actual Values</h4>
                        {data.diagnostic_plots?.predicted_vs_actual ? (
                            <img
                                src={`data:image/png;base64,${data.diagnostic_plots.predicted_vs_actual}`}
                                alt="Predicted vs Actual"
                                className="w-full rounded-lg border"
                                onError={(e) => {
                                    const target = e.target as HTMLImageElement;
                                    target.style.display = 'none';
                                    target.nextElementSibling!.classList.remove('hidden');
                                }}
                            />
                        ) : null}
                        <div className="hidden p-8 bg-gray-100 rounded-lg text-center text-gray-500">
                            Plot unavailable
                        </div>
                    </div>
                    <div>
                        <h4 className="text-md font-medium mb-2">Residual Distribution</h4>
                        {data.diagnostic_plots?.residual_distribution ? (
                            <img
                                src={`data:image/png;base64,${data.diagnostic_plots.residual_distribution}`}
                                alt="Residual Distribution"
                                className="w-full rounded-lg border"
                                onError={(e) => {
                                    const target = e.target as HTMLImageElement;
                                    target.style.display = 'none';
                                    target.nextElementSibling!.classList.remove('hidden');
                                }}
                            />
                        ) : null}
                        <div className="hidden p-8 bg-gray-100 rounded-lg text-center text-gray-500">
                            Plot unavailable
                        </div>
                    </div>
                </div>
            </Card>            {/* Data Source Info */}
            <Card>
                <div className="text-sm text-gray-600">
                    <p>Analysis performed on <span className="font-medium">{data.data_source}</span> dataset</p>
                </div>
            </Card>

            {/* AI Explanation Panel */}
            <AIExplanationPanel
                isOpen={showAIExplanation}
                onClose={() => setShowAIExplanation(false)}
                analysisType="regression_stats"
                analysisData={data}
            />
        </div>
    );
};

export default RegressionStats;
