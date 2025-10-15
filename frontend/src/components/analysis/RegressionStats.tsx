import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { Target, TrendingUp, Activity, AlertCircle, Loader2, BarChart3 } from 'lucide-react';
import Card from '../common/Card';
import AIExplanationPanel from '../common/AIExplanationPanel';
import ExplainWithAIButton from '../common/ExplainWithAIButton';
import { postRegressionStats } from '../../services/api.stateless';
import { useS3Config } from '../../context/S3ConfigContext';

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
    residual_plot: {
        fitted_values: number[];
        residuals: number[];
        trend_line: {
            x: number[];
            y: number[];
            slope: number;
            intercept: number;
        };
    };
    qq_plot: {
        theoretical_quantiles: number[];
        ordered_residuals: number[];
        slope: number;
        intercept: number;
        correlation: number;
    };
    predicted_vs_actual: {
        actual: number[];
        predicted: number[];
        perfect_line: {
            x: number[];
            y: number[];
        };
    };
    residual_distribution: {
        histogram: {
            bin_edges: number[];
            counts: number[];
        };
        normal_curve: {
            x: number[];
            y: number[];
            mu: number;
            sigma: number;
        };
    };
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
    const { config } = useS3Config();

    useEffect(() => {
        fetchRegressionStats();
    }, [config]);

    const fetchRegressionStats = async () => {
        try {
            setLoading(true);
            const payload = {
                model: config.modelUrl,
                train_dataset: config.trainDatasetUrl,
                test_dataset: config.testDatasetUrl,
                target_column: config.targetColumn,
            };
            const response = await postRegressionStats(payload);
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
    };

    // Memoize Plotly chart data for performance
    const plotlyData = React.useMemo(() => {
        if (!data?.diagnostic_plots) return null;

        const plots = data.diagnostic_plots;
        
        return {
            residualPlot: {
                data: [
                    {
                        x: plots.residual_plot.fitted_values,
                        y: plots.residual_plot.residuals,
                        mode: 'markers' as const,
                        type: 'scatter' as const,
                        name: 'Residuals',
                        marker: { color: '#3b82f6', size: 6, opacity: 0.7 }
                    },
                    {
                        x: plots.residual_plot.trend_line.x,
                        y: plots.residual_plot.trend_line.y,
                        mode: 'lines' as const,
                        type: 'scatter' as const,
                        name: 'Trend Line',
                        line: { color: '#ef4444', width: 2 }
                    }
                ],
                layout: {
                    title: { text: 'Residuals vs Fitted Values' },
                    xaxis: { title: { text: 'Fitted Values' } },
                    yaxis: { title: { text: 'Residuals' } },
                    showlegend: true,
                    height: 350,
                    margin: { l: 50, r: 50, t: 50, b: 50 }
                }
            },
            qqPlot: {
                data: [
                    {
                        x: plots.qq_plot.theoretical_quantiles,
                        y: plots.qq_plot.ordered_residuals,
                        mode: 'markers' as const,
                        type: 'scatter' as const,
                        name: 'Q-Q Points',
                        marker: { color: '#10b981', size: 6, opacity: 0.7 }
                    },
                    {
                        x: [Math.min(...plots.qq_plot.theoretical_quantiles), Math.max(...plots.qq_plot.theoretical_quantiles)],
                        y: [
                            plots.qq_plot.slope * Math.min(...plots.qq_plot.theoretical_quantiles) + plots.qq_plot.intercept,
                            plots.qq_plot.slope * Math.max(...plots.qq_plot.theoretical_quantiles) + plots.qq_plot.intercept
                        ],
                        mode: 'lines' as const,
                        type: 'scatter' as const,
                        name: 'Normal Line',
                        line: { color: '#f59e0b', width: 2 }
                    }
                ],
                layout: {
                    title: { text: 'Q-Q Plot (Normality Check)' },
                    xaxis: { title: { text: 'Theoretical Quantiles' } },
                    yaxis: { title: { text: 'Ordered Residuals' } },
                    showlegend: true,
                    height: 350,
                    margin: { l: 50, r: 50, t: 50, b: 50 }
                }
            },
            predictedVsActual: {
                data: [
                    {
                        x: plots.predicted_vs_actual.actual,
                        y: plots.predicted_vs_actual.predicted,
                        mode: 'markers' as const,
                        type: 'scatter' as const,
                        name: 'Predictions',
                        marker: { color: '#8b5cf6', size: 6, opacity: 0.7 }
                    },
                    {
                        x: plots.predicted_vs_actual.perfect_line.x,
                        y: plots.predicted_vs_actual.perfect_line.y,
                        mode: 'lines' as const,
                        type: 'scatter' as const,
                        name: 'Perfect Line',
                        line: { color: '#dc2626', width: 2 }
                    }
                ],
                layout: {
                    title: { text: 'Predicted vs Actual Values' },
                    xaxis: { title: { text: 'Actual Values' } },
                    yaxis: { title: { text: 'Predicted Values' } },
                    showlegend: true,
                    height: 350,
                    margin: { l: 50, r: 50, t: 50, b: 50 }
                }
            },
            residualDistribution: {
                data: [
                    {
                        x: plots.residual_distribution.histogram.bin_edges.slice(0, -1),
                        y: plots.residual_distribution.histogram.counts,
                        type: 'bar' as const,
                        name: 'Histogram',
                        marker: { color: '#f97316', opacity: 0.8 }
                    },
                    {
                        x: plots.residual_distribution.normal_curve.x,
                        y: plots.residual_distribution.normal_curve.y,
                        mode: 'lines' as const,
                        type: 'scatter' as const,
                        name: 'Normal Curve',
                        line: { color: '#059669', width: 2 },
                        yaxis: 'y2'
                    }
                ],
                layout: {
                    title: { text: 'Residual Distribution' },
                    xaxis: { title: { text: 'Residuals' } },
                    yaxis: { title: { text: 'Density (Histogram)' } },
                    yaxis2: { 
                        title: { text: 'Normal Curve' }, 
                        overlaying: 'y' as const, 
                        side: 'right' as const,
                        showgrid: false
                    },
                    showlegend: true,
                    height: 350,
                    margin: { l: 50, r: 50, t: 50, b: 50 }
                }
            }
        };
    }, [data?.diagnostic_plots]); 
    
    if (loading) {
        return (
            <div className="p-6 flex justify-center items-center h-full">
                <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
            </div>
        );
    }

    if (error) {
        return (
            <div className="p-6 bg-gray-50 dark:bg-gray-800">
                <Card>
                    <div className="text-red-600 text-center">
                        <AlertCircle className="w-6 h-6 mx-auto mb-2" />
                        <p>{error}</p>
                        <button
                            onClick={fetchRegressionStats}
                            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                        >
                            Retry
                        </button>
                    </div>
                </Card>
            </div>
        );
    }

    if (!data) return null;

    // Add comprehensive null checks for nested data structures
    if (!data.metrics || !data.performance_summary || !data.performance_summary.model_quality || !data.performance_summary.error_analysis) {
        return (
            <div className="p-6 bg-gray-50 dark:bg-gray-800">
                <Card>
                    <div className="text-red-600 text-center">
                        <AlertCircle className="w-6 h-6 mx-auto mb-2" />
                        <p>Invalid data structure received from API</p>
                        <button
                            onClick={fetchRegressionStats}
                            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                        >
                            Retry
                        </button>
                    </div>
                </Card>
            </div>
        );
    }

    return (
        <div className="p-6 space-y-6 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100">
            {/* Header */}
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Regression Analysis</h1>
                    <p className="text-gray-600 dark:text-gray-400 mt-2">
                        Comprehensive regression model performance analysis and diagnostics
                    </p>
                </div>
                <ExplainWithAIButton
                    onClick={() => setShowAIExplanation(true)}
                />
            </div>

            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="p-4 bg-white dark:bg-gray-800 rounded-lg shadow">
                    <div className="flex items-center">
                        <div className="p-3 bg-blue-100 dark:bg-blue-900/50 rounded-full">
                            <Target className="w-6 h-6 text-blue-600" />
                        </div>
                        <div className="ml-4">
                            <p className="text-sm text-gray-500 dark:text-gray-400">R² Score</p>
                            <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                {formatPercentage(data.metrics.r2_score)}
                            </p>
                        </div>
                    </div>
                </div>
                <div className="p-4 bg-white dark:bg-gray-800 rounded-lg shadow">
                    <div className="flex items-center">
                        <div className="p-3 bg-red-100 dark:bg-red-900/50 rounded-full">
                            <TrendingUp className="w-6 h-6 text-red-600" />
                        </div>
                        <div className="ml-4">
                            <p className="text-sm text-gray-500 dark:text-gray-400">RMSE</p>
                            <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                {formatNumber(data.metrics.rmse)}
                            </p>
                        </div>
                    </div>
                </div>
                <div className="p-4 bg-white dark:bg-gray-800 rounded-lg shadow">
                    <div className="flex items-center">
                        <div className="p-3 bg-orange-100 dark:bg-orange-900/50 rounded-full">
                            <Activity className="w-6 h-6 text-orange-600" />
                        </div>
                        <div className="ml-4">
                            <p className="text-sm text-gray-500 dark:text-gray-400">MAE</p>
                            <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                {formatNumber(data.metrics.mae)}
                            </p>
                        </div>
                    </div>
                </div>
                <div className="p-4 bg-white dark:bg-gray-800 rounded-lg shadow">
                    <div className="flex items-center">
                        <div className="p-3 bg-purple-100 dark:bg-purple-900/50 rounded-full">
                            <BarChart3 className="w-6 h-6 text-purple-600" />
                        </div>
                        <div className="ml-4">
                            <p className="text-sm text-gray-500 dark:text-gray-400">MAPE</p>
                            <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                {data.metrics.mape ? formatPercentage(data.metrics.mape / 100) : 'N/A'}
                            </p>
                        </div>
                    </div>
                </div>
            </div>            {/* Performance Summary */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow">
                    <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Model Quality</h3>
                    <div className="space-y-3">
                        <div className="flex justify-between">
                            <span className="text-gray-600 dark:text-gray-400">Explained Variance:</span>
                            <span className="font-medium text-gray-900 dark:text-white">{formatPercentage(data.performance_summary.model_quality.explained_variance)}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-600 dark:text-gray-400">Model Fit:</span>
                            <span className={`font-medium ${getFitColor(data.performance_summary.model_quality.model_fit || '')}`}>
                                {data.performance_summary.model_quality.model_fit || 'N/A'}
                            </span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-600 dark:text-gray-400">Overfitting Risk:</span>
                            <span className={`font-medium ${data.performance_summary.model_quality.overfitting_risk === 'Low' ? 'text-green-600' :
                                data.performance_summary.model_quality.overfitting_risk === 'Medium' ? 'text-yellow-600' : 'text-red-600'
                                }`}>
                                {data.performance_summary.model_quality.overfitting_risk || 'N/A'}
                            </span>
                        </div>
                    </div>
                </div>

                <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow">
                    <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Error Analysis</h3>
                    <div className="space-y-3">
                        <div className="flex justify-between">
                            <span className="text-gray-600 dark:text-gray-400">Average Error:</span>
                            <span className="font-medium text-gray-900 dark:text-white">{formatNumber(data.performance_summary.error_analysis.average_error)}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-600 dark:text-gray-400">Prediction Spread:</span>
                            <span className="font-medium text-gray-900 dark:text-white">{formatNumber(data.performance_summary.error_analysis.prediction_spread)}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-600 dark:text-gray-400">Relative Error:</span>
                            <span className="font-medium text-gray-900 dark:text-white">{formatPercentage(data.performance_summary.error_analysis.relative_error_pct / 100)}</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Detailed Metrics */}
            <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow">
                <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Detailed Regression Metrics</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                        <div className="text-2xl font-bold text-blue-600">{formatNumber(data.metrics.mse)}</div>
                        <div className="text-sm text-gray-600 dark:text-gray-400">Mean Squared Error</div>
                    </div>
                    <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                        <div className="text-2xl font-bold text-green-600">{formatPercentage(data.metrics.adjusted_r2)}</div>
                        <div className="text-sm text-gray-600 dark:text-gray-400">Adjusted R²</div>
                    </div>
                    <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                        <div className="text-2xl font-bold text-purple-600">{formatPercentage(data.metrics.explained_variance)}</div>
                        <div className="text-sm text-gray-600 dark:text-gray-400">Explained Variance</div>
                    </div>
                </div>
            </div>

            {/* Residual Statistics */}
            {data.residual_stats ? (
                <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow">
                    <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Residual Statistics</h3>
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                        <div className="text-center">
                            <div className="text-lg font-bold text-gray-900 dark:text-white">{formatNumber(data.residual_stats.mean)}</div>
                            <div className="text-sm text-gray-600 dark:text-gray-400">Mean</div>
                        </div>
                        <div className="text-center">
                            <div className="text-lg font-bold text-gray-900 dark:text-white">{formatNumber(data.residual_stats.std)}</div>
                            <div className="text-sm text-gray-600 dark:text-gray-400">Std Dev</div>
                        </div>
                        <div className="text-center">
                            <div className="text-lg font-bold text-gray-900 dark:text-white">{formatNumber(data.residual_stats.min)}</div>
                            <div className="text-sm text-gray-600 dark:text-gray-400">Min</div>
                        </div>
                        <div className="text-center">
                            <div className="text-lg font-bold text-gray-900 dark:text-white">{formatNumber(data.residual_stats.max)}</div>
                            <div className="text-sm text-gray-600 dark:text-gray-400">Max</div>
                        </div>
                        <div className="text-center">
                            <div className="text-lg font-bold text-gray-900 dark:text-white">{formatNumber(data.residual_stats.median)}</div>
                            <div className="text-sm text-gray-600 dark:text-gray-400">Median</div>
                        </div>
                    </div>
                </div>
            ) : null}

            {/* Diagnostic Plots */}
            {plotlyData && (
                <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow">
                    <h3 className="text-lg font-semibold mb-6 text-gray-900 dark:text-white flex items-center">
                        <BarChart3 className="w-5 h-5 mr-2 text-blue-600" />
                        Diagnostic Plots
                    </h3>
                    <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                        {/* Residuals vs Fitted Values */}
                        <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
                            <h4 className="text-md font-medium mb-3 text-gray-800 dark:text-gray-200 flex items-center">
                                <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
                                Residuals vs Fitted Values
                            </h4>
                            <div className="bg-white dark:bg-gray-800 p-2 rounded">
                                <Plot
                                    data={plotlyData.residualPlot.data}
                                    layout={plotlyData.residualPlot.layout}
                                    config={{ responsive: true, displayModeBar: false }}
                                    style={{ width: '100%', height: '100%' }}
                                />
                            </div>
                        </div>

                        {/* Q-Q Plot */}
                        <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
                            <h4 className="text-md font-medium mb-3 text-gray-800 dark:text-gray-200 flex items-center">
                                <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
                                Q-Q Plot (Normality Check)
                            </h4>
                            <div className="bg-white dark:bg-gray-800 p-2 rounded">
                                <Plot
                                    data={plotlyData.qqPlot.data}
                                    layout={plotlyData.qqPlot.layout}
                                    config={{ responsive: true, displayModeBar: false }}
                                    style={{ width: '100%', height: '100%' }}
                                />
                            </div>
                        </div>

                        {/* Predicted vs Actual Values */}
                        <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
                            <h4 className="text-md font-medium mb-3 text-gray-800 dark:text-gray-200 flex items-center">
                                <div className="w-3 h-3 bg-purple-500 rounded-full mr-2"></div>
                                Predicted vs Actual Values
                            </h4>
                            <div className="bg-white dark:bg-gray-800 p-2 rounded">
                                <Plot
                                    data={plotlyData.predictedVsActual.data}
                                    layout={plotlyData.predictedVsActual.layout}
                                    config={{ responsive: true, displayModeBar: false }}
                                    style={{ width: '100%', height: '100%' }}
                                />
                            </div>
                        </div>

                        {/* Residual Distribution */}
                        <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
                            <h4 className="text-md font-medium mb-3 text-gray-800 dark:text-gray-200 flex items-center">
                                <div className="w-3 h-3 bg-orange-500 rounded-full mr-2"></div>
                                Residual Distribution
                            </h4>
                            <div className="bg-white dark:bg-gray-800 p-2 rounded">
                                <Plot
                                    data={plotlyData.residualDistribution.data}
                                    layout={plotlyData.residualDistribution.layout}
                                    config={{ responsive: true, displayModeBar: false }}
                                    style={{ width: '100%', height: '100%' }}
                                />
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Data Source Info */}
            <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow">
                <div className="text-sm text-gray-600 dark:text-gray-400">
                    <p>Analysis performed on <span className="font-medium text-gray-900 dark:text-white">{data.data_source}</span> dataset</p>
                </div>
            </div>

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
