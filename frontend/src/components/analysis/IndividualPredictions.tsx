import React, { useEffect, useMemo, useState } from 'react';
import { AlertCircle, Loader2, User, Target, TrendingUp, TrendingDown, Info, Filter } from 'lucide-react';
import ExplainWithAIButton from '../common/ExplainWithAIButton';
import AIExplanationPanel from '../common/AIExplanationPanel';
import { useS3Config } from '../../context/S3ConfigContext';
import { postIndividualPrediction, postExplainInstance, postListInstances } from '../../services/api.stateless';
   
const FeatureContribution = ({ name, value, contribution, maxContribution }: {
    name: string;
    value: any;
    contribution: number;
    maxContribution: number;
}) => {
    const widthPercentage = Math.abs(contribution / maxContribution) * 100;
    const isPositive = contribution > 0;

    return (
        <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg mb-3">
            <div className="flex justify-between items-center mb-2">
                <div className="flex-1">
                    <span className="font-medium text-sm">{name}</span>
                    <span className="text-xs text-gray-500 ml-2">= {value}</span>
                </div>
                <div className="flex items-center space-x-2">
                    {isPositive ? (
                        <TrendingUp className="w-4 h-4 text-green-500" />
                    ) : (
                        <TrendingDown className="w-4 h-4 text-red-500" />
                    )}
                    <span className="text-sm font-mono">
                        {contribution > 0 ? '+' : ''}{contribution.toFixed(3)}
                    </span>
                </div>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-1.5">
                <div
                    className={`h-1.5 rounded-full ${isPositive ? 'bg-green-500' : 'bg-red-500'
                        }`}
                    style={{ width: `${Math.max(widthPercentage, 3)}%` }}
                ></div>
            </div>
        </div>
    );
};

const PredictionCard = ({ prediction, baseValue, actual }: {
    prediction: number;
    baseValue: number;
    actual: number;
}) => {
    const error = Math.abs(prediction - actual);
    const errorPercentage = Math.abs((prediction - actual) / actual) * 100;

    return (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold flex items-center">
                    <Target className="mr-2 text-blue-600" />
                    Prediction Summary
                </h3>
                <div className="text-right">
                    <div className="text-2xl font-bold text-blue-600">
                        {prediction.toFixed(3)}
                    </div>
                    <div className="text-sm text-gray-500">Predicted Value</div>
                </div>
            </div>
            <div className="grid grid-cols-4 gap-4 text-center mt-2">
                <div>
                    <div className="text-lg font-bold text-purple-600">{baseValue.toFixed(3)}</div>
                    <div className="text-xs text-gray-500">Base Value</div>
                </div>
                <div>
                    <div className="text-lg font-bold text-gray-700 dark:text-gray-300">{actual.toFixed(3)}</div>
                    <div className="text-xs text-gray-500">Actual Value</div>
                </div>
                <div>
                    <div className="text-lg font-bold text-red-600">{error.toFixed(3)}</div>
                    <div className="text-xs text-gray-500">Absolute Error</div>
                </div>
                <div>
                    <div className="text-lg font-bold text-orange-600">{errorPercentage.toFixed(1)}%</div>
                    <div className="text-xs text-gray-500">Error %</div>
                </div>
            </div>
        </div>
    );
}

const IndividualPredictions: React.FC<{ modelType?: string }> = () => {
    const [explanation, setExplanation] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [instanceIdx, setInstanceIdx] = useState<number | null>(null);
    const [instances, setInstances] = useState<Array<{ id: number; prediction: number; actual: number }>>([]);
    const [sortBy, setSortBy] = useState<'prediction' | 'confidence'>('prediction');
    // AI explanation state
    const [showAIExplanation, setShowAIExplanation] = useState(false);
    // S3 config context
    const { config } = useS3Config();

    const fetchExplanation = async (idx?: number) => {
        try {
            setLoading(true);
            setError('');
            const targetIdx = typeof idx === 'number' ? idx : (instanceIdx ?? 0);
            // Use stateless API: send S3 URLs, target_column, instance_idx
            const payload = {
                model: config.modelUrl,
                train_dataset: config.trainDatasetUrl,
                test_dataset: config.testDatasetUrl,
                target_column: config.targetColumn,
                instance_idx: targetIdx,
            };
            const [summary, legacy] = await Promise.all([
                postIndividualPrediction(payload),
                postExplainInstance(payload)
            ]);
            const merged = {
                ...legacy,
                prediction: summary.prediction_value || summary.prediction_percentage / 100,
                base_value: summary.base_value,
                actual_value: summary.actual_outcome || summary.actual_value
            };
            setExplanation(merged);
            setInstanceIdx(targetIdx);
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Failed to fetch instance explanation.');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        (async () => {
            try {
                // Use stateless API: send S3 URLs, target_column, sort_by, limit
                const payload = {
                    model: config.modelUrl,
                    train_dataset: config.trainDatasetUrl,
                    test_dataset: config.testDatasetUrl,
                    target_column: config.targetColumn,
                    sort_by: sortBy,
                    limit: 100,
                };
                const data = await postListInstances(payload);
                setInstances(data.instances || []);
                if ((data.instances || []).length > 0 && instanceIdx === null) {
                    fetchExplanation(data.instances[0].id);
                }
            } catch (e) { /* ignore */ }
        })();
    }, [sortBy, config]);

    const ordered = useMemo(() => explanation?.ordered_contributions, [explanation]);

    return (
        <>
            <div className="min-h-screen bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 w-full">
                <div className="p-0 sm:p-2 md:p-4 lg:p-6 max-w-none">
                    <div className="space-y-6">
                        <div className="flex items-center justify-between">
                            <h1 className="text-3xl font-bold flex items-center">
                                <User className="mr-3 text-blue-600" />
                                Individual Prediction Analysis
                            </h1>
                            <div className="flex items-center space-x-4">
                                <div className="flex items-center space-x-2 text-sm text-gray-500">
                                    <Info className="w-4 h-4" />
                                    <span>Explain specific instances</span>
                                </div>
                                <ExplainWithAIButton onClick={() => setShowAIExplanation(true)} size="md" />
                            </div>
                        </div>

                        {/* Layout with left selector */}
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                            {/* Selector */}
                            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                                <div className="flex items-center justify-between mb-3">
                                    <h2 className="text-lg font-semibold">Select Instance</h2>
                                    <div className="flex items-center space-x-2">
                                        <Filter className="w-4 h-4 text-gray-400" />
                                        <button className={`text-xs px-2 py-1 rounded ${sortBy === 'prediction' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100'}`} onClick={() => setSortBy('prediction')}>By Prediction</button>
                                        <button className={`text-xs px-2 py-1 rounded ${sortBy === 'confidence' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100'}`} onClick={() => setSortBy('confidence')}>By Confidence</button>
                                    </div>
                                </div>
                                <div className="space-y-2 max-h-[28rem] overflow-y-auto">
                                    {instances.map((it) => (
                                        <div key={it.id} onClick={() => fetchExplanation(it.id)} className={`p-3 rounded border cursor-pointer ${instanceIdx === it.id ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'border-gray-200 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700'}`}>
                                            <div className="flex justify-between">
                                                <div>
                                                    <div className="text-sm font-medium">Instance {it.id}</div>
                                                    <div className="text-xs text-gray-500">Actual: {it.actual}</div>
                                                </div>
                                                <div className="text-sm font-semibold">{(it.prediction * 100).toFixed(1)}%</div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Main content */}
                            <div className="lg:col-span-2 space-y-6">
                                {error && (
                                    <div className="p-4 text-red-500 bg-red-50 dark:bg-red-900/20 rounded-lg flex items-center">
                                        <AlertCircle className="mr-2 flex-shrink-0" />
                                        <span>{error}</span>
                                    </div>
                                )}

                                {loading && (
                                    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 h-32 flex items-center justify-center">
                                        <Loader2 className="w-6 h-6 animate-spin" />
                                    </div>
                                )}

                                {explanation && !loading && (
                                    <div className="space-y-6">
                                        <PredictionCard prediction={explanation.prediction} baseValue={explanation.base_value} actual={explanation.actual_value} />

                                        {/* SHAP Waterfall */}
                                        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                                            <h3 className="text-lg font-semibold mb-4">SHAP Waterfall Chart</h3>
                                            {ordered ? (
                                                <div className="max-h-96 overflow-y-auto">
                                                    {ordered.feature_names.map((name: string, i: number) => {
                                                        const value = ordered.shap_values[i];
                                                        const isPositive = value >= 0;
                                                        const width = Math.min(100, Math.abs(value) / Math.max(...ordered.shap_values.map((v: number) => Math.abs(v))) * 100);
                                                        return (
                                                            <div key={i} className="flex items-center mb-2">
                                                                <div className="w-48 text-xs truncate pr-2">{name}</div>
                                                                <div className="flex-1 bg-gray-200 dark:bg-gray-700 h-3 rounded">
                                                                    <div className={`h-3 rounded ${isPositive ? 'bg-green-500' : 'bg-red-500'}`} style={{ width: `${Math.max(3, width)}%` }}></div>
                                                                </div>
                                                                <div className="w-20 text-right text-xs font-mono pl-2">{value.toFixed(3)}</div>
                                                            </div>
                                                        );
                                                    })}
                                                </div>
                                            ) : (
                                                <div className="text-sm text-gray-500">No SHAP data</div>
                                            )}
                                        </div>

                                        {/* Feature Contributions */}
                                        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                                            <h3 className="text-lg font-semibold mb-4">Feature Contributions</h3>
                                            {ordered ? (
                                                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                                                    {ordered.feature_names.map((name: string, index: number) => {
                                                        const contribution = ordered.shap_values[index];
                                                        const featureValue = ordered.feature_values[index];
                                                        const maxContribution = Math.max(...ordered.shap_values.map((v: number) => Math.abs(v)));
                                                        return (
                                                            <FeatureContribution
                                                                key={index}
                                                                name={name}
                                                                value={featureValue}
                                                                contribution={contribution}
                                                                maxContribution={maxContribution}
                                                            />
                                                        );
                                                    })}
                                                </div>
                                            ) : (
                                                <div className="text-center py-8 text-gray-500">No feature contribution data available</div>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>

                        {!explanation && !loading && !error && (
                            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-8 text-center">
                                <User className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                                <h3 className="text-lg font-medium text-gray-600 dark:text-gray-400 mb-2">No Analysis Yet</h3>
                                <p className="text-sm text-gray-500">Select an instance on the left to see the explanation.</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
            {/* AI Explanation Panel */}
            <AIExplanationPanel
                isOpen={showAIExplanation}
                onClose={() => setShowAIExplanation(false)}
                analysisType="individual_prediction"
                analysisData={{
                    instance_idx: instanceIdx,
                    instances,
                    prediction: explanation?.prediction,
                    base_value: explanation?.base_value,
                    actual_value: explanation?.actual_value,
                    ordered_contributions: ordered ? {
                        feature_names: ordered.feature_names,
                        shap_values: ordered.shap_values,
                        feature_values: ordered.feature_values
                    } : null
                }}
                title="Individual Prediction - AI Explanation"
            />
        </>
    );
};

export default IndividualPredictions;
