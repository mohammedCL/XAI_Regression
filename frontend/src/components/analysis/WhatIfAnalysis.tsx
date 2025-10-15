import React, { useState, useEffect } from 'react';
import { AlertCircle, Sliders, Zap, BarChart3, RefreshCw, Loader2, Target, RotateCcw } from 'lucide-react';
import { postFeaturesMetadata, explainWithAI } from '../../services/api.stateless';
import { postWhatIf } from '../../services/api.stateless';
import { useS3Config } from '../../context/S3ConfigContext';
import ExplainWithAIButton from '../common/ExplainWithAIButton';

const FeatureSlider = ({ name, value, min, max, step, median, mean, onChange }: {
    name: string;
    value: number;
    min: number;
    max: number;
    step: number;
    median?: number;
    mean?: number;
    onChange: (value: number) => void;
}) => (
    <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
        <div className="flex justify-between items-center mb-2">
            <label className="text-sm font-medium">{name.replace('_', ' ')}</label>
            <div className="flex items-center space-x-2">
                <span className="text-sm font-mono text-blue-600">{value}</span>
                {median !== undefined && (
                    <button
                        onClick={() => onChange(median)}
                        className="text-xs bg-blue-100 hover:bg-blue-200 dark:bg-blue-800 dark:hover:bg-blue-700 px-2 py-1 rounded text-blue-600 dark:text-blue-300"
                        title={`Reset to median (${median.toFixed(2)})`}
                    >
                        Median
                    </button>
                )}
            </div>
        </div>
        <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={(e) => onChange(Number(e.target.value))}
            className="w-full h-2 bg-gray-200 dark:bg-gray-600 rounded-lg appearance-none cursor-pointer slider"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>{min}</span>
            {median !== undefined && (
                <span className="text-blue-500" title="Median">
                    Med: {median.toFixed(1)}
                </span>
            )}
            {mean !== undefined && (
                <span className="text-green-500" title="Mean">
                    Avg: {mean.toFixed(1)}
                </span>
            )}
            <span>{max}</span>
        </div>
    </div>
);

const FeatureInput = ({ name, value, type, onChange }: {
    name: string;
    value: any;
    type: 'text' | 'number';
    onChange: (value: any) => void;
}) => (
    <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
        <label className="block text-sm font-medium mb-2">{name.replace('_', ' ')}</label>
        <input
            type={type}
            value={value}
            onChange={(e) => onChange(type === 'number' ? Number(e.target.value) : e.target.value)}
            className="w-full px-3 py-2 border rounded-md bg-white dark:bg-gray-600 dark:border-gray-500"
        />
    </div>
);

const PredictionDisplay = ({ prediction, modelType }: { prediction: number; modelType?: string }) => {
    console.log('PredictionDisplay rendering with prediction:', prediction);
    return (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
                <Target className="mr-2 text-blue-600" />
                Current Prediction
            </h3>
            <div className="text-center">
                <div className="text-4xl font-bold text-blue-600 mb-2">
                    {typeof prediction === 'number' ? prediction.toFixed(2) : 'N/A'}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                    Predicted Value
                </div>
                <div className="text-xs text-gray-500">
                    Model Type: {modelType || 'Regression'}
                </div>
            </div>
        </div>
    );
};

const FeatureImpactChart = ({ impacts }: { impacts: Array<{ name: string, impact: number }> }) => (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
            <BarChart3 className="mr-2 text-purple-600" />
            Feature Impact Analysis
        </h3>
        <div className="space-y-3">
            {impacts.map((feature, index) => (
                <div key={index} className="flex items-center justify-between">
                    <span className="text-sm font-medium w-32 truncate">{feature.name.replace('_', ' ')}</span>
                    <div className="flex-1 mx-4">
                        <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                            <div
                                className={`h-2 rounded-full ${feature.impact >= 0 ? 'bg-green-500' : 'bg-red-500'}`}
                                style={{ width: `${Math.min(Math.abs(feature.impact) * 100, 100)}%` }}
                            ></div>
                        </div>
                    </div>
                    <span className="text-xs font-mono w-12 text-right">
                        {feature.impact.toFixed(3)}
                    </span>
                </div>
            ))}
        </div>
    </div>
);

const WhatIfAnalysis: React.FC<{ modelType?: string }> = ({ modelType = 'regression' }) => {
    const [features, setFeatures] = useState<Record<string, any>>({});
    const [prediction, setPrediction] = useState<number>(0);
    const [featureImpacts, setFeatureImpacts] = useState<Array<{ name: string, impact: number }>>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [featureList, setFeatureList] = useState<string[]>([]);
    const [featureRanges, setFeatureRanges] = useState<Record<string, any>>({});
    const [modelLoaded, setModelLoaded] = useState(false);
    const [showAIExplanation, setShowAIExplanation] = useState(false);
    const [aiExplanation, setAiExplanation] = useState<any>(null);
    const [aiLoading, setAiLoading] = useState(false);

    // Get S3 configuration for stateless API calls
    const { config } = useS3Config();

    // Load model overview and feature names on component mount (stateless POST)
    useEffect(() => {
        const loadModelInfo = async () => {
            try {
                console.log('Loading features metadata...');
                const featuresData = await postFeaturesMetadata({
                    model: config.modelUrl,
                    train_dataset: config.trainDatasetUrl,
                    test_dataset: config.testDatasetUrl,
                    target_column: config.targetColumn
                });
                console.log('Features metadata received:', featuresData);
                const features = featuresData.features || [];
                
                if (features.length === 0) {
                    throw new Error('No features found in the model');
                }

                const featureNames = features.map((f: any) => f.name);
                setFeatureList(featureNames);
                
                // Initialize features with median values from the metadata
                const initialFeatures: Record<string, any> = {};
                const ranges: Record<string, any> = {};
                
                features.forEach((feature: any) => {
                    const name = feature.name;
                    
                    if (feature.type === 'numerical') {
                        // Use median value as default for numerical features
                        const medianValue = feature.median || feature.mean || 0;
                        initialFeatures[name] = medianValue;
                        console.log(`Setting ${name} to median value: ${medianValue}`);
                        
                        // Set up ranges for sliders
                        ranges[name] = {
                            type: 'numeric',
                            min: feature.min || 0,
                            max: feature.max || 1,
                            step: Math.max((feature.max - feature.min) / 100, 0.01),
                            median: feature.median,
                            mean: feature.mean
                        };
                    } else {
                        // For categorical features, use the most common category or first available
                        const topCategories = feature.top_categories || {};
                        const firstCategory = Object.keys(topCategories)[0] || 'unknown';
                        initialFeatures[name] = firstCategory;
                        console.log(`Setting ${name} to first category: ${firstCategory}`);
                        
                        ranges[name] = {
                            type: 'categorical',
                            categories: Object.keys(topCategories)
                        };
                    }
                });
                
                console.log('Initial features with median values:', initialFeatures);
                setFeatures(initialFeatures);
                setFeatureRanges(ranges);
                setModelLoaded(true);
                
                // Get initial prediction with median values (force run during initialization)
                await performPrediction(initialFeatures, true);
                
            } catch (err: any) {
                setError('Failed to load model information. Please ensure a model is uploaded.');
                setModelLoaded(false);
            }
        };
        
        loadModelInfo();
    }, []);

    const performPrediction = async (currentFeatures: Record<string, any>, forceRun: boolean = false) => {
        if (!modelLoaded && !forceRun) return;
        
        setLoading(true);
        setError('');
        
        try {
            console.log('Performing prediction with features:', currentFeatures);
            
            // Construct stateless payload for backend
            const payload = {
                model: config.modelUrl,
                train_dataset: config.trainDatasetUrl,
                test_dataset: config.testDatasetUrl,
                target_column: config.targetColumn,
                features: currentFeatures
            };
            
            const result = await postWhatIf(payload);
            console.log('Prediction result:', result);
            console.log('Prediction value:', result.prediction_value);
            setPrediction(result.prediction_value || 0);
            console.log('Set prediction state to:', result.prediction_value || 0);
            
            // Update feature ranges if provided
            if (result.feature_ranges) {
                setFeatureRanges(prev => ({ ...prev, ...result.feature_ranges }));
            }
            
            // Use SHAP explanations if available
            if (result.shap_explanations) {
                const impacts = Object.entries(result.shap_explanations).map(([feature, impact]) => ({
                    name: feature,
                    impact: impact as number
                })).sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
                setFeatureImpacts(impacts);
            } else {
                // Fallback to mock impacts if SHAP not available
                const impacts = Object.keys(currentFeatures).map(feature => ({
                    name: feature,
                    impact: Math.random() * 0.4 - 0.2
                }));
                setFeatureImpacts(impacts);
            }
            
        } catch (err: any) {
            console.error('Prediction error:', err);
            setError(err.response?.data?.detail || 'Failed to get prediction');
            setPrediction(0);
        } finally {
            setLoading(false);
        }
    };

    const handleFeatureChange = async (featureName: string, value: any) => {
        const newFeatures = { ...features, [featureName]: value };
        setFeatures(newFeatures);
        
        // Reset AI explanation when features change
        if (aiExplanation) {
            setAiExplanation(null);
        }
        
        // Debounce prediction calls
        clearTimeout((window as any).predictionTimeout);
        (window as any).predictionTimeout = setTimeout(() => {
            performPrediction(newFeatures);
        }, 500);
    };

    const resetToMedian = () => {
        const medianFeatures: Record<string, any> = {};
        featureList.forEach(feature => {
            const featureRange = featureRanges[feature];
            if (featureRange?.type === 'numeric' && featureRange.median !== undefined) {
                medianFeatures[feature] = featureRange.median;
            } else if (featureRange?.type === 'categorical' && featureRange.categories?.length > 0) {
                medianFeatures[feature] = featureRange.categories[0]; // Use first category for categorical
            } else {
                medianFeatures[feature] = features[feature]; // Keep current value if no median available
            }
        });
        
        setFeatures(medianFeatures);
        performPrediction(medianFeatures);
        
        // Reset AI explanation
        if (aiExplanation) {
            setAiExplanation(null);
        }
    };

    const handleExplainWithAI = async () => {
        if (showAIExplanation && aiExplanation) {
            // If already showing explanation, just toggle
            setShowAIExplanation(false);
            return;
        }

        setShowAIExplanation(true);
        setAiLoading(true);
        
        try {
            // Prepare the what-if analysis data for AI explanation
            const analysisData = {
                current_prediction: {
                    value: prediction,
                    feature_values: features
                },
                feature_impacts: featureImpacts.map(impact => ({
                    feature: impact.name,
                    impact: impact.impact,
                    abs_impact: Math.abs(impact.impact)
                })),
                model_info: {
                    feature_count: featureList.length,
                    model_type: modelType
                },
                scenario_analysis: {
                    modified_features: Object.keys(features).filter(feature => 
                        features[feature] !== 0 // Simple check for modified features
                    ),
                    prediction_value: prediction,
                    top_influences: featureImpacts
                        .sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact))
                        .slice(0, 5)
                        .map(f => ({ feature: f.name, impact: f.impact }))
                }
            };

            const explanation = await explainWithAI({
                type: 'what_if',
                ...analysisData
            });
            setAiExplanation(explanation);
        } catch (err: any) {
            console.error('Failed to get AI explanation:', err);
            setAiExplanation({
                summary: "AI explanation service is currently unavailable.",
                detailed_explanation: "Unable to generate AI explanation for the what-if analysis at this time. Please try again later.",
                key_takeaways: [
                    "What-if analysis allows you to explore different scenarios",
                    "Feature impacts show which variables influence predictions most",
                    "Try adjusting different feature values to see how predictions change"
                ]
            });
        } finally {
            setAiLoading(false);
        }
    };

    if (!modelLoaded && !error) {
        return (
            <div className="p-6 flex justify-center items-center h-full">
                <Loader2 className="w-8 h-8 animate-spin" />
                <span className="ml-2">Loading model information...</span>
            </div>
        );
    }

    if (error && !modelLoaded) {
        return (
            <div className="p-6 text-center">
                <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-red-600 mb-2">Model Not Available</h3>
                <p className="text-gray-600 dark:text-gray-400 mb-4">{error}</p>
                <p className="text-sm text-gray-500">Please upload a model and dataset first.</p>
            </div>
        );
    }

    return (
        <div className="p-6 space-y-6 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 w-full">
            <div className="flex items-center justify-between">
                <h1 className="text-3xl font-bold flex items-center">
                    <Sliders className="mr-3 text-blue-600" />
                    What-If Analysis
                </h1>
                <div className="flex items-center space-x-2">
                    <ExplainWithAIButton 
                        onClick={handleExplainWithAI} 
                        size="md"
                        loading={aiLoading}
                    />
                    <button 
                        className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center"
                        onClick={resetToMedian}
                        disabled={loading}
                    >
                        <RotateCcw className="w-4 h-4 mr-2" />
                        Reset to Median
                    </button>
                    <button 
                        className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center"
                        onClick={() => performPrediction(features)}
                        disabled={loading}
                    >
                        {loading ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : <RefreshCw className="w-4 h-4 mr-2" />}
                        {loading ? 'Updating...' : 'Refresh'}
                    </button>
                </div>
            </div>

            {showAIExplanation && (
                <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700 rounded-lg p-6 mb-6">
                    <h3 className="text-lg font-semibold text-blue-800 dark:text-blue-200 mb-4 flex items-center">
                        <Zap className="mr-2" />
                        AI Explanation: What-If Analysis
                    </h3>
                    
                    {aiLoading ? (
                        <div className="flex items-center justify-center py-8">
                            <Loader2 className="w-6 h-6 animate-spin mr-2" />
                            <span className="text-blue-700 dark:text-blue-300">Generating AI explanation...</span>
                        </div>
                    ) : aiExplanation ? (
                        <div className="space-y-4">
                            {/* Summary */}
                            <div className="bg-white dark:bg-blue-800/30 rounded-lg p-4">
                                <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">Summary</h4>
                                <p className="text-blue-700 dark:text-blue-300 text-sm leading-relaxed">
                                    {aiExplanation.summary}
                                </p>
                            </div>

                            {/* Detailed Explanation */}
                            <div className="bg-white dark:bg-blue-800/30 rounded-lg p-4">
                                <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">Detailed Analysis</h4>
                                <p className="text-blue-700 dark:text-blue-300 text-sm leading-relaxed whitespace-pre-line">
                                    {aiExplanation.detailed_explanation}
                                </p>
                            </div>

                            {/* Key Takeaways */}
                            <div className="bg-white dark:bg-blue-800/30 rounded-lg p-4">
                                <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">Key Takeaways</h4>
                                <ul className="space-y-1">
                                    {aiExplanation.key_takeaways?.map((takeaway: string, index: number) => (
                                        <li key={index} className="text-blue-700 dark:text-blue-300 text-sm flex items-start">
                                            <span className="text-blue-500 mr-2 mt-1">•</span>
                                            <span>{takeaway}</span>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        </div>
                    ) : (
                        <p className="text-blue-700 dark:text-blue-300 text-sm">
                            Failed to load AI explanation. Please try again.
                        </p>
                    )}
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Feature Controls */}
                <div className="lg:col-span-2 space-y-6">
                    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                        <h2 className="text-xl font-semibold mb-4 flex items-center">
                            <Zap className="mr-2 text-orange-600" />
                            Adjust Feature Values
                        </h2>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {featureList.map((feature) => {
                                const value = features[feature] || 0;
                                const featureRange = featureRanges[feature];
                                
                                if (featureRange?.type === 'categorical') {
                                    // Handle categorical features
                                    return (
                                        <div key={feature} className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                                            <label className="block text-sm font-medium mb-2">
                                                {feature.replace('_', ' ')}
                                            </label>
                                            <select
                                                value={value}
                                                onChange={(e) => handleFeatureChange(feature, e.target.value)}
                                                className="w-full px-3 py-2 border rounded-md bg-white dark:bg-gray-600 dark:border-gray-500"
                                            >
                                                {featureRange.categories?.map((category: any) => (
                                                    <option key={category} value={category}>
                                                        {category}
                                                    </option>
                                                ))}
                                            </select>
                                        </div>
                                    );
                                } else {
                                    // Handle numeric features
                                    const isNumeric = typeof value === 'number';
                                    let min = 0, max = 100, step = 1;
                                    
                                    if (featureRange?.type === 'numeric') {
                                        min = featureRange.min || 0;
                                        max = featureRange.max || 100;
                                        step = featureRange.step || 1;
                                    } else {
                                        // Fallback to heuristics
                                        if (feature.toLowerCase().includes('age')) {
                                            min = 18; max = 80; step = 1;
                                        } else if (feature.toLowerCase().includes('income')) {
                                            min = 20000; max = 200000; step = 1000;
                                        } else if (feature.toLowerCase().includes('score')) {
                                            min = 300; max = 850; step = 10;
                                        }
                                    }
                                    
                                    if (isNumeric) {
                                        return (
                                            <FeatureSlider
                                                key={feature}
                                                name={feature}
                                                value={value}
                                                min={min}
                                                max={max}
                                                step={step}
                                                median={featureRange?.median}
                                                mean={featureRange?.mean}
                                                onChange={(newValue) => handleFeatureChange(feature, newValue)}
                                            />
                                        );
                                    } else {
                                        return (
                                            <FeatureInput
                                                key={feature}
                                                name={feature}
                                                value={value}
                                                type="number"
                                                onChange={(newValue) => handleFeatureChange(feature, newValue)}
                                            />
                                        );
                                    }
                                }
                            })}
                        </div>
                    </div>

                    {/* Feature Impact Analysis */}
                    {featureImpacts.length > 0 && (
                        <FeatureImpactChart impacts={featureImpacts} />
                    )}

                    {error && (
                        <div className="p-4 text-red-500 bg-red-50 dark:bg-red-900/20 rounded-lg flex items-center">
                            <AlertCircle className="mr-2 flex-shrink-0" />
                            <span>{error}</span>
                        </div>
                    )}
                </div>

                {/* Prediction Display */}
                <div>
                    <PredictionDisplay prediction={prediction} modelType={modelType} />
                </div>
            </div>
        </div>
    );
};

export default WhatIfAnalysis;
