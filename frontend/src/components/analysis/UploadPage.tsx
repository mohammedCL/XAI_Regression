import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
// import { uploadModelAndData, uploadModelAndSeparateDatasets } from '../../services/api';
import { UploadCloud, Loader2 } from 'lucide-react';
import { useS3Config } from '../../context/S3ConfigContext';

const UploadPage: React.FC = () => {
    // File upload states removed for stateless S3 URL input only
    const [targetColumn, setTargetColumn] = useState('');
    const [modelUrl, setModelUrl] = useState('');
    const [trainDatasetUrl, setTrainDatasetUrl] = useState('');
    const [testDatasetUrl, setTestDatasetUrl] = useState('');
    const { setConfig } = useS3Config();
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const navigate = useNavigate();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!modelUrl || !trainDatasetUrl || !testDatasetUrl || !targetColumn) {
            setError('Please provide all S3 URLs and target column.');
            return;
        }
        setError('');
        setIsLoading(true);
        try {
            setConfig({
                modelUrl,
                trainDatasetUrl,
                testDatasetUrl,
                targetColumn,
            });
            navigate('/overview');
        } catch (err: any) {
            setError('An unexpected error occurred.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex items-center justify-center min-h-screen bg-gray-100 dark:bg-gray-900">
            <div className="w-full max-w-lg p-8 space-y-6 bg-white rounded-lg shadow-md dark:bg-gray-800">
                <div className="text-center">
                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Start New Analysis</h1>
                    <p className="text-gray-600 dark:text-gray-400">Enter your S3 URLs and target column to begin.</p>
                </div>

                {/* Only S3 URL and target column inputs for stateless API */}

                <form className="space-y-6" onSubmit={handleSubmit}>
                    <div>
                        <label className="block mb-2 text-sm font-medium text-gray-700 dark:text-gray-300">Model S3 URL</label>
                        <input type="text" value={modelUrl} onChange={e => setModelUrl(e.target.value)} placeholder="https://s3.amazonaws.com/bucket/model.joblib" className="w-full px-3 py-2 border rounded-md dark:bg-gray-700 dark:border-gray-600" />
                    </div>
                    <div>
                        <label className="block mb-2 text-sm font-medium text-gray-700 dark:text-gray-300">Train Dataset S3 URL</label>
                        <input type="text" value={trainDatasetUrl} onChange={e => setTrainDatasetUrl(e.target.value)} placeholder="https://s3.amazonaws.com/bucket/train.csv" className="w-full px-3 py-2 border rounded-md dark:bg-gray-700 dark:border-gray-600" />
                    </div>
                    <div>
                        <label className="block mb-2 text-sm font-medium text-gray-700 dark:text-gray-300">Test Dataset S3 URL</label>
                        <input type="text" value={testDatasetUrl} onChange={e => setTestDatasetUrl(e.target.value)} placeholder="https://s3.amazonaws.com/bucket/test.csv" className="w-full px-3 py-2 border rounded-md dark:bg-gray-700 dark:border-gray-600" />
                    </div>
                    <div>
                        <label className="block mb-2 text-sm font-medium text-gray-700 dark:text-gray-300">Target Column Name</label>
                        <input type="text" value={targetColumn} onChange={e => setTargetColumn(e.target.value)} placeholder="e.g., 'target' or 'has_churned'" className="w-full px-3 py-2 border rounded-md dark:bg-gray-700 dark:border-gray-600" />
                    </div>
                    {error && <p className="text-sm text-red-500">{error}</p>}
                    <button type="submit" disabled={isLoading} className="w-full px-4 py-2 font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:bg-blue-300 flex items-center justify-center">
                        {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <UploadCloud className="w-5 h-5 mr-2" />}
                        {isLoading ? 'Analyzing...' : 'Upload & Analyze'}
                    </button>
                </form>
            </div>
        </div>
    );
};

export default UploadPage;