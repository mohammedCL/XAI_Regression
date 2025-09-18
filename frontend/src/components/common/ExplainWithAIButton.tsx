import React from 'react';
import { Brain, Loader2 } from 'lucide-react';

interface ExplainWithAIButtonProps {
  onClick: () => void;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
}

const ExplainWithAIButton: React.FC<ExplainWithAIButtonProps> = ({
  onClick,
  className = '',
  size = 'md',
  loading = false
}) => {
  const sizeClasses = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-sm',
    lg: 'px-5 py-2.5 text-base'
  };

  return (
    <button
      onClick={onClick}
      disabled={loading}
      className={`
        inline-flex items-center space-x-2 
        bg-gradient-to-r from-blue-600 to-blue-700 
        hover:from-blue-700 hover:to-blue-800 
        text-white font-medium rounded-lg 
        transition-all duration-200 
        shadow-md hover:shadow-lg 
        transform hover:scale-105 
        disabled:opacity-50 disabled:cursor-not-allowed 
        disabled:transform-none 
        ${sizeClasses[size]} 
        ${className}
      `}
      title="Explain with AI"
    >
      {loading ? (
        <Loader2 className={`${size === 'sm' ? 'w-4 h-4' : size === 'lg' ? 'w-5 h-5' : 'w-4 h-4'} animate-spin`} />
      ) : (
        <Brain className={`${size === 'sm' ? 'w-4 h-4' : size === 'lg' ? 'w-5 h-5' : 'w-4 h-4'}`} />
      )}
      <span>{loading ? 'Generating...' : 'Explain with AI'}</span>
    </button>
  );
};

export default ExplainWithAIButton;
