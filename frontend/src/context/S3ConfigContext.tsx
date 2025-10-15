import React, { createContext, useContext, useState } from 'react';

export interface S3Config {
  modelUrl: string;
  trainDatasetUrl: string;
  testDatasetUrl: string;
  targetColumn: string;
}

interface S3ConfigContextType {
  config: S3Config;
  setConfig: (config: S3Config) => void;
}

const defaultConfig: S3Config = {
  modelUrl: '',
  trainDatasetUrl: '',
  testDatasetUrl: '',
  targetColumn: 'target',
};

const S3ConfigContext = createContext<S3ConfigContextType | undefined>(undefined);

export const S3ConfigProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [config, setConfig] = useState<S3Config>(defaultConfig);
  return (
    <S3ConfigContext.Provider value={{ config, setConfig }}>
      {children}
    </S3ConfigContext.Provider>
  );
};

export const useS3Config = () => {
  const context = useContext(S3ConfigContext);
  if (!context) {
    throw new Error('useS3Config must be used within a S3ConfigProvider');
  }
  return context;
};
