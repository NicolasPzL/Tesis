/*
import React, { useState } from 'react';
import { Button, Spinner, Alert } from 'react-bootstrap';
import axios from 'axios';

const DataAnalysis = ({ 
  uploadedFile, 
  setAnalysisResults, 
  loading, 
  setLoading, 
  error, 
  setError 
}) => {
  const [analyzing, setAnalyzing] = useState(false);

  const handleAnalyze = async () => {
    if (!uploadedFile) {
      setError('Please upload a file first');
      return;
    }
    
    setAnalyzing(true);
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('http://localhost:5000/api/analyze', {
        filepath: uploadedFile.filepath
      });
      
      setAnalysisResults(response.data.results);
      setAnalyzing(false);
      setLoading(false);
    } catch (error) {
      setAnalyzing(false);
      setLoading(false);
      if (error.response) {
        setError(error.response.data.error || 'Analysis failed');
      } else {
        setError('Network error. Please check if the server is running.');
      }
    }
  };

  return (
    <div className="data-analysis-container">
      {error && (
        <Alert variant="danger" className="mb-3">
          {error}
        </Alert>
      )}
      
      <Button 
        variant="success" 
        onClick={handleAnalyze}
        disabled={!uploadedFile || analyzing}
        className="analyze-btn"
      >
        {analyzing ? (
          <>
            <Spinner
              as="span"
              animation="border"
              size="sm"
              role="status"
              aria-hidden="true"
              className="me-2"
            />
            Analyzing...
          </>
        ) : (
          'Analyze Data'
        )}
      </Button>
      
      <div className="mt-3">
        {uploadedFile ? (
          <p className="text-success">
            File ready for analysis: <strong>{uploadedFile.filename}</strong>
          </p>
        ) : (
          <p className="text-muted">Upload a file to enable analysis</p>
        )}
      </div>
    </div>
  );
};

export default DataAnalysis;
*/