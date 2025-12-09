// src/App.js
import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Button, Spinner, Alert } from 'react-bootstrap';
import FileUpload from './components/FileUpload';
import ColumnSelector from './components/ColumnSelector';
import ResultsDisplay from './components/ResultsDisplay';
import { analyzeData } from './services/api';
import './styles/App.css';

function App() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [availableOriginalColumns, setAvailableOriginalColumns] = useState([]);
  const [selectedOriginalColumns, setSelectedOriginalColumns] = useState([]);
  const [columnMapping, setColumnMapping] = useState({});
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
      if (!uploadedFile) {
          setAvailableOriginalColumns([]);
          setSelectedOriginalColumns([]);
          setColumnMapping({});
          setAnalysisResults(null);
          setError(null);
          setIsAnalyzing(false);
      }
  }, [uploadedFile]);

  const handleFileUploadSuccess = (uploadData) => {
    console.log("App.js: handleFileUploadSuccess received:", uploadData);
    if (uploadData && uploadData.filepath && uploadData.original_columns && uploadData.column_mapping) {
        setUploadedFile({
          filename: uploadData.filename,
          filepath: uploadData.filepath
        });
        setAvailableOriginalColumns(uploadData.original_columns);
        setColumnMapping(uploadData.column_mapping);
        setSelectedOriginalColumns([]);
        setAnalysisResults(null);
        setError(null);
    } else {
        console.error("App.js: Incomplete data from upload service:", uploadData);
        setError("Failed to process file information from server. Key data missing.");
        setUploadedFile(null);
    }
    
    setIsAnalyzing(false);
  };

  const handleColumnSelectionChange = (newSelectedOriginalNames) => {
    setSelectedOriginalColumns(newSelectedOriginalNames);
    setAnalysisResults(null);
    setError(null);
  };

  const handleStartAnalysis = async () => {
    if (!uploadedFile || !uploadedFile.filepath) {
      setError("File information is missing. Please re-upload the file.");
      return;
    }
    if (selectedOriginalColumns.length === 0) {
      setError("Please select at least one column from the list to analyze.");
      return;
    }

    let allColumnsMappedSuccessfully = true;
    let currentErrorMessages = [];

    const cleanedSelectedColumns = selectedOriginalColumns.map(originalName => {
        const cleanedName = columnMapping[originalName];
        if (cleanedName === undefined || cleanedName === null) {
            const mapWarning = `Warning: Column '${originalName}' couldn't be mapped and was skipped. Check name cleaning consistency.`;
            console.warn(`App.js: No mapping for original: '${originalName}'. Skipping.`);
            currentErrorMessages.push(mapWarning);
            allColumnsMappedSuccessfully = false;
            return null; 
        }
        return cleanedName;
    }).filter(Boolean);

    if (cleanedSelectedColumns.length === 0 && selectedOriginalColumns.length > 0) {
        currentErrorMessages.push("Critical Error: None of the selected columns could be mapped to processable names. Please check the file or re-upload.");
        setError(currentErrorMessages.join(" "));
        setIsAnalyzing(false); 
        return;
    }
    
    console.log("App.js: Starting analysis with filepath:", uploadedFile.filepath, 
                "and CLEANED selected columns for API:", cleanedSelectedColumns);
    
    setIsAnalyzing(true);
    // Mostrar warnings de mapeo si los hubo, sino limpiar errores
    if (currentErrorMessages.length > 0) {
        setError(currentErrorMessages.join(" "));
    } else {
        setError(null);
    }
    setAnalysisResults(null);

    try {
      const originalTruthColumn = "Fraud";
      const cleanedTruthColumn = columnMapping[originalTruthColumn] || "fraud"; // fallback por si acaso

      const data = await analyzeData(uploadedFile.filepath, cleanedSelectedColumns, cleanedTruthColumn); 
      
      console.log("App.js: Analysis data received:", data);
      setAnalysisResults(data);
      // Si hubo warnings de mapeo pero el análisis corrió, mantenerlos o añadir mensaje de éxito parcial
      if (!allColumnsMappedSuccessfully && data) { 
        setError(prevError => 
            (prevError ? prevError + " " : "") + 
            "Analysis completed, but note some columns may have been skipped due to naming issues."
        );
      } else if (data && error && error.startsWith("Warning:")) {
          // Mantener el warning si existía y no hubo nuevo error de API
      }
      else if (data) { // Si todo mapeó y el análisis corrió sin error de API
        setError(null); 
      }
    } catch (err) {
       const errorMsg = err.response?.data?.error || err.message || "Analysis failed on the server.";
       console.error("App.js: Analysis error caught:", err);
       setError(errorMsg); // Sobrescribir warnings con errores de API
       setAnalysisResults(null);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="app">
      <Container fluid="lg">
        <Row className="header">
          <Col><h1>Anomaly Detection System</h1><p>Upload, select columns, and analyze financial data.</p></Col>
        </Row>
        {error && (
            <Row className="mb-3">
                <Col><Alert variant={error.toLowerCase().includes("warning") || error.toLowerCase().includes("skipped") ? "warning" : "danger"} onClose={() => setError(null)} dismissible>{error}</Alert></Col>
            </Row>
        )}
        <Row className="mt-2">
          <Col md={5} lg={4} className="mb-3 mb-md-0">
            <Card className="upload-card h-100">
              <Card.Body className="d-flex flex-column">
                <Card.Title><span className="badge bg-secondary me-1">1</span> Upload Data</Card.Title>
                <FileUpload onUploadSuccess={handleFileUploadSuccess} setError={setError} clearUploadedFile={() => setUploadedFile(null)} />
              </Card.Body>
            </Card>
          </Col>
          <Col md={7} lg={8}>
            <Card className="analysis-card h-100">
              <Card.Body className="d-flex flex-column">
                <Card.Title><span className="badge bg-secondary me-1">2</span> Select Columns & Analyze</Card.Title>
                {uploadedFile && availableOriginalColumns.length > 0 ? (
                  <>
                    <div className="column-selector-wrapper mb-3">
                      <ColumnSelector columns={availableOriginalColumns} selectedColumns={selectedOriginalColumns} onChange={handleColumnSelectionChange}/>
                    </div>
                    <div className="mt-auto">
                       <Button variant="success" onClick={handleStartAnalysis} disabled={isAnalyzing || selectedOriginalColumns.length === 0 || !uploadedFile} className="analyze-btn w-100">
                          {isAnalyzing ? (<><Spinner animation="border" size="sm" className="me-2"/> Analyzing...</>) : (`Analyze ${selectedOriginalColumns.length} Column(s)`)}
                       </Button>
                     </div>
                  </>
                ) : (
                   <div className="text-center text-muted d-flex align-items-center justify-content-center h-100">
                       <p>{uploadedFile ? "No columns found or file processing issue." : "Upload a file to select columns."}</p>
                   </div>
                )}
              </Card.Body>
            </Card>
          </Col>
        </Row>
         {isAnalyzing && (
              <Row className="mt-4 text-center">
                   <Col><Spinner animation="border" variant="primary" role="status"><span className="visually-hidden">Analyzing...</span></Spinner><p className="mt-2 text-primary">Analyzing data...</p></Col>
              </Row>
         )}
        {!isAnalyzing && analysisResults && (
          <Row className="mt-4"><Col><ResultsDisplay data={analysisResults} /></Col></Row>
        )}
      </Container>
    </div>
  );
}
export default App;