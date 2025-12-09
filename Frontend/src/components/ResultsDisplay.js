// src/components/ResultsDisplay.js
import React from 'react';
import { Card, Row, Col, Table, Image, Alert } from 'react-bootstrap';

/**
 * Componente para mostrar los resultados del análisis, incluyendo gráficos orientados al cliente.
 * @param {object} props
 * @param {object} props.data - El objeto completo devuelto por /api/analyze (contiene 'results' y 'plots').
 */
const ResultsDisplay = ({ data }) => {
  // Log para depurar los datos recibidos
  console.log("ResultsDisplay props 'data':", JSON.stringify(data, null, 2));

  const results = data?.results;
  const plots = data?.plots; // Diccionario con los gráficos base64

  // Log específico para el objeto plots
  if (plots) {
    console.log("Objeto 'plots' recibido:", JSON.stringify(plots, null, 2));
  } else {
    console.warn("El objeto 'plots' NO está presente en props.data o es undefined.");
  }


  if (!results || typeof results !== 'object') {
    return (
        <Card className="results-card mt-4">
            <Card.Body>
                <Card.Title>Resultados del Análisis</Card.Title>
                <p className="text-muted fst-italic">Esperando resultados del análisis o los datos recibidos no son válidos.</p>
            </Card.Body>
        </Card>
    );
  }

  // Función auxiliar para formatear valores en la tabla
  const formatValue = (value) => {
    if (value === null || typeof value === 'undefined') return <span className="text-muted">N/A</span>;
    if (typeof value === 'number') {
      if (value % 1 !== 0) { // Es un decimal
        if (Math.abs(value) > 1e6 || (Math.abs(value) < 1e-3 && value !== 0)) return value.toExponential(2); // Notación científica para números muy grandes/pequeños
        return value.toFixed(4); // Decimales estándar
      }
      return value.toLocaleString(); // Enteros
    }
    if (typeof value === 'boolean') { return value ? <span className="badge bg-danger">Sí</span> : <span className="badge bg-secondary">No</span>; }
    const strValue = String(value);
    if (strValue.length > 100) { return strValue.substring(0, 97) + '...'; } // Acortar strings largos
    return strValue;
  };

  // Función para formatear nombres de columnas (Título Amigable)
  const formatTitle = (key) => key?.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()) ?? 'N/A';

  const anomalyRecords = results.sample_anomalous_records || [];
  const anomalyColumns = anomalyRecords.length > 0 ? Object.keys(anomalyRecords[0]) : ['No hay datos de muestra'];

  // Nombres de las columnas usadas para gráficos (para títulos), obtenidos del backend
  const plottedNumCol = results.plotted_numerical_feature;
  const plottedCatCol = results.plotted_categorical_feature;

  return (
    <Card className="results-card mt-4 shadow-sm">
      <Card.Header as="h5">Resultados del Análisis</Card.Header>
      <Card.Body>
        {/* Resumen Numérico */}
        <Alert variant="light" className="results-summary border mb-4">
            <Row className="align-items-center">
                <Col md={3} className="mb-2 mb-md-0 border-end"><div><strong>Registros Analizados:</strong></div><div className="fs-5">{results.total_records_analyzed?.toLocaleString() ?? 'N/A'}</div></Col>
                <Col md={3} className="mb-2 mb-md-0 border-end"><div className="text-danger"><strong>Anomalías Detectadas:</strong></div><div className="fs-5 text-danger">{results.detected_anomalies_count?.toLocaleString() ?? 'N/A'} ({results.detected_anomalies_percentage ?? 'N/A'}%)</div></Col>
                <Col md={3} className="mb-2 mb-md-0 border-end"><div> <strong>Umbral (MSE):</strong></div><div className="fs-5">{(results.anomaly_threshold ?? 0).toExponential(4)}</div></Col>
                {results.evaluation_metrics && (
                    <Col md={3} className="mb-2 mb-md-0"><div><strong>F1-Score (Fraude):</strong></div><div className="fs-5">{results.evaluation_metrics.f1_score_fraud ?? 'N/A'}</div></Col>
                )}
            </Row>
            <hr />
            <Row><Col><small><strong>Columnas Utilizadas en Análisis:</strong> {results.columns_analyzed?.join(', ') ?? 'N/A'}</small></Col></Row>
            {results.features_used_in_model && <Row><Col><small><strong>Características en Modelo:</strong> {results.features_used_in_model.join(', ') ?? 'N/A'}</small></Col></Row>}
        </Alert>

        {/* --- Gráficas --- */}
        {plots && Object.keys(plots).length > 0 ? ( // Solo mostrar sección si hay plots
            <div className="plots-container mb-4">
                <h5 className="mb-3">Perspectivas del Análisis Visual</h5>
                <Row>
                    {/* Pie Chart: Resumen de Predicciones */}
                    {plots.prediction_summary_pie && (
                        <Col xs={12} md={6} lg={4} className="mb-4">
                            <Card className="h-100 shadow-sm">
                                <Card.Header className="text-center small fw-bold bg-light">Resumen General de Predicciones</Card.Header>
                                <Card.Body className="text-center d-flex flex-column justify-content-center align-items-center p-2">
                                    <Image src={`data:image/png;base64,${plots.prediction_summary_pie}`} fluid thumbnail alt="Gráfico Circular de Resumen de Predicciones" style={{ maxHeight: '250px', objectFit: 'contain' }}/>
                                    <p className="text-muted small mt-2 mb-0">Porcentaje de registros normales vs. potencialmente anómalos según el modelo.</p>
                                </Card.Body>
                            </Card>
                        </Col>
                    )}

                    {/* Bar Chart: Severidad de Anomalías */}
                    {plots.predicted_anomaly_score_severity && (
                        <Col xs={12} md={6} lg={4} className="mb-4">
                            <Card className="h-100 shadow-sm">
                                <Card.Header className="text-center small fw-bold bg-light">Niveles de Severidad de Anomalías</Card.Header>
                                <Card.Body className="text-center d-flex flex-column justify-content-center align-items-center p-2">
                                    <Image src={`data:image/png;base64,${plots.predicted_anomaly_score_severity}`} fluid thumbnail alt="Gráfico de Barras de Niveles de Severidad" style={{ maxHeight: '250px', objectFit: 'contain' }}/>
                                    <p className="text-muted small mt-2 mb-0">Agrupación de registros anómalos por su nivel de puntuación de anomalía.</p>
                                </Card.Body>
                            </Card>
                        </Col>
                    )}

                    {/* Bar Chart: Top Categorías en Anomalías */}
                    {plots.top_predicted_anomaly_categories && plottedCatCol && (
                        <Col xs={12} md={6} lg={4} className="mb-4">
                            <Card className="h-100 shadow-sm">
                                <Card.Header className="text-center small fw-bold bg-light">Top "{formatTitle(plottedCatCol)}" en Anomalías</Card.Header>
                                <Card.Body className="text-center d-flex flex-column justify-content-center align-items-center p-2">
                                    <Image src={`data:image/png;base64,${plots.top_predicted_anomaly_categories}`} fluid thumbnail alt={`Gráfico de Top ${formatTitle(plottedCatCol)} en Anomalías`} style={{ maxHeight: '250px', objectFit: 'contain' }}/>
                                    <p className="text-muted small mt-2 mb-0">Categorías más frecuentes para "{formatTitle(plottedCatCol)}" entre los registros anómalos.</p>
                                </Card.Body>
                            </Card>
                        </Col>
                    )}

                    {/* Boxplot: Rango Numérico en Anomalías */}
                    {plots.predicted_anomaly_numerical_range && plottedNumCol && (
                        <Col xs={12} md={6} lg={4} className="mb-4">
                            <Card className="h-100 shadow-sm">
                                <Card.Header className="text-center small fw-bold bg-light">Rango de "{formatTitle(plottedNumCol)}" en Anomalías</Card.Header>
                                <Card.Body className="text-center d-flex flex-column justify-content-center align-items-center p-2">
                                    <Image src={`data:image/png;base64,${plots.predicted_anomaly_numerical_range}`} fluid thumbnail alt={`Gráfico de Rango de ${formatTitle(plottedNumCol)} en Anomalías`} style={{ maxHeight: '250px', objectFit: 'contain' }}/>
                                    <p className="text-muted small mt-2 mb-0">Distribución para "{formatTitle(plottedNumCol)}" dentro de los registros anómalos.</p>
                                </Card.Body>
                            </Card>
                        </Col>
                    )}

                    {/* Heatmap: Matriz de Confusión */}
                    {plots.confusion_matrix && (
                        <Col xs={12} md={6} lg={4} className="mb-4">
                            <Card className="h-100 shadow-sm">
                                <Card.Header className="text-center small fw-bold bg-light">Matriz de Confusión del Modelo</Card.Header>
                                <Card.Body className="text-center d-flex flex-column justify-content-center align-items-center p-2">
                                    <Image src={`data:image/png;base64,${plots.confusion_matrix}`} fluid thumbnail alt="Heatmap de Matriz de Confusión" style={{ maxHeight: '250px', objectFit: 'contain' }}/>
                                    <p className="text-muted small mt-2 mb-0">Rendimiento del modelo comparando predicciones con etiquetas reales.</p>
                                </Card.Body>
                            </Card>
                        </Col>
                    )}
                </Row>
            </div>
        ) : (
            <Alert variant="light" className="text-center border fst-italic mt-3">No hay gráficos disponibles para mostrar.</Alert>
        )}


        {/* Tabla de Registros Anómalos */}
        <div className="anomaly-records mt-4">
            <h5 className="mb-3">Muestra de Registros Anómalos <small className="text-muted">(Top 10 por Puntuación de Anomalía)</small></h5>
            {anomalyRecords.length > 0 ? (
                <div className="table-responsive">
                    <Table striped bordered hover responsive="sm" size="sm" className="shadow-sm">
                        <thead className="table-light">
                            <tr>{anomalyColumns.map((key) => (<th key={key} style={{whiteSpace: 'nowrap'}}>{formatTitle(key)}</th>))}</tr>
                        </thead>
                        <tbody>
                        {anomalyRecords.map((record, index) => (
                            <tr key={record.original_index_from_df ?? record.original_index ?? `anomaly-${index}`}>
                                {anomalyColumns.map((key) => (
                                    <td key={`${record.original_index_from_df ?? record.original_index ?? index}-${key}`}
                                        className={key === 'anomaly_score_mse' ? 'fw-bold text-danger' : (key === 'is_anomaly_pred' ? 'text-center' : '')}
                                        style={{maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'}}
                                        title={typeof record[key] === 'string' || typeof record[key] === 'number' ? record[key] : ''} // Tooltip para contenido completo
                                    >
                                        {formatValue(record[key])}
                                    </td>
                                ))}
                            </tr>
                        ))}
                        </tbody>
                    </Table>
                </div>
            ) : (
                <Alert variant="light" className="text-center border fst-italic">No se encontraron registros anómalos específicos por encima del umbral o la muestra está vacía.</Alert>
            )}
        </div>

        {/* Reporte de Clasificación (si existe) */}
        {results.evaluation_metrics?.classification_report && (
            <div className="classification-report mt-4">
                <h5 className="mb-3">Reporte de Clasificación Detallado</h5>
                <Card className="shadow-sm">
                    <Card.Body>
                        <pre style={{fontSize: '0.8rem', whiteSpace: 'pre-wrap', wordBreak: 'break-all'}}>
                            {results.evaluation_metrics.classification_report}
                        </pre>
                    </Card.Body>
                </Card>
            </div>
        )}


      </Card.Body>
      <Card.Footer className="text-muted small">
        Análisis completado el: {new Date().toLocaleString()}
      </Card.Footer>
    </Card>
  );
};

export default ResultsDisplay;