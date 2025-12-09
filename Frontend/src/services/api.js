// src/services/api.js
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

export const uploadFile = async (file, onUploadProgress) => {
  const formData = new FormData();
  formData.append('file', file);
  console.log("Service: Uploading file:", file.name);
  try {
      const response = await axios.post(`${API_BASE_URL}/api/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress
      });
      console.log("Service: Upload successful, response data:", response.data);
      return response.data;
  } catch (error) {
      console.error("Service: Error in uploadFile:", error.response?.data?.error || error.message);
      throw error;
  }
};

export const analyzeData = async (filepath, selectedCleanedColumns, truthColumnName = "Fraud") => { // Default 'Fraud'
  if (!filepath) throw new Error("Filepath is required for analysis.");
  if (!selectedCleanedColumns || !Array.isArray(selectedCleanedColumns) || selectedCleanedColumns.length === 0) {
      throw new Error("At least one cleaned column must be provided for analysis.");
  }

  console.log("Service: Requesting analysis for:", filepath, 
              "with cleaned columns:", selectedCleanedColumns,
              "and TRUTH COLUMN to be used by backend:", truthColumnName);
  try {
      const response = await axios.post(`${API_BASE_URL}/api/analyze`, {
         filepath: filepath,
         selected_columns: selectedCleanedColumns,
         truth_column_name: truthColumnName // Enviar nombre de la columna de verdad
      });
      console.log("Service: Analysis successful, response data:", response.data);
      return response.data;
  } catch (error) {
      console.error("Service: Error in analyzeData:", error.response?.data?.error || error.message);
      throw error;
  }
};