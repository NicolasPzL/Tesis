// src/components/FileUpload.js
import React, { useState, useRef } from 'react'; // Añadir useRef
import { Form, Button, Spinner, Alert, ProgressBar } from 'react-bootstrap';
import { uploadFile } from '../services/api'; // Importar función del servicio

/**
 * Componente para manejar la selección y subida de archivos.
 * @param {object} props
 * @param {function} props.onUploadSuccess - Callback llamado con datos del archivo subido ({filename, filepath, columns}).
 * @param {function} props.setError - Callback para reportar errores al componente padre.
 * @param {function} props.clearUploadedFile - Callback para notificar al padre que se deseleccionó el archivo.
 */
const FileUpload = ({ onUploadSuccess, setError, clearUploadedFile }) => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [localError, setLocalError] = useState(null);
  const fileInputRef = useRef(null); // Referencia al input para poder limpiarlo

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    // Resetear todo al cambiar archivo
    setFile(selectedFile || null); // Guardar null si no se selecciona nada
    setLocalError(null);
    setUploadProgress(0);
    if (setError) setError(null); // Limpiar error global

    // Notificar al padre si no hay archivo seleccionado
    if (!selectedFile && clearUploadedFile) {
         clearUploadedFile();
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file) {
      setLocalError('Please select a file first.');
      return;
    }

    // Validación de tipo (cliente) - puede ser menos estricta o quitarse si el backend valida bien
    const allowedExtensions = ['csv', 'xls', 'xlsx'];
    const fileExtension = file.name.split('.').pop()?.toLowerCase();
    if (!fileExtension || !allowedExtensions.includes(fileExtension)) {
       setLocalError('Invalid file type. Please upload a CSV or Excel file.');
       return;
    }
    /* // Validación MIME type (alternativa, a veces menos fiable)
     const allowedTypes = [
         'text/csv', 'application/csv', // Comunes para CSV
         'application/vnd.ms-excel', // .xls
         'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' // .xlsx
     ];
     if (!allowedTypes.includes(file.type)) {
       setLocalError(`Invalid file MIME type (${file.type}). Please upload a CSV or Excel file.`);
       return;
     }
    */


    setUploading(true);
    setLocalError(null);
    setUploadProgress(0);
    if (setError) setError(null);

    try {
      // --- USAR EL SERVICIO API ---
      console.log("Calling uploadFile service...");
      const uploadData = await uploadFile(file, (progressEvent) => {
         // Calcular y actualizar progreso
         if (progressEvent.total) {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploadProgress(percentCompleted);
         } else {
             // Manejar caso donde total no está disponible (raro con axios)
             setUploadProgress(50); // O algún indicador visual diferente
         }
      });
      // --------------------------

      // Llamar al callback del padre con los datos completos
      // (nombre original, filepath del servidor, columnas)
      onUploadSuccess({
          filename: file.name,
          filepath: uploadData.filepath,
          original_columns: uploadData.original_columns,
          column_mapping: uploadData.column_mapping
      });
      // Limpiar el input de archivo después de subir exitosamente (opcional)
      // if (fileInputRef.current) {
      //     fileInputRef.current.value = "";
      // }
      // setFile(null); // Resetear archivo local si se limpia el input


    } catch (error) {
      // Manejar errores de la llamada API (ya logueados en el servicio)
      const errorMsg = error.response?.data?.error || error.message || 'Upload failed.';
      setLocalError(errorMsg);
      if (setError) setError(errorMsg); // Propagar error al padre
    } finally {
      setUploading(false); // Terminar estado de subida en cualquier caso
    }
  };

  return (
     <div className="file-upload-container d-flex flex-column h-100"> {/* Flex para ocupar altura */}
        {/* Alerta para errores específicos de la subida */}
        {localError && (
            <Alert variant="warning" onClose={() => setLocalError(null)} dismissible className="mb-3 small">
                {localError}
            </Alert>
         )}

        {/* Formulario de subida */}
        <Form onSubmit={handleSubmit} style={{ flexGrow: 1 }}> {/* Crece para ocupar espacio */}
            <Form.Group controlId="formFile" className="mb-3">
             <Form.Label>Select CSV or Excel file:</Form.Label>
             <Form.Control
                 ref={fileInputRef} // Añadir referencia
                 type="file"
                 onChange={handleFileChange}
                 accept=".csv,.xls,.xlsx" // Filtro de extensión del navegador
                 disabled={uploading}
                 isInvalid={!!localError && !file} // Marcar como inválido si hay error y no hay archivo
              />
              <Form.Text className="text-muted">
                 Max file size: 50MB.
              </Form.Text>
            </Form.Group>

            {/* Barra de progreso durante la subida */}
            {uploading && ( // Mostrar siempre si está subiendo
                <ProgressBar
                   now={uploadProgress}
                   label={uploadProgress > 0 ? `${uploadProgress}%` : ''} // No mostrar 0%
                   className="mb-3"
                   animated={uploadProgress < 100} // Animar mientras sube
                   variant={uploadProgress === 100 ? 'success' : 'info'} // Cambiar color al completar
                />
             )}

            {/* Botón de subida */}
            <div className="mt-auto"> {/* Empujar botón al fondo */}
                <Button
                   variant="primary"
                   type="submit"
                   disabled={uploading || !file} // Deshabilitado si está subiendo o no hay archivo
                   className="upload-btn w-100" // Ocupar ancho completo
                >
                   {uploading ? (
                       <> <Spinner animation="border" size="sm" className="me-2" /> Uploading... </>
                    ) : (
                       'Upload & Get Columns'
                    )}
                 </Button>
             </div>
        </Form>
     </div>
  );
};

export default FileUpload;