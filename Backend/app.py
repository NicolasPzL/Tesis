# ./app.py
# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import time # Para medir tiempos y logging
from werkzeug.utils import secure_filename

# Intentar importar módulos locales con manejo de errores más informativo
try:
    from utils.procesador_archivo import process_file # Esta función debe devolver df_limpio, originales, mapeo
except ImportError as e:
    logging.critical(f"CRITICAL ERROR: Could not import 'process_file' from 'utils.procesador_archivo'. Check file existence and imports. Error: {e}")
    # Definir una función dummy para que la app pueda intentar arrancar y el error sea visible en logs
    def process_file(filepath): 
        logging.error("Dummy process_file called due to import error.")
        raise ImportError("Function 'process_file' is not available due to import error.")

try:
    # Asumiendo que Autoencoder es el algoritmo seleccionado
    from models.deteccion_anomalias import run_anomaly_detection 
except ImportError as e:
     logging.critical(f"CRITICAL ERROR: Could not import 'run_anomaly_detection' from 'models.deteccion_anomalias'. Check file existence and imports. Error: {e}")
     def run_anomaly_detection(df, selected_columns, truth_column_name=None): 
         logging.error("Dummy run_anomaly_detection called due to import error.")
         raise ImportError("Function 'run_anomaly_detection' is not available.")

app = Flask(__name__)

# Configuración de Logging
# Usar el logger de Flask directamente es a menudo más limpio
app.logger.setLevel(logging.INFO) # Asegurar que el logger de Flask capture INFO y superior
# Si quieres un formato específico para todos los logs (incluyendo los de Flask):
# Esto ya lo hace Flask si debug=False, o si debug=True usa su propio formato.
# Pero si quieres forzarlo siempre:
if not app.debug: # Solo si no estás en modo debug, para no interferir con el de Flask
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Configuración de CORS
CORS(app, resources={r"/api/*": {"origins": "*"}}) # ATENCIÓN: En producción, restringir orígenes!
app.logger.info("CORS enabled for /api/* with all origins (Development setting - RESTRICT IN PRODUCTION!).")

# Configuración de Subida de Archivos
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

if not os.path.exists(UPLOAD_FOLDER):
    try:
        os.makedirs(UPLOAD_FOLDER)
        app.logger.info(f"Upload folder created: {os.path.abspath(UPLOAD_FOLDER)}")
    except OSError as e:
         app.logger.error(f"Fatal: Could not create upload folder '{UPLOAD_FOLDER}': {e}", exc_info=True)
         # En un caso real, podrías querer que la app no inicie si esto falla.
         # exit(1) 

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 # Límite de 50MB
app.logger.info(f"Upload folder set to: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
app.logger.info(f"Max upload file size set to: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f} MB")


# --- Funciones Auxiliares ---
# Asegúrate que esta definición NO tenga caracteres extraños al inicio de la línea 'def'
def allowed_file(filename):
    """Verifica si la extensión del archivo está en ALLOWED_EXTENSIONS."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Rutas de la API ---

@app.route('/api/upload', methods=['POST'])
def upload_file_route():
    """
    Endpoint para subir archivos.
    Guarda el archivo, lo procesa para obtener nombres de columna originales,
    nombres limpios (implícito en el df devuelto por process_file), y el mapeo.
    Devuelve esta información al frontend.
    """
    endpoint_start_time = time.time()
    app.logger.info(f"Received /api/upload request from {request.remote_addr}")

    if 'file' not in request.files:
        app.logger.warning("Upload rejected: No 'file' part in request.")
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if not file or not file.filename: # Chequeo más robusto
        app.logger.warning("Upload rejected: No file selected or filename is empty.")
        return jsonify({'error': 'No file selected or filename empty'}), 400

    original_filename_for_log = file.filename # Para logging antes de secure_filename
    if not allowed_file(original_filename_for_log):
        app.logger.warning(f"Upload rejected: File type not allowed for '{original_filename_for_log}'. Allowed: {ALLOWED_EXTENSIONS}")
        return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    filename = secure_filename(original_filename_for_log) # Limpia el nombre para seguridad
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    app.logger.info(f"Processing allowed file: '{filename}' (original: '{original_filename_for_log}')")

    try:
        file.save(filepath)
        app.logger.info(f"File saved to: {filepath}")
        
        # process_file AHORA devuelve: df_con_nombres_limpios, lista_nombres_originales, mapeo_original_a_limpio
        df_processed, original_columns, column_mapping = process_file(filepath)
        app.logger.info(f"File '{filename}' processed. Original columns count: {len(original_columns)}. Mapping keys count: {len(column_mapping)}")

        response_data = {
            'message': 'File uploaded and columns extracted successfully',
            'filename': filename,
            'filepath': filepath, # El frontend lo necesitará para la llamada a /analyze
            'original_columns': original_columns, # Para mostrar al usuario en ColumnSelector
            'column_mapping': column_mapping   # Diccionario {original_name: cleaned_name}
        }
        # app.logger.debug(f"Response for /api/upload: {response_data}") # Cuidado con loguear datos grandes
        
        total_time = time.time() - endpoint_start_time
        app.logger.info(f"/api/upload for '{filename}' completed successfully in {total_time:.2f} seconds.")
        return jsonify(response_data), 200

    except ValueError as e: # Errores específicos durante process_file (formato, vacío, etc.)
        app.logger.error(f"ValueError during upload/processing for '{filename}': {e}", exc_info=False) # No es necesario el traceback completo al cliente
        if os.path.exists(filepath): # Intentar limpiar
            try: os.remove(filepath); app.logger.info(f"Removed partially processed file: {filepath}")
            except OSError as rm_err: app.logger.warning(f"Could not remove file {filepath} after error: {rm_err}")
        return jsonify({'error': f"Error processing file: {str(e)}"}), 400 # Error del cliente o datos
    except Exception as e: # Errores inesperados del servidor
        app.logger.error(f"Unexpected server error during upload of '{filename}': {e}", exc_info=True) # Loguear traceback completo
        if os.path.exists(filepath): # Intentar limpiar
             try: os.remove(filepath); app.logger.info(f"Removed file {filepath} after unexpected error.")
             except OSError as rm_err: app.logger.warning(f"Could not remove file {filepath} after error: {rm_err}")
        return jsonify({'error': "An unexpected server error occurred during upload."}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_data_route():
    """
    Endpoint para ejecutar el análisis de anomalías.
    1. Valida la petición JSON y los parámetros (filepath, selected_columns).
    2. Espera que 'selected_columns' contenga los nombres de columna LIMPIOS
       (traducidos por el frontend usando el mapeo).
    3. Espera 'truth_column_name' (default 'Fraud').
    4. Llama a process_file de nuevo para obtener el DataFrame limpio.
    5. Llama a run_anomaly_detection con el df y las columnas limpias seleccionadas.
    6. Devuelve los resultados y los gráficos generados.
    """
    endpoint_start_time = time.time()
    app.logger.info(f"Received /api/analyze request from {request.remote_addr}")

    if not request.is_json:
        app.logger.warning("Analyze request rejected: Content-Type is not application/json.")
        return jsonify({"error": "Request must be JSON"}), 415 # Unsupported Media Type

    data = request.get_json()
    filepath = data.get('filepath')
    selected_cleaned_columns = data.get('selected_columns') # Frontend DEBE enviar nombres ya limpios
    truth_column_name = data.get('truth_column_name', 'Fraud') # Default "Fraud" (con F mayúscula)

    # Validaciones de entrada robustas
    if not filepath or not isinstance(filepath, str):
         app.logger.warning("Analyze request rejected: Filepath missing or invalid.")
         return jsonify({'error': 'Filepath missing or invalid in the request'}), 400
    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        app.logger.error(f"Analysis target file not found or is not a file: {filepath}")
        return jsonify({'error': 'File for analysis not found on server. Please re-upload.'}), 404
    if not selected_cleaned_columns or not isinstance(selected_cleaned_columns, list) or not selected_cleaned_columns: # Chequeo de lista vacía
         app.logger.warning(f"Analyze request rejected: Invalid or missing selected_columns. Expected cleaned names. Got: {selected_cleaned_columns}")
         return jsonify({'error': 'No columns selected for analysis or invalid format.'}), 400
    if not all(isinstance(col, str) for col in selected_cleaned_columns): # Chequeo de que todos sean strings
         app.logger.warning(f"Analyze request rejected: Non-string element found in selected_columns: {selected_cleaned_columns}")
         return jsonify({'error': 'Invalid format: selected_columns must be a list of strings.'}), 400

    app.logger.info(f"Analyzing file: {filepath} with selected_cleaned_columns: {selected_cleaned_columns}, truth_column: '{truth_column_name}'")
    try:
        # Re-leer el archivo para obtener el DataFrame con nombres limpios.
        # La función process_file ya se encarga de limpiar los nombres consistentemente.
        df_for_analysis, _, column_mapping_on_reread = process_file(filepath) # df_for_analysis tiene nombres de columna limpios
        app.logger.info(f"File re-read for analysis. DataFrame columns: {df_for_analysis.columns.tolist()}")

        # Verificación crucial: las columnas limpias seleccionadas deben existir en el DataFrame
        missing_in_df = [col for col in selected_cleaned_columns if col not in df_for_analysis.columns]
        if missing_in_df:
            # Este error es grave e indica un desajuste entre la limpieza/mapeo y lo que el frontend envió.
            app.logger.error(f"CRITICAL MISMATCH: Cleaned columns from frontend {missing_in_df} are NOT in the processed DataFrame columns ({df_for_analysis.columns.tolist()}) for file {filepath}. Mapping on re-read was: {column_mapping_on_reread}. Columns sent by frontend: {selected_cleaned_columns}")
            return jsonify({'error': f'Internal error during column matching: Columns {missing_in_df} requested for analysis were not found after processing the file. This may indicate an issue with column name cleaning consistency or an error in the frontend mapping/selection logic.'}), 500

        # run_anomaly_detection espera el df completo (con columna de verdad y nombres limpios),
        # la lista de features seleccionadas (nombres limpios) y el nombre de la columna de verdad.
        results_data, result_plots_base64 = run_anomaly_detection(df_for_analysis, selected_cleaned_columns, truth_column_name)
        app.logger.info("Anomaly detection function executed successfully.")
        
        total_time = time.time() - endpoint_start_time
        app.logger.info(f"/api/analyze for '{filepath}' completed successfully in {total_time:.2f} seconds.")
        return jsonify({'message': 'Analysis completed', 'results': results_data, 'plots': result_plots_base64}), 200

    except FileNotFoundError: # Si process_file falla al releer
         app.logger.error(f"FileNotFoundError during analysis phase (re-reading file): {filepath}", exc_info=True)
         return jsonify({'error': 'File could not be re-read for analysis. Please try uploading again.'}), 500
    except ValueError as e: # Errores de validación/procesamiento
         app.logger.error(f"ValueError during analysis of {filepath}: {e}", exc_info=True)
         return jsonify({'error': str(e)}), 400 
    except ImportError as e: # Si falta alguna dependencia del modelo
         app.logger.error(f"ImportError during analysis: {e}", exc_info=True)
         return jsonify({'error': f"Server configuration error: A required library might be missing ({e}). Please contact support."}), 500
    except Exception as e: # Captura cualquier otro error inesperado (ej. memoria, error en modelo PyTorch)
        app.logger.error(f"Unexpected error during analysis of {filepath}: {e}", exc_info=True)
        return jsonify({'error': 'An unexpected server error occurred during analysis. Please check server logs or contact support.'}), 500

# --- Ejecución del Servidor (para desarrollo) ---
if __name__ == '__main__':
    app.logger.info(f"Starting Flask development server on http://0.0.0.0:5000 (DEBUG MODE: {'ON' if app.debug else 'OFF'})")
    # Para producción, usa un servidor WSGI como Gunicorn o Waitress.
    app.run(debug=True, host='0.0.0.0', port=5000)