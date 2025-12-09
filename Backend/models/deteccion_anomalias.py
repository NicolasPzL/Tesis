# ./models/deteccion_anomalias.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg') # Asegurar que se usa un backend no interactivo antes de importar pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import logging
import time
import math

# Configuración del logger mejorada
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, hidden_dim1, hidden_dim2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1), nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2), nn.ReLU(),
            nn.Linear(hidden_dim2, encoding_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim2), nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1), nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def preprocess_data(df, selected_cleaned_columns, truth_column_name):
    logger.info(f"Iniciando preprocesamiento con columnas seleccionadas: {selected_cleaned_columns}, columna de verdad: '{truth_column_name}'")
    start_time = time.time()

    if not selected_cleaned_columns:
        logger.error("No se seleccionaron columnas para el preprocesamiento.")
        raise ValueError("No columns were selected for preprocessing.")

    feature_columns_to_process = list(selected_cleaned_columns)
    cols_to_check_in_df = feature_columns_to_process + ([truth_column_name] if truth_column_name else [])
    missing_cols = [col for col in cols_to_check_in_df if col not in df.columns]
    if missing_cols:
        logger.error(f"Columnas críticas {missing_cols} no encontradas en el DataFrame. Columnas del DF: {df.columns.tolist()}")
        raise ValueError(f"Critical: Columns {missing_cols} not found in DataFrame. DF columns: {df.columns.tolist()}")

    df_subset_for_processing = df[cols_to_check_in_df].copy()
    logger.info(f"Forma del subconjunto para procesamiento: {df_subset_for_processing.shape}")

    numeric_features = df_subset_for_processing[feature_columns_to_process].select_dtypes(include=np.number).columns.tolist()
    categorical_features = df_subset_for_processing[feature_columns_to_process].select_dtypes(include=['object', 'category']).columns.tolist()
    # Asegurar que las categóricas no se solapen con numéricas si una columna fue mal interpretada inicialmente
    categorical_features = [col for col in categorical_features if col not in numeric_features]

    logger.info(f"Características numéricas identificadas: {numeric_features}")
    logger.info(f"Características categóricas identificadas: {categorical_features}")

    initial_rows = len(df_subset_for_processing)
    df_processed = df_subset_for_processing.dropna(subset=feature_columns_to_process)
    final_rows = len(df_processed)
    logger.info(f"Manejo de NaNs en columnas de características: Se eliminaron {initial_rows - final_rows} filas.")
    if final_rows == 0:
        logger.error("No hay datos después de la eliminación de NaNs de las columnas de características.")
        raise ValueError("No data after NaN removal from feature columns.")

    original_indices = df_processed.index # Guardar índices DESPUÉS de dropna

    final_feature_names_list = list(numeric_features)

    if categorical_features:
        logger.info("Codificando características categóricas...")
        for col in categorical_features:
            encoder = LabelEncoder()
            col_data = df_processed[col].astype(str).fillna('Missing_Value_In_Encoding')
            encoded_col_name = f"{col}_encoded"
            df_processed[encoded_col_name] = encoder.fit_transform(col_data)
            final_feature_names_list.append(encoded_col_name)
            logger.debug(f"Codificada '{col}' en '{encoded_col_name}'.")

    if not final_feature_names_list:
        logger.error("No hay características (numéricas o categóricas codificadas) disponibles para el escalado.")
        raise ValueError("No features (numeric or categorical) available for scaling.")

    logger.info(f"Características finales para escalado ({len(final_feature_names_list)}): {final_feature_names_list}")

    df_features_to_scale = df_processed[final_feature_names_list]
    scaler = StandardScaler()
    df_scaled_features_np = scaler.fit_transform(df_features_to_scale)
    logger.info(f"Escalado completado. Forma de las características escaladas: {df_scaled_features_np.shape}")

    preprocessing_time = time.time() - start_time
    logger.info(f"Preprocesamiento finalizado en {preprocessing_time:.2f}s.")

    # df_processed ahora contiene las columnas originales (incluyendo la de verdad), y las codificadas.
    # Los índices de df_processed están alineados con df_scaled_features_np
    return df_scaled_features_np, scaler, final_feature_names_list, original_indices, df_processed, categorical_features, numeric_features


def run_anomaly_detection(df_original_cleaned_names, selected_cleaned_columns, truth_column_name="Fraud"):
    logger.info("--- Iniciando Detección AE (PyTorch + Métricas) ---")
    overall_start_time = time.time()

    df_scaled_features_np, scaler, final_feature_names, original_indices, df_processed_with_truth_and_features, categorical_features, numeric_features = preprocess_data(
        df_original_cleaned_names,
        selected_cleaned_columns,
        truth_column_name
    )

    # Verificar que la columna de verdad exista en el DataFrame procesado que se usará para y_true_actual
    if truth_column_name not in df_processed_with_truth_and_features.columns:
        logger.error(f"La columna de verdad '{truth_column_name}' no está en el DataFrame procesado. Columnas disponibles: {df_processed_with_truth_and_features.columns.tolist()}")
        raise ValueError(f"Ground truth col '{truth_column_name}' not in processed DF. Cols: {df_processed_with_truth_and_features.columns.tolist()}")
    y_true_actual = df_processed_with_truth_and_features[truth_column_name].astype(int).values
    logger.info(f"Forma de y_true_actual: {y_true_actual.shape}. Valores únicos: {np.unique(y_true_actual, return_counts=True)}")


    input_dim = df_scaled_features_np.shape[1]
    if input_dim == 0:
        logger.error("No hay características para el entrenamiento del Autoencoder después del preprocesamiento.")
        raise ValueError("No features for AE training.")

    data_tensor = torch.tensor(df_scaled_features_np, dtype=torch.float32)
    batch_size = 128  # Puedes ajustar esto
    dataset = TensorDataset(data_tensor, data_tensor) # Para AE, input y target son los mismos

    # División train/val (opcional pero buena práctica)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    if val_size <= 0 : # Si el dataset es muy pequeño, usar todo para entrenar
        train_dataset = dataset
        val_dataset = None
        logger.info(f"Dataset muy pequeño ({len(dataset)} muestras). Usando todo para entrenamiento.")
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        logger.info(f"Dataset dividido en: Entrenamiento={train_size}, Validación={val_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoding_dim = max(1, min(8, int(input_dim / 2))) # Ajustes dinámicos para dimensiones de capas
    hidden_dim1 = max(2, min(64, int(input_dim * 0.75)))
    hidden_dim2 = max(2, min(32, int(input_dim * 0.5)))
    logger.info(f"Dimensiones del Autoencoder: input_dim={input_dim}, hidden1={hidden_dim1}, hidden2={hidden_dim2}, encoding_dim={encoding_dim}")

    model = Autoencoder(input_dim, encoding_dim, hidden_dim1, hidden_dim2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    logger.info(f"Usando dispositivo: {device}. Estructura del modelo:\n{model}")

    epochs = 50 # Puedes ajustar esto
    logger.info(f"Entrenando por {epochs} épocas...")
    # train_losses, val_losses = [], [] # No necesario para la respuesta final al cliente

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, _ in train_loader: # Target es el mismo que input
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs) # Comparar reconstrucción con original
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        # train_losses.append(epoch_train_loss)

        epoch_val_loss = float('nan')
        if val_loader:
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for inputs_val, _ in val_loader:
                    inputs_val = inputs_val.to(device)
                    outputs_val = model(inputs_val)
                    val_loss = criterion(outputs_val, inputs_val)
                    running_val_loss += val_loss.item() * inputs_val.size(0)
            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            # val_losses.append(epoch_val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f'Época [{epoch+1}/{epochs}], Pérdida Entrenamiento: {epoch_train_loss:.6f}, Pérdida Validación: {epoch_val_loss:.6f}')

    logger.info("Calculando error de reconstrucción en todo el dataset...")
    model.eval()
    all_outputs_list = []
    # Usar un DataLoader para todo el dataset (data_tensor) para la inferencia final
    full_feature_loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size*2, shuffle=False) # Batch más grande para inferencia
    with torch.no_grad():
        for features_batch in full_feature_loader: # features_batch es una tupla
            outputs = model(features_batch[0].to(device)) # Acceder al tensor de características
            all_outputs_list.append(outputs.cpu())
    reconstructed_tensor = torch.cat(all_outputs_list, dim=0)

    # Calcular MSE por muestra
    mse_loss_fn = nn.MSELoss(reduction='none') # Para obtener pérdidas por muestra
    mse_per_sample = mse_loss_fn(reconstructed_tensor, data_tensor).mean(dim=1) # Media a través de las características
    mse_numpy = mse_per_sample.detach().numpy()
    logger.info(f"Forma de mse_numpy (errores de reconstrucción): {mse_numpy.shape}")


    # Determinar umbral para anomalías
    threshold_percentile = 95
    threshold = np.percentile(mse_numpy, threshold_percentile)
    # Fallback si el percentil da un umbral no útil (ej. 0 o NaN)
    if np.isnan(threshold) or threshold <= 1e-9: # Umbral muy pequeño o NaN
        mean_mse = np.mean(mse_numpy)
        std_mse = np.std(mse_numpy)
        if std_mse > 1e-9: # Evitar división por cero o std muy pequeño
            threshold = mean_mse + 3 * std_mse
        else: # Si std es muy pequeño, usar media más un pequeño epsilon o un valor por defecto
            threshold = mean_mse + 1e-6
        logger.warning(f"Percentil {threshold_percentile} dio un umbral no útil. Usando fallback: media + 3*std = {threshold:.6f}")
    if threshold <= 1e-9: # Último fallback si todo lo demás falla
        threshold = 1e-6 # Un pequeño valor por defecto
        logger.warning(f"Umbral sigue siendo muy bajo. Establecido a {threshold:.6f} por defecto.")

    logger.info(f"Umbral de anomalía (MSE en percentil {threshold_percentile}): {threshold:.6f}")

    y_pred_model = (mse_numpy > threshold).astype(int)
    logger.info(f"Forma de y_pred_model: {y_pred_model.shape}. Valores únicos: {np.unique(y_pred_model, return_counts=True)}")


    logger.info("Calculando métricas de rendimiento...")
    precision, recall, f1 = 0.0, 0.0, 0.0
    report_str = "Métricas N/A (longitudes de y_true y y_pred no coinciden o datos insuficientes)"
    cm_array = np.zeros((2,2), dtype=int)

    if len(y_true_actual) == len(y_pred_model) and len(y_true_actual) > 0:
        try:
            precision = precision_score(y_true_actual, y_pred_model, pos_label=1, zero_division=0)
            recall = recall_score(y_true_actual, y_pred_model, pos_label=1, zero_division=0)
            f1 = f1_score(y_true_actual, y_pred_model, pos_label=1, zero_division=0)
            report_str = classification_report(y_true_actual, y_pred_model, target_names=['Normal', 'Fraude'], zero_division=0, output_dict=False)
            cm_array = confusion_matrix(y_true_actual, y_pred_model, labels=[0, 1]) # Asegurar etiquetas
            if cm_array.shape != (2,2): # Si solo hay una clase en y_true o y_pred, ajustar
                logger.warning(f"Matriz de confusión no es 2x2 ({cm_array.shape}). Ajustando...")
                new_cm = np.zeros((2,2), dtype=int)
                if len(np.unique(y_true_actual)) == 1 or len(np.unique(y_pred_model)) == 1:
                    # Caso simple: rellenar basado en lo que es cm_array
                    # Esto es una simplificación, puede necesitar lógica más robusta si esto ocurre a menudo
                    if cm_array.shape == (1,1):
                        unique_label = np.unique(y_true_actual)[0] # Asumimos que este es el label
                        if unique_label == 0: new_cm[0,0] = cm_array[0,0]
                        else: new_cm[1,1] = cm_array[0,0]
                    # ... más lógica podría ser necesaria para otros tamaños de cm_array
                cm_array = new_cm

            logger.info(f"Métricas - Precisión(Fraude): {precision:.4f}, Recall(Fraude): {recall:.4f}, F1(Fraude): {f1:.4f}")
            logger.info(f"Reporte de Clasificación:\n{report_str}")
            logger.info(f"Matriz de Confusión:\n{cm_array}")
        except Exception as e:
            logger.error(f"Error calculando métricas: {e}", exc_info=True)
            report_str = f"Error calculando métricas: {e}"

    else:
        logger.error(f"Discrepancia de longitud para cálculo de métricas: y_true_actual ({len(y_true_actual)}), y_pred_model ({len(y_pred_model)})")


    # df_result_final es df_processed_with_truth_and_features que tiene las columnas originales (incluida la de verdad)
    # y las características codificadas. Ya está alineado con mse_numpy y y_pred_model por original_indices.
    df_result_final = df_processed_with_truth_and_features.copy() # df_processed_with_truth_and_features ya tiene los índices correctos
    df_result_final['anomaly_score_mse'] = mse_numpy
    df_result_final['is_anomaly_pred'] = y_pred_model

    anomalies_pred_df = df_result_final[df_result_final['is_anomaly_pred'] == 1].copy()
    anomaly_count_pred = len(anomalies_pred_df)
    total_records_processed = len(df_result_final) # Basado en filas después de preprocesamiento y dropna
    anomaly_percentage_pred = (anomaly_count_pred / total_records_processed * 100) if total_records_processed > 0 else 0
    normal_count_pred = total_records_processed - anomaly_count_pred

    anomalies_pred_df.sort_values(by='anomaly_score_mse', ascending=False, inplace=True)
    sample_anomalies = anomalies_pred_df.head(10)
    sample_anomalies_json = []
    if not sample_anomalies.empty:
        try:
            # Asegurar que 'original_index' (si no es el nombre del índice) esté como columna para JSON
            if sample_anomalies.index.name != 'original_index':
                 sample_anomalies_copy = sample_anomalies.reset_index().rename(columns={'index': 'original_index_from_df'})
            else:
                 sample_anomalies_copy = sample_anomalies.reset_index() # Si el índice se llama 'original_index'

            # Manejo de tipos para serialización JSON
            for col in sample_anomalies_copy.columns:
                if pd.api.types.is_datetime64_any_dtype(sample_anomalies_copy[col]):
                    sample_anomalies_copy[col] = sample_anomalies_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S').fillna('NaT')
                elif pd.api.types.is_timedelta64_dtype(sample_anomalies_copy[col]):
                    sample_anomalies_copy[col] = sample_anomalies_copy[col].astype(str).fillna('NaT')
                elif pd.api.types.is_float_dtype(sample_anomalies_copy[col]):
                    if col not in ['original_index_from_df', 'anomaly_score_mse', 'original_index']: # no redondear índices
                        sample_anomalies_copy[col] = sample_anomalies_copy[col].round(4)
                    elif col == 'anomaly_score_mse':
                        sample_anomalies_copy[col] = sample_anomalies_copy[col].round(6)
                # Convertir booleanos a int (0 o 1)
                elif pd.api.types.is_bool_dtype(sample_anomalies_copy[col]):
                     sample_anomalies_copy[col] = sample_anomalies_copy[col].astype(int)


            sample_anomalies_json = sample_anomalies_copy.replace([np.inf, -np.inf], None).where(pd.notnull(sample_anomalies_copy), None).to_dict(orient='records')
        except Exception as json_err:
            logger.error(f"Error serializando muestras de anomalías a JSON: {json_err}", exc_info=True)
            sample_anomalies_json = [{"error": "Serialization_failed", "details": str(json_err)}]


    # Seleccionar características para graficar (usar las originales, no las codificadas ni escaladas)
    # Estas son las columnas que el usuario seleccionó, y existen en df_result_final.
    possible_numerical_for_plot = [col for col in selected_cleaned_columns if col in df_result_final.columns and pd.api.types.is_numeric_dtype(df_result_final[col]) and col not in [truth_column_name, 'anomaly_score_mse', 'is_anomaly_pred']]
    first_numerical_plot = next(iter(possible_numerical_for_plot), None)

    possible_categorical_for_plot = [col for col in selected_cleaned_columns if col in df_result_final.columns and not pd.api.types.is_numeric_dtype(df_result_final[col]) and col != truth_column_name]
    first_categorical_plot = next(iter(possible_categorical_for_plot), None)

    logger.info(f"Característica numérica seleccionada para graficar: {first_numerical_plot}")
    logger.info(f"Característica categórica seleccionada para graficar: {first_categorical_plot}")

    results_data = {
        "total_records_analyzed": int(total_records_processed),
        "anomaly_threshold": float(threshold) if not np.isnan(threshold) else 0.0,
        "detected_anomalies_count": int(anomaly_count_pred),
        "detected_anomalies_percentage": round(anomaly_percentage_pred, 2),
        "columns_analyzed": selected_cleaned_columns,
        "features_used_in_model": final_feature_names,
        "sample_anomalous_records": sample_anomalies_json,
        "plotted_numerical_feature": first_numerical_plot,
        "plotted_categorical_feature": first_categorical_plot,
        "evaluation_metrics": {
            "precision_fraud": round(precision, 4),
            "recall_fraud": round(recall, 4),
            "f1_score_fraud": round(f1, 4),
            "classification_report": report_str, # Ya es un string
            "confusion_matrix": cm_array.tolist() if isinstance(cm_array, np.ndarray) else cm_array
        }
    }

    plots_base64 = {}
    plt.style.use('seaborn-v0_8-whitegrid')
    plots_start_time = time.time()

    # --- Gráfico 1: Resumen Anomalías PREDICHAS (Pie Chart) ---
    # Clave: prediction_summary_pie
    fig1 = None
    try:
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        labels = 'Normal Predicho', 'Anomalía Predicha'
        sizes = [normal_count_pred, anomaly_count_pred]
        colors = ['#66b3ff','#ff9999']
        explode = (0, 0.1) if anomaly_count_pred > 0 and normal_count_pred > 0 else (0,0) # Evitar explode si una categoría es 0
        if sum(sizes) > 0: # Solo graficar si hay datos
            ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 10})
            ax1.axis('equal')
            plt.title('Resumen de Predicciones del Modelo', fontsize=12)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plots_base64['prediction_summary_pie'] = base64.b64encode(buf.read()).decode('utf-8')
        else:
            logger.warning("No se generó el gráfico 'prediction_summary_pie' porque no hay datos (normal_count_pred y anomaly_count_pred son cero).")
            plots_base64['prediction_summary_pie'] = None # O una imagen placeholder en base64
    except Exception as e:
        logger.error(f"Error generando 'prediction_summary_pie': {e}", exc_info=True)
        plots_base64['prediction_summary_pie'] = None
    finally:
        if fig1: plt.close(fig1)

    # --- Gráfico 2: Top Categorías en Anomalías PREDICHAS ---
    # Clave: top_predicted_anomaly_categories
    fig2 = None
    if first_categorical_plot and not anomalies_pred_df.empty and first_categorical_plot in anomalies_pred_df.columns:
        try:
            top_n = 10
            # Asegurar que la columna es tratada como string para value_counts
            anomaly_cat_counts = anomalies_pred_df[first_categorical_plot].astype(str).value_counts().nlargest(top_n)
            if not anomaly_cat_counts.empty:
                fig2 = plt.figure(figsize=(10, max(5, len(anomaly_cat_counts) * 0.5))) # Ajustar altura dinámicamente
                sns.barplot(x=anomaly_cat_counts.values, y=anomaly_cat_counts.index, palette='viridis', orient='h')
                title_cat_plot = f'Top {len(anomaly_cat_counts)} "{first_categorical_plot.replace("_"," ").title()}" en Anomalías Predichas'
                plt.title(title_cat_plot, fontsize=12)
                plt.xlabel('Número de Registros Anómalos Predichos'); plt.ylabel(f'{first_categorical_plot.replace("_"," ").title()}');
                plt.xticks(fontsize=9); plt.yticks(fontsize=9);
                for idx, val in enumerate(anomaly_cat_counts.values):
                    plt.text(val + (plt.xlim()[1]*0.01), idx, str(val), va='center', fontsize=9)
                plt.tight_layout()
                buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
                plots_base64['top_predicted_anomaly_categories'] = base64.b64encode(buf.read()).decode('utf-8')
            else:
                logger.warning(f"No se generó 'top_predicted_anomaly_categories' para '{first_categorical_plot}' porque no hay cuentas de categorías anómalas.")
                plots_base64['top_predicted_anomaly_categories'] = None
        except Exception as e:
            logger.error(f"Error generando 'top_predicted_anomaly_categories' para '{first_categorical_plot}': {e}", exc_info=True)
            plots_base64['top_predicted_anomaly_categories'] = None
        finally:
            if fig2: plt.close(fig2)
    else:
        logger.info(f"No se generará 'top_predicted_anomaly_categories'. Razón: first_categorical_plot='{first_categorical_plot}', anomalies_pred_df.empty={anomalies_pred_df.empty}")
        plots_base64['top_predicted_anomaly_categories'] = None


    # --- Gráfico 3: Rango Numérico en Anomalías PREDICHAS ---
    # Clave: predicted_anomaly_numerical_range
    fig3 = None
    if first_numerical_plot and not anomalies_pred_df.empty and first_numerical_plot in anomalies_pred_df.columns:
        try:
            # Verificar que haya varianza en los datos para el boxenplot
            if anomalies_pred_df[first_numerical_plot].nunique() > 1:
                fig3 = plt.figure(figsize=(8, 5))
                sns.boxenplot(x=anomalies_pred_df[first_numerical_plot].dropna(), color='#ff9999', width=0.3) # .dropna() por si acaso
                title_num_plot = f'Rango de "{first_numerical_plot.replace("_"," ").title()}" en Anomalías Predichas'
                plt.title(title_num_plot, fontsize=12)
                plt.xlabel(f'Valor de {first_numerical_plot.replace("_"," ").title()}'); plt.xticks(fontsize=9);
                plt.grid(axis='x', linestyle='--', alpha=0.7); plt.tight_layout();
                buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
                plots_base64['predicted_anomaly_numerical_range'] = base64.b64encode(buf.read()).decode('utf-8')
            else:
                logger.warning(f"No se generó 'predicted_anomaly_numerical_range' para '{first_numerical_plot}' debido a datos insuficientes o sin varianza en las anomalías.")
                plots_base64['predicted_anomaly_numerical_range'] = None
        except Exception as e:
            logger.error(f"Error generando 'predicted_anomaly_numerical_range' para '{first_numerical_plot}': {e}", exc_info=True)
            plots_base64['predicted_anomaly_numerical_range'] = None
        finally:
            if fig3: plt.close(fig3)
    else:
        logger.info(f"No se generará 'predicted_anomaly_numerical_range'. Razón: first_numerical_plot='{first_numerical_plot}', anomalies_pred_df.empty={anomalies_pred_df.empty}")
        plots_base64['predicted_anomaly_numerical_range'] = None

    # --- Gráfico 4: Severidad de Anomalías PREDICHAS ---
    # Clave: predicted_anomaly_score_severity
    fig4 = None
    if not anomalies_pred_df.empty and 'anomaly_score_mse' in anomalies_pred_df.columns:
        try:
            anomaly_scores = anomalies_pred_df['anomaly_score_mse'].dropna()
            if len(anomaly_scores) >= 3: # Necesitamos al menos algunos puntos para los percentiles
                q50 = np.percentile(anomaly_scores, 50)
                q90 = np.percentile(anomaly_scores, 90)
                # Asegurar que los bins sean únicos y ordenados
                bins = sorted(list(set([-np.inf, q50, q90, np.inf])))
                if len(bins) < 3 : # Si q50 y q90 son iguales, o muy cercanos a inf
                    labels = ['Anomalías Detectadas']
                    bins = [-np.inf, np.inf]
                elif len(bins) == 3: # p.ej. q50 == q90
                     labels = ['Severidad Media/Baja', 'Severidad Alta']
                else: # len(bins) == 4
                     labels = ['Severidad Baja', 'Severidad Media', 'Severidad Alta']

                severity_series = pd.cut(anomaly_scores, bins=bins, labels=labels, right=True, include_lowest=True)
                severity_counts = severity_series.value_counts().reindex(labels).fillna(0)

                if not severity_counts.empty and severity_counts.sum() > 0:
                    fig4 = plt.figure(figsize=(7, 5))
                    sns.barplot(x=severity_counts.index, y=severity_counts.values, palette='coolwarm')
                    plt.title('Nivel de Severidad de Anomalías Predichas', fontsize=12)
                    plt.xlabel('Nivel de Severidad'); plt.ylabel('Número de Registros Anómalos Predichos');
                    plt.xticks(fontsize=9); plt.yticks(fontsize=9);
                    for i, v_val in enumerate(severity_counts.values):
                        plt.text(i, v_val + (plt.ylim()[1] * 0.01), str(int(v_val)), ha='center', va='bottom', fontsize=9)
                    plt.tight_layout(); buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
                    plots_base64['predicted_anomaly_score_severity'] = base64.b64encode(buf.read()).decode('utf-8')
                else:
                    logger.warning("No se generó 'predicted_anomaly_score_severity' porque no hay cuentas de severidad.")
                    plots_base64['predicted_anomaly_score_severity'] = None
            else:
                logger.warning(f"No se generó 'predicted_anomaly_score_severity' debido a insuficientes ({len(anomaly_scores)}) puntajes de anomalía.")
                plots_base64['predicted_anomaly_score_severity'] = None
        except Exception as e:
            logger.error(f"Error generando 'predicted_anomaly_score_severity': {e}", exc_info=True)
            plots_base64['predicted_anomaly_score_severity'] = None
        finally:
            if fig4: plt.close(fig4)
    else:
        logger.info(f"No se generará 'predicted_anomaly_score_severity'. Razón: anomalies_pred_df.empty={anomalies_pred_df.empty} o 'anomaly_score_mse' no existe.")
        plots_base64['predicted_anomaly_score_severity'] = None


    # --- Gráfico 5: Matriz de Confusión ---
    # Clave: confusion_matrix
    fig5 = None
    try:
        # Asegurar que cm_array sea una matriz numpy 2x2
        if isinstance(cm_array, list): cm_array = np.array(cm_array)
        if cm_array.shape == (2,2):
            fig5 = plt.figure(figsize=(6, 5))
            sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Pred. Normal', 'Pred. Fraude'],
                        yticklabels=['Real Normal', 'Real Fraude'],
                        annot_kws={"size": 10}) # Tamaño de la anotación
            plt.xlabel('Predicción del Modelo', fontsize=10); plt.ylabel(f'Etiqueta Real ("{truth_column_name}")', fontsize=10)
            plt.title('Rendimiento del Modelo: Matriz de Confusión', fontsize=12); plt.tight_layout()
            buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
            plots_base64['confusion_matrix'] = base64.b64encode(buf.read()).decode('utf-8')
        else:
            logger.warning(f"No se generó el gráfico 'confusion_matrix' porque cm_array no tiene forma (2,2). Forma actual: {cm_array.shape}")
            plots_base64['confusion_matrix'] = None
    except Exception as e:
        logger.error(f"Error generando 'confusion_matrix': {e}", exc_info=True)
        plots_base64['confusion_matrix'] = None
    finally:
        if fig5: plt.close(fig5)

    plots_generation_time = time.time() - plots_start_time
    logger.info(f"Generación de gráficos completada en {plots_generation_time:.2f}s.")

    # Log de depuración final para las claves y snippets de base64
    logger.info("--- Información de Gráficos Base64 Generados ---")
    for key, value in plots_base64.items():
        if value:
            logger.debug(f"Clave del Gráfico: '{key}', Longitud Base64: {len(value)}, Snippet: {value[:80]}...")
        else:
            logger.debug(f"Clave del Gráfico: '{key}', Valor: None (No generado o error)")
    logger.info(f"--- Detección AE Finalizada en {time.time() - overall_start_time:.2f}s ---")

    return results_data, plots_base64