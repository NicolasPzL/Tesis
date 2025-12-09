# ./utils/procesador_archivo.py
import pandas as pd
import os
import logging
import unicodedata # Para normalizar y quitar tildes
import re # Para limpieza más avanzada de nombres

logger = logging.getLogger(__name__)

def clean_column_name(col_name):
    """
    Limpia y estandariza un nombre de columna de forma robusta.
    """
    if not isinstance(col_name, str):
        col_name = str(col_name)
    
    normalized_name = unicodedata.normalize('NFD', col_name)
    ascii_name = "".join(c for c in normalized_name if unicodedata.category(c) != 'Mn' and ord(c) < 128)
    
    cleaned_name = ascii_name.strip().lower()
    cleaned_name = re.sub(r'[^a-z0-9_]+', '_', cleaned_name)
    cleaned_name = re.sub(r'_{2,}', '_', cleaned_name)
    cleaned_name = cleaned_name.strip('_')
    
    if not cleaned_name or len(cleaned_name) < 1:
        return f"col_{abs(hash(col_name)) % 100000}" 
    return cleaned_name


def process_file(filepath):
    """
    Procesa archivo CSV/Excel. Devuelve:
    1. DataFrame con nombres de columna limpios.
    2. Lista de nombres de columna originales.
    3. Diccionario de mapeo {original_name: cleaned_name}.
    """
    logger.info(f"Processing file: {filepath}")
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")

    file_extension = filepath.split('.')[-1].lower()
    df = None

    try:
        if file_extension == 'csv':
            logger.info("Attempting to read as CSV")
            try: df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
            except UnicodeDecodeError:
                logger.warning("UTF-8 failed, trying latin1...")
                try: df = pd.read_csv(filepath, encoding='latin1', low_memory=False)
                except UnicodeDecodeError:
                    logger.warning("Latin1 failed, trying iso-8859-1...")
                    df = pd.read_csv(filepath, encoding='iso-8859-1', low_memory=False)
                except pd.errors.ParserError:
                    logger.warning("Latin1 failed parsing, trying utf-8 with semicolon...")
                    df = pd.read_csv(filepath, encoding='utf-8', delimiter=';', low_memory=False)
            except pd.errors.ParserError:
                logger.warning("UTF-8 comma failed parsing, trying utf-8 with semicolon...")
                df = pd.read_csv(filepath, encoding='utf-8', delimiter=';', low_memory=False)
            except Exception as e_csv:
                logger.error(f"Unhandled error reading CSV {filepath}: {e_csv}", exc_info=True)
                raise ValueError(f"Could not parse CSV file: {e_csv}")
        elif file_extension in ['xlsx', 'xls']:
            logger.info("Attempting to read as Excel")
            df = pd.read_excel(filepath, engine=None)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        if df is None or df.empty:
             raise ValueError("Could not read the file, or the file is empty after reading.")
        logger.info(f"File read successfully. Initial shape: {df.shape}")

        original_columns = df.columns.tolist()
        logger.info(f"Original columns from file: {original_columns}")

        df.dropna(axis=0, how='all', inplace=True)
        if df.empty: raise ValueError("File is empty after dropping all-NaN rows.")
        
        rows_before_duplicates = len(df)
        df.drop_duplicates(inplace=True)
        if len(df) < rows_before_duplicates:
            logger.info(f"Removed {rows_before_duplicates - len(df)} duplicate rows.")
        if df.empty: raise ValueError("File is empty after dropping duplicate rows.")
        
        cleaned_columns_final_list = []
        column_mapping = {}
        from collections import Counter
        cleaned_name_counts = Counter()

        base_cleaned_names = [clean_column_name(col) for col in original_columns]
        
        for i, original_col_name in enumerate(original_columns):
            base_name = base_cleaned_names[i]
            current_count = cleaned_name_counts[base_name]
            final_cleaned_name = base_name
            if current_count > 0:
                final_cleaned_name = f"{base_name}_{current_count}"
            
            while final_cleaned_name in cleaned_columns_final_list:
                current_count += 1
                final_cleaned_name = f"{base_name}_{current_count}"
            
            cleaned_name_counts[base_name] = current_count + 1
            cleaned_columns_final_list.append(final_cleaned_name)
            column_mapping[str(original_col_name)] = final_cleaned_name
        
        df.columns = cleaned_columns_final_list
        logger.info(f"Final cleaned DataFrame columns: {df.columns.tolist()}")

        return df, original_columns, column_mapping

    except FileNotFoundError: raise
    except ValueError as ve:
        logger.error(f"ValueError during file processing for {filepath}: {ve}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error processing file {filepath}")
        raise RuntimeError(f"An unexpected error occurred processing file: {e}") from e