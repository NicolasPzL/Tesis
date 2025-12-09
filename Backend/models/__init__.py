# ./models/__init__.py
# Exportar solo la función principal que usa la API
from .deteccion_anomalias import run_anomaly_detection

__all__ = [
    'run_anomaly_detection'
]