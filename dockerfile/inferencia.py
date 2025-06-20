import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import json

from preprocessing import RegionClusterTransformer,DateTransformer,RainTodayTomorrowEncoder,CustomImputer,WindDirectionTransformer,OutlierTreatmentTransformer,FeatureEngineeringTransformer,WeatherPreprocessingPipeline

def cargar_modelo_y_pipeline():
    """
    Carga el pipeline de preprocesamiento serializado y el modelo TensorFlow SavedModel.
    
    Returns:
        tuple: (pipeline, modelo) - Pipeline sklearn y modelo TensorFlow cargados
    """
    # Deserialización del pipeline de sklearn
    with open('weather_preprocessing_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    
    # Carga del modelo TensorFlow en formato SavedModel
    modelo = tf.saved_model.load('modelo_tensorflow')
    
    return pipeline, modelo

def realizar_inferencia(datos_entrada, pipeline, modelo):
    """
    Ejecuta el pipeline completo de inferencia: preprocesamiento + predicción.
    
    Args:
        datos_entrada (pd.DataFrame): Datos de entrada sin procesar
        pipeline: Pipeline de preprocesamiento sklearn
        modelo: Modelo TensorFlow cargado
        
    Returns:
        tf.Tensor: Predicciones del modelo
    """
    # Aplicación del pipeline de preprocesamiento
    datos_procesados = pipeline.transform(datos_entrada)
    
    # Conversión a tensor TensorFlow con dtype float64
    if not isinstance(datos_procesados, tf.Tensor):
        datos_procesados = tf.constant(datos_procesados, dtype=tf.float64)
    else:
        datos_procesados = tf.cast(datos_procesados, tf.float64)
    
    # Tratamiento de valores NaN - reemplazo por ceros
    datos_procesados = tf.where(tf.math.is_nan(datos_procesados), 
                               tf.zeros_like(datos_procesados), 
                               datos_procesados)
    
    # Ejecución de la función de inferencia del modelo
    prediccion_func = modelo.signatures['serving_default']
    prediccion = prediccion_func(dense_input=datos_procesados)
    
    # Extracción del tensor de salida desde el diccionario de resultados
    if isinstance(prediccion, dict):
        # Búsqueda por claves estándar de salida
        for key in ['dense_3', 'output_0', 'predictions']:
            if key in prediccion:
                return prediccion[key]
        # Fallback: retorna el primer valor disponible
        return list(prediccion.values())[0]
    
    return prediccion

def inspect_model_signature(modelo):
    """
    Utilidad de debugging para examinar la estructura del modelo SavedModel.
    
    Args:
        modelo: Modelo TensorFlow cargado
    """
    print("=== INFORMACIÓN DEL MODELO ===")
    print("Signaturas disponibles:")
    for name, signature in modelo.signatures.items():
        print(f"  - {name}")
        print(f"    Inputs: {signature.inputs}")
        print(f"    Outputs: {signature.outputs}")
    print("=" * 30)

def main():
    """
    Función principal que maneja la interfaz CLI y orquesta el proceso de inferencia.
    """
    parser = argparse.ArgumentParser(description='Sistema de inferencia para modelo de predicción climática')
    parser.add_argument('--input', required=True, help='Ruta del archivo CSV con datos de entrada')
    parser.add_argument('--output', default='predicciones.json', help='Ruta del archivo de salida JSON')
    parser.add_argument('--inspect', action='store_true', help='Activar inspección de signatura del modelo')
    
    args = parser.parse_args()
    
    try:
        # Inicialización del sistema
        print("Cargando modelo y pipeline...")
        pipeline, modelo = cargar_modelo_y_pipeline()
        
        # Modo debugging opcional
        if args.inspect:
            inspect_model_signature(modelo)
        
        # Carga de datos de entrada
        print(f"Leyendo datos de entrada desde {args.input}...")
        datos = pd.read_csv(args.input)
        print(f"Datos cargados: {datos.shape}")
        
        # Proceso de inferencia
        print("Realizando inferencia...")
        predicciones = realizar_inferencia(datos, pipeline, modelo)
        
        # Conversión de tensor a numpy array para serialización
        if hasattr(predicciones, 'numpy'):
            predicciones = predicciones.numpy()
        
        print(f"Predicciones generadas: {predicciones}")
        print(f"Forma: {predicciones.shape if hasattr(predicciones, 'shape') else 'N/A'}")
        
        # Serialización de resultados en formato JSON
        resultados = {
            'predicciones': predicciones.tolist(),
            'numero_muestras': len(predicciones),
            'forma_predicciones': list(predicciones.shape) if hasattr(predicciones, 'shape') else None
        }
        
        with open(args.output, 'w') as f:
            json.dump(resultados, f, indent=2)
        
        print(f"Inferencia completada. Resultados guardados en {args.output}")
        
    except Exception as e:
        # Manejo de errores con información detallada para debugging
        print(f"Error durante la inferencia: {str(e)}")
        print(f"Tipo de error: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()