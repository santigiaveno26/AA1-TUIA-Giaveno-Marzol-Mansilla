import pandas as pd
import numpy as np
import pickle
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import time
import warnings
warnings.filterwarnings('ignore')

class RegionClusterTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador que agrupa locaciones geográficas en regiones mediante clustering K-means.
    Utiliza coordenadas geográficas predefinidas para crear 6 regiones climáticas.
    """
    def __init__(self):
        self.kmeans = None
        self.coordenadas_df = None
        self.location_to_region = {}
        
    def _get_coordinates(self, locations):
        """
        Obtiene coordenadas geográficas para las locaciones utilizando un diccionario predefinido.
        
        Args:
            locations: Lista de nombres de locaciones
            
        Returns:
            DataFrame con columnas Location, latitud, longitud
        """
        # Diccionario de coordenadas geográficas para locaciones australianas
        coordinates_data = {
            'Albury': (-36.073773, 146.913526),
            'BadgerysCreek': (-33.883145, 150.742466),
            'Cobar': (-31.966663, 145.304505),
            'CoffsHarbour': (-32.404454, 115.766609),
            'Moree': (-29.461720, 149.840715),
            'Newcastle': (-32.919295, 151.779535),
            'NorahHead': (-33.281667, 151.567778),
            'Penrith': (-33.751195, 150.694171),
            'Richmond': (-37.807450, 144.990721),
            'Sydney': (-33.869844, 151.208285),
            'SydneyAirport': (-33.869844, 151.208285),
            'WaggaWagga': (-28.335487, 116.938870),
            'Williamtown': (-32.815000, 151.842778),
            'Wollongong': (-34.424394, 150.893850),
            'Canberra': (-35.297591, 149.101268),
            'Tuggeranong': (-35.420977, 149.092134),
            'MountGinini': (-35.529744, 148.772540),
            'Ballarat': (-37.562301, 143.860565),
            'Bendigo': (-36.759018, 144.282672),
            'Sale': (-38.109446, 147.065672),
            'MelbourneAirport': (-37.814245, 144.963173),
            'Melbourne': (-37.814245, 144.963173),
            'Mildura': (-34.195274, 142.150315),
            'Nhil': (-35.432540, 141.283386),
            'Portland': (-38.345623, 141.604230),
            'Watsonia': (-37.710947, 145.083781),
            'Dartmoor': (-37.895212, 141.267943),
            'Brisbane': (-27.468962, 153.023501),
            'Cairns': (-16.920666, 145.772185),
            'GoldCoast': (-28.002373, 153.414599),
            'Townsville': (-19.256939, 146.823954),
            'Adelaide': (-34.928181, 138.599931),
            'MountGambier': (-37.830139, 140.784263),
            'Nuriootpa': (-34.469335, 138.993901),
            'Woomera': (-31.199914, 136.825353),
            'Albany': (-35.024782, 117.883608),
            'Witchcliffe': (-34.026335, 115.100477),
            'PearceRAAF': (-31.663737, 116.027266),
            'PerthAirport': (-31.941521, 115.965577),
            'Perth': (-31.955897, 115.860578),
            'SalmonGums': (-32.981517, 121.644079),
            'Walpole': (-34.977680, 116.731006),
            'Hobart': (-42.882509, 147.328123),
            'Launceston': (-41.434081, 147.137350),
            'AliceSprings': (-23.698388, 133.881289),
            'Darwin': (-12.460440, 130.841047),
            'Katherine': (-14.464616, 132.263599),
            'Uluru': (-25.345554, 131.036961),
            'NorfolkIsland': (-29.033300, 167.950000)
        }
        
        data_coordenadas = []
        
        for location in locations:
            if location in coordinates_data:
                lat, lon = coordinates_data[location]
                data_coordenadas.append({
                    "Location": location,
                    "latitud": lat,
                    "longitud": lon
                })
            else:
                # Coordenadas del centro geográfico de Australia para locaciones no encontradas
                data_coordenadas.append({
                    "Location": location,
                    "latitud": -25.2744,
                    "longitud": 133.7751
                })
        
        coordenadas_df = pd.DataFrame(data_coordenadas)
        coordenadas_df.dropna(inplace=True)
        
        return coordenadas_df
    
    def fit(self, X, y=None):
        """
        Entrena el algoritmo K-means para crear regiones geográficas.
        """
        locations = X['Location'].unique()
        self.coordenadas_df = self._get_coordinates(locations)
        
        # Clustering K-means con 6 clusters para regiones climáticas
        self.kmeans = KMeans(n_clusters=6, random_state=42)
        self.coordenadas_df['region'] = self.kmeans.fit_predict(
            self.coordenadas_df[['latitud', 'longitud']]
        )
        
        # Mapeo de locación a región
        self.location_to_region = dict(
            zip(self.coordenadas_df['Location'], self.coordenadas_df['region'])
        )
        
        return self
    
    def transform(self, X):
        """
        Asigna región a cada observación basada en la locación.
        """
        X_copy = X.copy()
        X_copy['region'] = X_copy['Location'].map(self.location_to_region)
        # Región 0 por defecto para locaciones no mapeadas
        X_copy['region'] = X_copy['region'].fillna(0)
        return X_copy

class DateTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador que extrae características temporales de la fecha.
    Convierte fechas en mes y estación del año (hemisferio sur).
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        Extrae mes y estación del año de la columna Date.
        """
        X_copy = X.copy()
        X_copy['Date'] = pd.to_datetime(X_copy['Date'])
        X_copy['Month'] = X_copy['Date'].dt.month
        
        def asignar_estacion(mes):
            """Mapeo de mes a estación para hemisferio sur"""
            if mes in [12, 1, 2]:
                return 0  # Verano
            elif mes in [3, 4, 5]:
                return 1  # Otoño
            elif mes in [6, 7, 8]:
                return 2  # Invierno
            elif mes in [9, 10, 11]:
                return 3  # Primavera
        
        X_copy['Season'] = X_copy['Month'].apply(asignar_estacion)
        return X_copy

class RainTodayTomorrowEncoder(BaseEstimator, TransformerMixin):
    """
    Codificador para variables categóricas de lluvia.
    Convierte 'Yes'/'No' a valores binarios 1/0.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        Codifica variables de lluvia categóricas a numéricas.
        """
        X_copy = X.copy()
        X_copy['RainToday'] = X_copy['RainToday'].map({'No': 0, 'Yes': 1})
        if 'RainTomorrow' in X_copy.columns:
            X_copy['RainTomorrow'] = X_copy['RainTomorrow'].map({'No': 0, 'Yes': 1})
        return X_copy

class CustomImputer(BaseEstimator, TransformerMixin):
    """
    Imputador personalizado que utiliza diferentes estrategias según el tipo de variable:
    - Variables numéricas: media/mediana por región y mes
    - Variables categóricas: moda por región
    - Variables de nubosidad: distribución probabilística por región y mes
    """
    def __init__(self):
        self.medianas_train = {}
        self.medias_train = {}
        self.modas_train = {}
        self.cloud_distributions = {}
        
    def fit(self, X, y=None):
        """
        Calcula estadísticos de imputación basados en agrupaciones geográficas y temporales.
        """
        X_copy = X.copy()
        
        # Ordenamiento para procesamiento secuencial
        X_copy = X_copy.sort_values(['region', 'Date'])
        
        # Variables para imputación con mediana por región y mes
        variables_mediana = ['Sunshine', 'Evaporation']
        for var in variables_mediana:
            if var in X_copy.columns:
                self.medianas_train[var] = X_copy.groupby(['region', 'Month'])[var].median()
        
        # Variables para imputación con media por región y mes
        variables_media = [
            'MaxTemp', 'MinTemp', 'WindGustSpeed', 'Rainfall', 'Pressure9am', 
            'Pressure3pm', 'Temp9am', 'Temp3pm', 'WindSpeed3pm', 'Humidity9am', 
            'WindSpeed9am', 'Humidity3pm'
        ]
        for var in variables_media:
            if var in X_copy.columns:
                self.medias_train[var] = X_copy.groupby(['region', 'Month'])[var].mean()
        
        # Variables categóricas para imputación con moda por región
        variables_categoricas = ['WindDir9am', 'WindGustDir', 'WindDir3pm']
        for var in variables_categoricas:
            if var in X_copy.columns:
                self.modas_train[var] = X_copy.groupby('region')[var].agg(
                    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
                )
        
        # Distribuciones probabilísticas para variables de nubosidad
        cloud_vars = ['Cloud3pm', 'Cloud9am']
        for var in cloud_vars:
            if var in X_copy.columns:
                distribuciones = {}
                for keys, group_df in X_copy.groupby(['region', 'Month']):
                    valores = group_df[var].dropna()
                    if not valores.empty:
                        probs = valores.value_counts(normalize=True)
                        distribuciones[keys] = probs
                self.cloud_distributions[var] = distribuciones
        
        return self
    
    def transform(self, X):
        """
        Aplica las estrategias de imputación calculadas durante el fit.
        """
        X_copy = X.copy()
        
        # Imputación con mediana por grupos
        for var, medianas in self.medianas_train.items():
            if var in X_copy.columns:
                X_copy[var] = X_copy.groupby(['region', 'Month'])[var].transform(
                    lambda x: x.fillna(x.median())
                )
        
        # Imputación con media por grupos
        for var, medias in self.medias_train.items():
            if var in X_copy.columns:
                X_copy[var] = X_copy.groupby(['region', 'Month'])[var].transform(
                    lambda x: x.fillna(x.mean())
                )
        
        # Imputación categórica con moda por región
        for var, modas in self.modas_train.items():
            if var in X_copy.columns:
                X_copy[var] = X_copy.apply(
                    lambda row: modas.get(row['region'], np.nan) if pd.isna(row[var]) else row[var],
                    axis=1
                )
        
        # Imputación probabilística para nubosidad
        for var, distribuciones in self.cloud_distributions.items():
            if var in X_copy.columns:
                def imputar_fila(fila):
                    key = (fila['region'], fila['Month'])
                    if pd.isna(fila[var]):
                        if key in distribuciones:
                            dist = distribuciones[key]
                            return np.random.choice(dist.index, p=dist.values)
                    return fila[var]
                
                X_copy[var] = X_copy.apply(imputar_fila, axis=1)
        
        # Eliminación de filas con RainToday nulo (variable crítica)
        if 'RainToday' in X_copy.columns:
            X_copy = X_copy.dropna(subset=['RainToday'])
        
        return X_copy

class WindDirectionTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador para direcciones del viento categóricas.
    Convierte direcciones cardinales a grados y luego a componentes sin/cos
    para mantener la naturaleza cíclica de la dirección.
    """
    def __init__(self):
        # Mapeo de direcciones cardinales a grados
        self.wind_dir_map = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        Convierte direcciones del viento a componentes trigonométricas sin/cos.
        """
        X_copy = X.copy()
        
        # Transformación de WindGustDir con nombres específicos
        if 'WindGustDir' in X_copy.columns:
            X_copy['WindGustDir'] = X_copy['WindGustDir'].map(self.wind_dir_map)
            X_copy['WindDir_sin'] = np.sin(np.deg2rad(X_copy['WindGustDir']))
            X_copy['WindDir_cos'] = np.cos(np.deg2rad(X_copy['WindGustDir']))
        
        # Transformación de direcciones horarias específicas
        for col in ['WindDir9am', 'WindDir3pm']:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].map(self.wind_dir_map)
                X_copy[f'{col}_sin'] = np.sin(np.deg2rad(X_copy[col]))
                X_copy[f'{col}_cos'] = np.cos(np.deg2rad(X_copy[col]))
        
        return X_copy

class OutlierTreatmentTransformer(BaseEstimator, TransformerMixin):
    """
    Tratamiento de valores atípicos y normalización robusta.
    Aplica límites específicos por variable y transformaciones log+robust scaling.
    """
    def __init__(self):
        self.limite_evaporacion = 30  # Límite físicamente plausible para evaporación
        self.medianas_evap = None
        self.robust_scaler = None
        
    def fit(self, X, y=None):
        """
        Calcula límites de outliers y ajusta transformaciones robustas.
        """
        X_copy = X.copy()
        
        # Cálculo de medianas regionales para evaporación tras filtrar outliers
        X_copy_evap = X_copy.copy()
        X_copy_evap.loc[X_copy_evap['Evaporation'] > self.limite_evaporacion, 'Evaporation'] = np.nan
        self.medianas_evap = X_copy_evap.groupby('region')['Evaporation'].median()
        
        # Ajuste de RobustScaler para variables con distribuciones asimétricas
        self.robust_scaler = RobustScaler()
        cols_to_scale = ['Rainfall', 'WindSpeed9am', 'WindSpeed3pm', 'WindGustSpeed']
        
        # Transformación logarítmica previa al escalado para variables sesgadas
        X_for_scaling = X_copy[cols_to_scale].copy()
        X_for_scaling = np.log1p(X_for_scaling)
        
        self.robust_scaler.fit(X_for_scaling)
        
        return self
    
    def transform(self, X):
        """
        Aplica tratamiento de outliers y transformaciones robustas.
        """
        X_copy = X.copy()
        
        # Tratamiento de outliers de evaporación mediante reemplazo por mediana regional
        X_copy.loc[X_copy['Evaporation'] > self.limite_evaporacion, 'Evaporation'] = np.nan
        X_copy['Evaporation'] = X_copy.apply(
            lambda row: self.medianas_evap[row['region']] if pd.isna(row['Evaporation']) else row['Evaporation'],
            axis=1
        )
        
        # Aplicación de log1p + RobustScaler para variables asimétricas
        cols_to_scale = ['Rainfall', 'WindSpeed9am', 'WindSpeed3pm', 'WindGustSpeed']
        X_copy[cols_to_scale] = np.log1p(X_copy[cols_to_scale])
        X_copy[cols_to_scale] = self.robust_scaler.transform(X_copy[cols_to_scale])
        
        return X_copy

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador final que prepara el dataset para el modelo de ML.
    Elimina columnas no predictivas, crea variables dummy y normaliza features numéricas.
    """
    def __init__(self):
        self.standard_scaler = None
        # Columnas a eliminar del dataset final
        self.columns_to_drop = ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'Month', 'RainTomorrow']
        self.columns_to_dummy = ['region', 'Season']
        # Variables para normalización estándar
        self.columns_to_scale = [
            'MinTemp', 'MaxTemp', 'Evaporation', 'Sunshine', 'Humidity9am', 
            'Humidity3pm', 'Temp9am', 'Temp3pm'
        ]
        
    def fit(self, X, y=None):
        """
        Ajusta el StandardScaler para variables continuas seleccionadas.
        """
        X_copy = X.copy()
        
        # Eliminación de columnas no predictivas
        cols_to_drop = [col for col in self.columns_to_drop if col in X_copy.columns]
        X_copy = X_copy.drop(columns=cols_to_drop)
        
        # Generación de variables dummy para categóricas
        X_copy = pd.get_dummies(X_copy, columns=self.columns_to_dummy, drop_first=True, dtype='int')
        
        # Ajuste de StandardScaler para variables continuas
        cols_to_scale = [col for col in self.columns_to_scale if col in X_copy.columns]
        self.standard_scaler = StandardScaler()
        self.standard_scaler.fit(X_copy[cols_to_scale])
        
        return self
    
    def transform(self, X):
        """
        Transforma el dataset a su formato final para inferencia.
        """
        X_copy = X.copy()
        
        # Eliminación de columnas no predictivas
        cols_to_drop = [col for col in self.columns_to_drop if col in X_copy.columns]
        X_copy = X_copy.drop(columns=cols_to_drop)
        
        # Generación de variables dummy
        X_copy = pd.get_dummies(X_copy, columns=self.columns_to_dummy, drop_first=True, dtype='int')
        
        # Normalización de variables continuas
        cols_to_scale = [col for col in self.columns_to_scale if col in X_copy.columns]
        X_copy[cols_to_scale] = self.standard_scaler.transform(X_copy[cols_to_scale])
        
        # Definición del esquema final de features esperado por el modelo
        expected_columns = [
            'MinTemp', 'MaxTemp', 'Evaporation', 'Sunshine', 'Humidity9am', 
            'Humidity3pm', 'Temp9am', 'Temp3pm', 'Rainfall', 'WindGustSpeed', 
            'WindSpeed9am', 'WindSpeed3pm', 'Pressure9am', 'Pressure3pm', 
            'Cloud9am', 'Cloud3pm', 'RainToday', 'WindDir_sin', 'WindDir_cos', 
            'WindDir9am_sin', 'WindDir9am_cos', 'WindDir3pm_sin', 'WindDir3pm_cos', 
            'region_1', 'region_2', 'region_3', 'region_4', 'region_5', 
            'Season_1', 'Season_2', 'Season_3'
        ]
        
        # Garantía de consistencia del esquema - agregar columnas faltantes
        for col in expected_columns:
            if col not in X_copy.columns:
                X_copy[col] = 0
        
        # Reordenamiento según esquema esperado
        X_copy = X_copy[expected_columns]
        
        # Asegurar tipos de datos correctos para el modelo
        float_cols = [
            'MinTemp', 'MaxTemp', 'Evaporation', 'Sunshine', 'Humidity9am', 
            'Humidity3pm', 'Temp9am', 'Temp3pm', 'Rainfall', 'WindGustSpeed', 
            'WindSpeed9am', 'WindSpeed3pm', 'Pressure9am', 'Pressure3pm', 
            'Cloud9am', 'Cloud3pm', 'RainToday', 'WindDir_sin', 'WindDir_cos', 
            'WindDir9am_sin', 'WindDir9am_cos', 'WindDir3pm_sin', 'WindDir3pm_cos'
        ]
        
        int_cols = [
            'region_1', 'region_2', 'region_3', 'region_4', 'region_5', 
            'Season_1', 'Season_2', 'Season_3'
        ]
        
        X_copy[float_cols] = X_copy[float_cols].astype('float64')
        X_copy[int_cols] = X_copy[int_cols].astype('int64')
        
        return X_copy

class WeatherPreprocessingPipeline:
    """
    Pipeline principal que orquesta todo el flujo de preprocesamiento.
    Combina todos los transformadores en un pipeline secuencial de sklearn.
    """
    def __init__(self):
        self.pipeline = Pipeline([
            ('region_cluster', RegionClusterTransformer()),
            ('date_transform', DateTransformer()),
            ('rain_encoder', RainTodayTomorrowEncoder()),
            ('imputer', CustomImputer()),
            ('wind_transform', WindDirectionTransformer()),
            ('outlier_treatment', OutlierTreatmentTransformer()),
            ('feature_engineering', FeatureEngineeringTransformer())
        ])
        
    def fit(self, X, y=None):
        """Entrena todo el pipeline de preprocesamiento."""
        self.pipeline.fit(X, y)
        return self
    
    def transform(self, X):
        """Aplica todas las transformaciones del pipeline."""
        return self.pipeline.transform(X)
    
    def fit_transform(self, X, y=None):
        """Entrena y transforma en una sola operación."""
        return self.pipeline.fit_transform(X, y)
    
    def save_pipeline(self, filename):
        """Serializa el pipeline entrenado para uso en producción."""
        with open(filename, 'wb') as f:
            pickle.dump(self.pipeline, f)
    
    def load_pipeline(self, filename):
        """Carga un pipeline previamente entrenado."""
        with open(filename, 'rb') as f:
            self.pipeline = pickle.load(f)

def train_and_save_pipeline(data_path, pipeline_path):
    """
    Función principal para entrenar el pipeline completo y guardarlo para producción.
    
    Args:
        data_path (str): Ruta al archivo CSV con datos de entrenamiento
        pipeline_path (str): Ruta donde guardar el pipeline serializado
        
    Returns:
        tuple: (pipeline, X_train_processed, X_test_processed)
    """
    # Carga del dataset completo
    df = pd.read_csv(data_path)
    
    # División estratificada para entrenamiento y validación
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Inicialización y entrenamiento del pipeline
    preprocessing_pipeline = WeatherPreprocessingPipeline()
    
    print("Entrenando pipeline de preprocesamiento...")
    preprocessing_pipeline.fit(train_df)
    
    print("Guardando pipeline...")
    preprocessing_pipeline.save_pipeline(pipeline_path)
    
    print(f"Pipeline guardado en: {pipeline_path}")
    
    # Procesamiento de conjuntos de datos para verificación
    print("Procesando datos de prueba...")
    X_train_processed = preprocessing_pipeline.transform(train_df)
    X_test_processed = preprocessing_pipeline.transform(test_df)
    
    print(f"Datos de entrenamiento procesados: {X_train_processed.shape}")
    print(f"Datos de prueba procesados: {X_test_processed.shape}")
    print(f"Columnas finales: {list(X_train_processed.columns)}")
    print(f"Tipos de datos:\n{X_train_processed.dtypes}")
    
    return preprocessing_pipeline, X_train_processed, X_test_processed

def load_and_use_pipeline(pipeline_path, new_data):
    """
    Función para cargar pipeline entrenado y procesar nuevos datos en producción.
    
    Args:
        pipeline_path (str): Ruta al pipeline serializado
        new_data (pd.DataFrame): Datos nuevos para procesar
        
    Returns:
        pd.DataFrame: Datos procesados listos para inferencia
    """
    pipeline = WeatherPreprocessingPipeline()
    pipeline.load_pipeline(pipeline_path)
    
    processed_data = pipeline.transform(new_data)
    return processed_data

if __name__ == "__main__":
    # Configuración para ejecución standalone del pipeline
    data_path = "weatherAUS.csv"
    pipeline_path = "weather_preprocessing_pipeline.pkl"
    
    # Entrenamiento y persistencia del pipeline
    pipeline, X_train, X_test = train_and_save_pipeline(data_path, pipeline_path)
    
    print("Pipeline entrenado y guardado exitosamente!")
    print(f"Columnas finales: {list(X_train.columns)}")
    print(f"Forma de datos procesados: {X_train.shape}")