# Modelo de Inferencia del Clima (Docker)

Este repositorio contiene los archivos necesarios para realizar inferencias usando un modelo de aprendizaje automático entrenado con TensorFlow y un pipeline de preprocesamiento.

---

## Estructura del proyecto

```
dockerfile/
├── Dockerfile
├── inferencia.py
├── preprocessing.py
├── weather_preprocessing_pipeline.pkl
├── requirements.txt
├── datos_prueba.csv       
```

---

## 🐳 Construcción de la imagen Docker

Desde la carpeta que contiene el `Dockerfile`, ejecutar:

```bash
docker build -t modelo-clima .
```

Esto crea una imagen llamada `modelo-clima` con todas las dependencias necesarias para ejecutar la inferencia.

---

## Ejecución del contenedor

Para ejecutar la inferencia con un archivo CSV de entrada:

```bash
docker run --rm -v ${PWD}:/app modelo-clima python inferencia.py --input datos_prueba.csv --output resultados.json
```

Notas:
- `${PWD}` (PowerShell) monta la carpeta local dentro del contenedor.
- `--input`: ruta del archivo CSV de entrada (debe existir en la carpeta local).
- `--output`: ruta donde se guardará el archivo de salida (JSON con resultados).
- El archivo de salida quedará disponible en tu carpeta local al finalizar.

---

##  Resultado

El archivo `resultados.json` contendrá un JSON con las predicciones del modelo. El formato de salida es:

```json
{
  "predicciones": [0.78, 0.12, 0.65, ...]
}
```

Cada valor representa la probabilidad (en formato float) de que llueva mañana. Se eligió este formato porque es más útil para el análisis, ya que permite aplicar distintos umbrales según el caso de uso.

Para convertir las probabilidades en etiquetas binarias (0 o 1), se puede utilizar un umbral. En este caso, se determinó que el **umbral óptimo es 0.61**, por lo que un simple script puede realizar la conversión así:

```python
pred_binarias = [1 if p > 0.61 else 0 for p in predicciones]
```

---

## Ejemplo de uso

```bash
docker build -t modelo-clima .
docker run --rm -v ${PWD}:/app modelo-clima python inferencia.py --input datos_prueba.csv --output resultados.json

#window + bash
winpty docker run --rm -v "/$(pwd)":/app modelo-clima python inferencia.py --input datos_prueba.csv --output resultados.json

```

---

## Limpieza

El contenedor se elimina automáticamente al terminar gracias a `--rm`, por lo tanto no se generan residuos.

