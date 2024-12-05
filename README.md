# Sistema de Seguimiento y Análisis de Vehículos

Este proyecto implementa un sistema de detección y seguimiento de vehículos utilizando modelos YOLO (You Only Look Once) y proporciona una interfaz web interactiva para analizar los resultados.

## 🚀 Características

- **Detección y seguimiento de vehículos** en tiempo real
- **Soporte para múltiples modelos** YOLO (nano, small, medium)
- **Interfaz web interactiva** para análisis de datos
- Visualización de **estadísticas y métricas**
- Análisis de **trayectorias individuales**
- **Mapas de calor** de detecciones
- Estadísticas de **tamaño y distribución**

## 📋 Requisitos del Sistema

- Python 3.11 o superior
- Tarjeta gráfica compatible con CUDA (recomendado)
- Espacio en disco para modelos YOLO

## 🛠 Instalación

1. Clonar el repositorio:

    ```bash
    git clone https://github.com/agutierrezmorag/yolo-streamlit-car-tracker.git
    cd car-tracking
    ```

2. Instalar dependencias:

    ```bash
    pip install -r requirements.txt
    ```

## 💻 Uso

1. Definir un **ROI**:

    ```bash
    python polygon_roi_selector.py
    ```

    El ROI será definido mediante los puntos definidos con clicks del mouse. Una vez definido, presionar la tecla `Q` para terminar el proceso. Los valores del ROI serán registrados en `constants.py` automaticamente.

2. Iniciar el tracking de los elementos del video:

    ```bash
    python tracker.py
    ```

    Este archivo leerá el video y el ROI definido en el paso anterior para realizar el registro de objetos detectados. La deteccion es hecha con los modelos **YOLO11**, desde el *nano* hasta el *medium*, o los que se hayan definido en el archivo `constants.py`.

    Toda deteccion sera registrada en su correspondiente archivo  `.csv`. Tambien se llevará registro de los vehiculos detectados con una screenshot, las que se almacenaran automaticamente en una subcarpeta de su correspondiente modelo.

3. Iniciar la interfaz web de análisis:

    ```bash
    streamlit run st_app.py
    ```

    Esta aplicacion, hecha con Streamlit, leerá directamente desde los archivos `.csv` generados y dispondrá de análisis de esa data mediante gráficos.

## 📊 Características de la Interfaz

- **Estadísticas Generales**:
  - **Total** de vehículos detectados
  - **Promedio** de vehículos por minuto
  - **Máximo** de vehículos por minuto

- **Visualizaciones**:
  - Gráfico de conteo temporal
  - Mapa de calor de detecciones
  - Análisis de trayectorias individuales
  - Distribución de tamaños

## 📁 Estructura del Proyecto

```md
├── polygon_roi_selector.py     # Selector ROI interactivo
├── constants.py                # Archivo de utilidad para registro de variables
├── tracker.py                  # Procesamiento principal de video
├── st_app.py                   # Interfaz web Streamlit
├── requirements.txt            # Dependencias del proyecto
└── output_yolo*/               # Carpetas de salida por modelo
    ├── data/                   # Datos CSV de detecciones
    └── detected_cars/          # Imágenes de vehículos detectados
```
