# Aplicación de Detección de Emociones y Transformación de Estilo

Esta aplicación en Kivy utiliza una cámara para capturar imágenes, detecta la emoción en el rostro de la persona y aplica una transformación de estilo basada en la emoción detectada. Además, permite guardar la imagen transformada localmente. El proyecto hace uso de modelos de clasificación de emociones y transferencia de estilo.

## Tabla de Contenidos

- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Ejecución](#ejecución)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Funcionalidades](#funcionalidades)
- [Créditos](#créditos)

---

## Requisitos

- **Python 3.7 o superior**
- **Kivy** - para la creación de interfaces gráficas.
- **OpenCV** - para manipulación y procesamiento de imágenes.
- **TensorFlow** y **TensorFlow Hub** - para el modelo de clasificación de emociones y transferencia de estilo.
- **Matplotlib** - para guardar las imágenes estilizadas.

Instala los paquetes requeridos con el siguiente comando:

```bash
pip install kivy opencv-python tensorflow tensorflow-hub matplotlib

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/deteccion-emociones-y-estilo.git
   cd deteccion-emociones-y-estilo

2. Configura el modelo de emociones:
Descarga o entrena un modelo de emociones y guarda el archivo como modelo_emociones.h5 en el directorio principal del proyecto.
El modelo debe estar entrenado para predecir emociones a partir de imágenes en escala de grises de tamaño 48x48.

3. Crea un directorio de estilos:
Agrega imágenes de estilo en el directorio estilos/ con nombres correspondientes a las emociones que detectará la aplicación.

---

## Ejecución

Ejecuta la aplicación con:

```bash
    python main.py