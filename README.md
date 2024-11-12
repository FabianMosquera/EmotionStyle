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
```

1. Clona el repositorio:
   ```bash
   git clone https://github.com/FabianMosquera/EmotionStyle.git
   cd EmotionStyle

2. Configura el modelo de emociones:
Descarga o entrena un modelo de emociones y guarda el archivo como modelo_emociones.h5 en el directorio principal del proyecto.
El modelo debe estar entrenado para predecir emociones a partir de imágenes en escala de grises de tamaño 48x48. para este proyecto ya cuenta con uno.

---

## Ejecución

Ejecuta la aplicación con:

```bash
    python appKivy.py
```

## Funcionalidades

1. Detección de emociones: La aplicación detecta rostros en la imagen y clasifica la emoción en una de las siguientes categorías:

* Alegría
* Tristeza
* Ira
* Miedo
* Calma
* Sorpresa

2. Transformación de estilo: Basado en la emoción detectada, se selecciona un estilo artístico y se aplica una transferencia de estilo sobre la imagen completa.

3. Guardado de la imagen: La aplicación permite guardar la imagen estilizada localmente, con un nombre definido por el usuario.


## Créditos
- Detección de rostros: Usa el clasificador en cascada de Haar de OpenCV.
- Modelo de emociones: El modelo modelo_emociones.h5 debe ser un clasificador entrenado previamente en un conjunto de datos de emociones.
- Transferencia de estilo: Basado en el modelo de arbitrary-image-stylization-v1-256 de TensorFlow Hub.