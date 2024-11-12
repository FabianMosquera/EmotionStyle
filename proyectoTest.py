import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# Función para cargar y verificar la imagen del usuario
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: No se pudo cargar la imagen. Verifica la ruta.")
    return image

# Cargar y mostrar la imagen original del usuario
image_path = 'sapa.jpeg'  # Cambia a la ruta de tu imagen
image = load_image(image_path)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Imagen Original del Usuario")
plt.show()

# Detección de rostro con OpenCV
def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        raise ValueError("No se detectó ningún rostro en la imagen.")
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image, faces

# Detectar rostro y mostrar la imagen con la detección
image_with_faces, faces = detect_face(image.copy())
plt.imshow(cv2.cvtColor(image_with_faces, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Detección de Rostro")
plt.show()

# Clasificación de emociones con el modelo entrenado 'modelo_emociones.h5'
def classify_emotion(face_image):
    emotion_model = tf.keras.models.load_model('modelo_emociones.h5')

    if len(face_image.shape) == 3 and face_image.shape[2] == 3:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image_resized = cv2.resize(face_image, (48, 48))  # Ajustar tamaño a 48x48
    face_image_resized = face_image_resized / 255.0
    face_image_resized = face_image_resized.reshape(1, 48, 48, 1)  # Añadir batch y canal de color
    emotion_prediction = emotion_model.predict(face_image_resized)
    emotion_label = np.argmax(emotion_prediction)
    return emotion_label  # Aquí regresa el índice de la emoción detectada

# Selección de estilo basado en la emoción detectada
def select_style(emotion_label):
    emotion_names = ['Alegría', 'Tristeza', 'Ira', 'Miedo', 'Calma', 'Sorpresa']
    
    # Verificar que el valor de emotion_label esté dentro del rango esperado
    if emotion_label < 0 or emotion_label >= len(emotion_names):
        print(f"Error: El índice de emoción {emotion_label} está fuera del rango esperado.")
        emotion_label = 0  # Establecer un valor predeterminado si hay un error
    
    print(f"Emoción detectada: {emotion_names[emotion_label]}")

    # Mapear las emociones a estilos de imagen
    styles = {
        0: 'pop-art-style.jpg',  # Alegría
        1: 'expressionism_style.jpg',  # Tristeza
        2: 'abstract_expressionism_style.jpg',  # Ira
        3: 'dark_art_style.jpg',  # Miedo
        4: 'zen_art_style.jpg',  # Calma
        5: 'cubism_style.jpg',  # Sorpresa
    }
    
    # Retornar el estilo seleccionado basado en la emoción
    return styles.get(emotion_label, 'pop-art-style.jpg')

# Función para cargar y procesar la imagen de estilo de TensorFlow Hub
def load_and_process_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # Convertir la imagen a float32
    image = tf.image.resize(image, (256, 256))  # Ajustar el tamaño de la imagen
    image = image[tf.newaxis, :]
    return image

# Cargar el modelo de TensorFlow Hub para estilización
style_model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

# Aplicar estilo usando el modelo de TensorFlow Hub
def apply_style_transfer(content_image_path, style_image_path):
    # Cargar y procesar las imágenes
    content_image = load_and_process_image(content_image_path)
    style_image = load_and_process_image(style_image_path)

    # Aplicar la transferencia de estilo
    stylized_image = style_model(content_image, style_image)[0]
    return stylized_image

# Aplicar el estilo al rostro segmentado
face_image = image_with_faces[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]]
emotion_label = classify_emotion(face_image)
style_path = select_style(emotion_label)

# Aplicar el estilo seleccionado al rostro
styled_image = apply_style_transfer(image_path, style_path)

# Convertir la imagen estilizada de TensorFlow a un formato que se pueda mostrar
styled_image = tf.squeeze(styled_image)  # Eliminar dimensión extra

# Convertir la imagen estilizada a formato Numpy y mostrarla
styled_image = styled_image.numpy()
styled_image = np.clip(styled_image, 0, 1)  # Asegurarse de que los valores estén en el rango [0, 1]

# Mostrar la imagen generada
plt.imshow(styled_image)
plt.axis('off')
plt.title("Imagen Estilizada")
plt.show()

# Guardar la imagen final estilizada
output_image_path = 'imagen_transformada.png'
plt.imsave(output_image_path, styled_image)
print(f"La imagen transformada se guardó en {output_image_path}")
