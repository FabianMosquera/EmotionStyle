from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooserIconView
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput

# Función para cargar y verificar la imagen
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: No se pudo cargar la imagen. Verifica la ruta.")
    return image

# Función para detectar el rostro en la imagen
def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        raise ValueError("No se detectó ningún rostro en la imagen.")
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image, faces

# Función para clasificar la emoción
def classify_emotion(face_image):
    emotion_model = tf.keras.models.load_model('modelo_emociones.h5')
    face_image_resized = cv2.resize(face_image, (48, 48))
    if len(face_image_resized.shape) == 3:  # Verificar si la imagen tiene 3 canales
        face_image_resized = cv2.cvtColor(face_image_resized, cv2.COLOR_BGR2GRAY)
    face_image_resized = face_image_resized / 255.0
    face_image_resized = face_image_resized.reshape(1, 48, 48, 1)  # Añadir batch y canal de color
    emotion_prediction = emotion_model.predict(face_image_resized)
    emotion_label = np.argmax(emotion_prediction)
    return emotion_label

# Función para seleccionar el estilo basado en la emoción detectada
def select_style(emotion_label):
    emotion_names = ['Alegría', 'Tristeza', 'Ira', 'Miedo', 'Calma', 'Sorpresa']
    styles = {
        0: 'pop-art-style.jpg',  # Alegría
        1: 'expressionism_style.jpg',  # Tristeza
        2: 'abstract_expressionism_style.jpg',  # Ira
        3: 'dark_art_style.jpg',  # Miedo
        4: 'zen_art_style.jpg',  # Calma
        5: 'cubism_style.jpg',  # Sorpresa
    }
    print(f"Emoción detectada: {emotion_names[emotion_label]}")
    return emotion_names[emotion_label], styles.get(emotion_label, 'pop-art-style.jpg')

# Cargar el modelo de estilización desde TensorFlow Hub
style_model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

# Función para aplicar la transferencia de estilo
def apply_style_transfer(content_image_path, style_image_path):
    content_image = load_and_process_image(content_image_path)
    style_image = load_and_process_image(style_image_path)
    stylized_image = style_model(content_image, style_image)[0]
    return stylized_image

# Función para cargar y procesar las imágenes para el modelo de estilo
def load_and_process_image(image_path):
    image_cv = cv2.imread(image_path)
    temp_path = "temp_style_image.jpg"
    cv2.imwrite(temp_path, image_cv)
    image = tf.io.read_file(temp_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, (256, 256))
    image = image[tf.newaxis, :]
    return image

class EmotionApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')
        
        # Crear una instancia de la cámara
        self.camera = Camera(play=True, resolution=(640, 480))
        layout.add_widget(self.camera)
        
        # Crear una etiqueta para mostrar la emoción detectada
        self.emotion_label = Label(text="Emoción: ")
        layout.add_widget(self.emotion_label)
        
        # Crear un botón para capturar la foto
        btn_capture = Button(text="Tomar Foto")
        btn_capture.bind(on_press=self.take_photo)
        layout.add_widget(btn_capture)
        
        # Crear una imagen para mostrar la imagen transformada
        self.transformed_image = Image()
        layout.add_widget(self.transformed_image)

        # Crear un botón para guardar la imagen transformada
        btn_save = Button(text="Guardar Imagen Transformada")
        btn_save.bind(on_press=self.save_image)
        layout.add_widget(btn_save)
        
        return layout
    
    def take_photo(self, instance):
        # Acceder a la textura de la cámara
        texture = self.camera.texture
        frame = np.frombuffer(texture.pixels, dtype=np.uint8)
        frame = frame.reshape((texture.height, texture.width, 4))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # Guardar la imagen capturada en formato .jpg
        self.image_path = "captura.jpg"
        cv2.imwrite(self.image_path, frame)
        
        # Cargar y procesar la imagen guardada
        image = load_image(self.image_path)
        image_with_faces, faces = detect_face(image.copy())
        
        # Obtener el rostro detectado para clasificación de emociones
        x, y, w, h = faces[0]
        face_image = image_with_faces[y:y+h, x:x+w]
        
        # Clasificar la emoción y seleccionar el estilo
        emotion_label = classify_emotion(face_image)
        emotion_name, style_path = select_style(emotion_label)
        
        # Actualizar la etiqueta de emoción detectada
        self.emotion_label.text = f"Emoción: {emotion_name}"
        
        # Aplicar el estilo a la imagen completa y obtener la imagen transformada
        styled_image = apply_style_transfer(self.image_path, style_path)
        
        # Convertir y guardar la imagen estilizada
        self.final_image = tf.squeeze(styled_image).numpy()
        self.final_image = np.clip(self.final_image, 0, 1)
        self.output_image_path = 'imagen_transformada.jpg'
        plt.imsave(self.output_image_path, self.final_image)
        
        # Actualizar la imagen transformada en la interfaz
        self.transformed_image.source = self.output_image_path
        self.transformed_image.reload()  # Recargar para mostrar la imagen nueva
    
    def save_image(self, instance):
        # Popup para seleccionar el nombre del archivo
        popup_content = BoxLayout(orientation='vertical')
        file_name_input = TextInput(hint_text='Nombre del archivo', multiline=False)
        popup_content.add_widget(file_name_input)
        
        # Botón para guardar el archivo
        btn_save_confirm = Button(text="Guardar")
        popup_content.add_widget(btn_save_confirm)
        
        # Crear y mostrar el popup
        popup = Popup(title='Guardar Imagen Transformada', content=popup_content, size_hint=(0.8, 0.4))
        
        def save_confirm(instance):
            # Guardar la imagen con el nombre proporcionado
            file_name = file_name_input.text if file_name_input.text else "imagen_transformada"
            save_path = f"{file_name}.jpg"
            plt.imsave(save_path, self.final_image)
            popup.dismiss()
            print(f"Imagen guardada como: {save_path}")
        
        btn_save_confirm.bind(on_press=save_confirm)
        popup.open()

# Ejecutar la aplicación
if __name__ == '__main__':
    EmotionApp().run()
