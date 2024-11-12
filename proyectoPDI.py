# Importación de las librerías necesarias
import cv2
import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import torch.nn as nn
import torch.optim as optim

# Función para cargar y verificar la imagen del usuario
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: No se pudo cargar la imagen. Verifica la ruta.")
    return image

# Cargar y mostrar la imagen original del usuario
image_path = 'triste.jpg'  # Cambia a la ruta de tu imagen
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
    # Asegúrate de que el valor de emotion_label esté dentro del rango correcto
    emotion_names = ['Alegría', 'Tristeza', 'Ira', 'Miedo', 'Calma', 'Sorpresa']
    
    # Verificar si emotion_label está dentro del rango
    if emotion_label < 0 or emotion_label >= len(emotion_names):
        print(f"Advertencia: Valor de emotion_label fuera de rango ({emotion_label}). Usando 'Alegría' por defecto.")
        emotion_label = 0  # Valor por defecto (Alegría)

    print(f"Emoción detectada: {emotion_names[emotion_label]}")
    
    styles = {
        0: 'pop-art-style.jpg',  # Alegría
        1: 'expressionism_style.jpg',  # Tristeza
        2: 'abstract_expressionism_style.jpg',  # Ira
        3: 'dark_art_style.jpg',  # Miedo
        4: 'zen_art_style.jpg',  # Calma
        5: 'cubism_style.jpg',  # Sorpresa
    }
    
    return styles.get(emotion_label, 'pop-art-style.jpg')

# Obtener la emoción y el estilo para la imagen
emotion_label = classify_emotion(image_with_faces[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]])
style_path = select_style(emotion_label)

# Mostrar el estilo seleccionado
print(f"Estilo seleccionado: {style_path}")


# Ahora puedes seguir con el resto del código para aplicar el estilo

# Función para obtener las características de contenido y estilo de VGG19
def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # Para la pérdida de contenido
            '28': 'conv5_1'
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# Función para calcular la matriz de Gram para la pérdida de estilo
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Función para aplicar transferencia de estilo
# Función para aplicar transferencia de estilo
def apply_style_transfer(content_img_path, style_img_path):
    # Cargar y procesar imágenes
    def load_and_process_image(img_path):
        image = Image.open(img_path)
        preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        image = preprocess(image).unsqueeze(0)  # Agregar dimensión de batch
        return image.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Cargar contenido y estilo
    content_img = load_and_process_image(content_img_path)
    style_img = load_and_process_image(style_img_path)

    # Cargar VGG19 preentrenado
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg = models.vgg19(pretrained=True).features.to(device).eval()

    # Definir los pesos para cada capa de estilo
    style_weights = {
        'conv1_1': 1.8,
        'conv2_1': 1.5,
        'conv3_1': 1.2,
        'conv4_1': 0.8,
        'conv5_1': 0.5
    }

    # Definir pesos para las pérdidas de contenido y estilo
    content_weight = 1e4  # Peso de la pérdida de contenido
    style_weight = 1e2    # Peso de la pérdida de estilo

    # Obtener características de contenido y estilo
    content_features = get_features(content_img, vgg)
    style_features = get_features(style_img, vgg)

    # Calcular las matrices de Gram de la imagen de estilo
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # Copiar imagen de contenido para inicializar la imagen generada
    generated_img = content_img.clone().requires_grad_(True)

    # Definir optimizador
    optimizer = optim.Adam([generated_img], lr=0.003)

    # Iterar para ajustar la imagen generada
    steps = 550
    for step in range(steps):
        generated_features = get_features(generated_img, vgg)
        
        # Pérdida de contenido
        content_loss = torch.mean((generated_features['conv4_2'] - content_features['conv4_2']) ** 2)
        
        # Pérdida de estilo
        style_loss = 0
        for layer in style_weights:
            generated_feature = generated_features[layer]
            generated_gram = gram_matrix(generated_feature)
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * torch.mean((generated_gram - style_gram) ** 2)
            style_loss += layer_style_loss / (generated_feature.shape[1] * generated_feature.shape[2] * generated_feature.shape[3])

        # Pérdida total
        total_loss = content_weight * content_loss + style_weight * style_loss

        # Optimizar
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()

        # Mostrar progreso
        if step % 50 == 0:
            print(f"Paso {step}/{steps}, Pérdida Total: {total_loss.item():.4f}")

    # Al final, devolver la imagen generada
    return generated_img.cpu().detach()

# Llamar a la función de transferencia de estilo y guardar la imagen generada
emotion_label = classify_emotion(image_with_faces[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]])
style_path = select_style(emotion_label)
styled_image = apply_style_transfer(image_path, style_path)

# Convertir tensor a imagen PIL
from torchvision.transforms import ToPILImage
to_pil = ToPILImage()
styled_image_pil = to_pil(styled_image.squeeze(0))  # Asegúrate de que el tensor tenga la forma correcta

# Guardar la imagen generada
output_image_path = 'imagen_transformada2.png'
styled_image_pil.save(output_image_path)
print(f"La imagen transformada se guardó en {output_image_path}")
