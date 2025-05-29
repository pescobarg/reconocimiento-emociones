import cv2
import os
from keras.models import model_from_json
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join("emotiondetector.json"), "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

model.load_weights(os.path.join(r"..\archivo\ModeloTensorFlowKerasEntrenado.h5"))


haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Función para extraer características de la imagen y normalizarlas
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Asegúrate de que la imagen tenga la forma correcta
    return feature / 255.0  # Normalizar la imagen entre 0 y 1

# Inicializar la cámara
webcam = cv2.VideoCapture(0)

# Diccionario de etiquetas de emociones
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    # Capturar un fotograma de la webcam
    i, im = webcam.read()

    if not i:  # Si no se pudo leer el fotograma, continuamos con la siguiente iteración
        continue

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detectar caras en la imagen
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    try:
        for (p, q, r, s) in faces:
            # Recortar la región de la cara detectada
            face_image = gray[q:q + s, p:p + r]

            # Dibujar un rectángulo alrededor de la cara
            cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)

            # Redimensionar la imagen de la cara para que sea de tamaño 48x48
            face_image_resized = cv2.resize(face_image, (48, 48))

            # Extraer las características de la cara
            img = extract_features(face_image_resized)

            # Realizar la predicción con el modelo
            pred = model.predict(img)

            # Obtener la etiqueta de la emoción predicha
            prediction_label = labels[pred.argmax()]

            # Mostrar la etiqueta de la emoción en la imagen
            cv2.putText(im, prediction_label, (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

        # Mostrar el resultado en la ventana
        cv2.imshow("Output", im)

        # Detener el proceso si se presiona la tecla 'Esc'
        if cv2.waitKey(1) & 0xFF == 27:
            break

    except cv2.error:
        pass

# Liberar la webcam y cerrar las ventanas de OpenCV
webcam.release()
cv2.destroyAllWindows()