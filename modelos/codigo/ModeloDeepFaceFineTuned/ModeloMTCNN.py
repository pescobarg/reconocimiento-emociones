import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from mtcnn import MTCNN

# Cargar el modelo entrenado
model = load_model(r"..\..\archivo\ModeloDeepFaceTunedEntrenado.h5")

# Las clases deben coincidir con las usadas al entrenar
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Inicializar detector MTCNN
detector = MTCNN()

# Captura desde webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo acceder a la cámara")
else:
    print("Presiona 'q' para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar rostros con MTCNN
    results = detector.detect_faces(frame)

    for result in results:
        # Extraer las coordenadas de la cara
        x, y, w, h = result['box']
        
        # Extraer rostro, redimensionar y normalizar
        face_roi = frame[y:y+h, x:x+w]
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
        face_roi = cv2.resize(face_roi, (48, 48))  # Redimensionar a 48x48
        face_roi = face_roi.astype("float32") / 255.0  # Normalizar
        face_roi = img_to_array(face_roi)
        face_roi = np.expand_dims(face_roi, axis=0)  # (1, 48, 48, 1)
        face_roi = np.expand_dims(face_roi, axis=-1)  # Añadir canal si no lo tiene

        # Predecir emoción utilizando el modelo cargado
        preds = model.predict(face_roi)
        emotion_label = class_labels[np.argmax(preds)]  # Obtener la etiqueta de emoción con mayor probabilidad

        # Mostrar resultados
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Dibujar un rectángulo alrededor de la cara
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)  # Mostrar la emoción en la imagen

    # Mostrar el frame con las detecciones y predicciones
    cv2.imshow("Detección de Emociones", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos de la cámara
cap.release()
cv2.destroyAllWindows()
