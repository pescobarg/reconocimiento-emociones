import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Cargamos el modelo entrenado
model = load_model(r"..\..\archivo\ModeloDeepFaceTunedEntrenado.h5")

# Las clases deben coincidir con las usadas al entrenar
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Cargamos el clasificador Haar Cascade para detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

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

    # Convertimos a escala de grises para la detección con Haar
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectamos rostros
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Se extrae el rostro, luego se redimensiona y normaliza antes de darselo a deepface para que analice
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi.astype("float32") / 255.0
        face_roi = img_to_array(face_roi)
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=-1)

        # Predicción de emoción
        preds = model.predict(face_roi)
        emotion_label = class_labels[np.argmax(preds)]

        # Dibujamos los resultados en el frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Se muestra el frame
    cv2.imshow("Detección de Emociones", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
