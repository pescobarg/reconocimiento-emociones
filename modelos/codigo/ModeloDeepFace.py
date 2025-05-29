from deepface import DeepFace
import cv2
import numpy as np

def obtener_emocion_dominante(result):
    return result['dominant_emotion']

# Iniciar cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo acceder a la cámara")
else:
    print("Presiona 'q' para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analizar emociones en el frame completo
        results = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='yolov8',
            align=True
        )

        # Si solo devuelve un dict, lo convertimos en lista
        if not isinstance(results, list):
            results = [results]

        for result in results:
            face_region = result['region']
            x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
            emocion = obtener_emocion_dominante(result)

            # Dibujar rectángulo y emoción
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emocion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    except Exception as e:
        print("Error:", e)

    # Mostrar frame
    cv2.imshow('Detección de Emociones', frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
