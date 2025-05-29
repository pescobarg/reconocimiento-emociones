import os
import cv2
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import classification_report, f1_score, accuracy_score
from deepface import DeepFace
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ===== CONFIG =====
DF_tuned = load_model(r"..\modelos\archivo\ModeloDeepFaceTunedEntrenado.h5")
cnn_casera = load_model(r"modelos\archivo\CNN_casera.h5")

class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
test_dir = "Data/test"

# ===== INIT METRICS =====
preds_finetuned = []
preds_cnn_casera = []
preds_deepface = []
true_labels = []

# ===== PROCESAR CADA IMAGEN =====
for label in os.listdir(test_dir):
    label_dir = os.path.join(test_dir, label)
    if not os.path.isdir(label_dir):
        continue
    for img_name in tqdm(os.listdir(label_dir), desc=f"Procesando {label}"):
        img_path = os.path.join(label_dir, img_name)

        # Leer imagen en RGB para DeepFace
        img_rgb = cv2.imread(img_path)
        if img_rgb is None:
            continue
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

        # Leer imagen en escala de grises para tu modelo
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.resize(img_gray, (48, 48))
        face = img_gray.astype("float32") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        # ==== Predicción con el fine tuneado ==== 
        preds = DF_tuned.predict(face, verbose=0)
        df_tuned_pred = class_labels[np.argmax(preds)]
        preds_finetuned.append(df_tuned_pred)

        # ==== Predicción con CNN casera ==== 
        preds = cnn_casera.predict(face, verbose=0)
        cnn_tuned_pred = class_labels[np.argmax(preds)]
        preds_cnn_casera.append(cnn_tuned_pred)


        # ==== predicción con deepface ==== 
        try:
            result = DeepFace.analyze(
                img_rgb,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='yolov8',
                align=True 
            )
            if isinstance(result, list):
                result = result[0]
            deepface_pred = result['dominant_emotion']
        except Exception as e:
            deepface_pred = "unknown"

        preds_deepface.append(deepface_pred)

        # ==== VERDADERA ETIQUETA ==== 
        true_labels.append(label)


# Crear DataFrames con nombres de emociones como columnas
df_finetuned = pd.DataFrame(probs_finetuned, columns=class_labels)
df_cnn = pd.DataFrame(probs_cnn_casera, columns=class_labels)

# Matriz de correlación
corr_finetuned = df_finetuned.corr()
corr_cnn = df_cnn.corr()

# Graficar
plt.figure(figsize=(8, 6))
sns.heatmap(corr_finetuned, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de correlación - Modelo Fine-tuneado")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_cnn, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de correlación - CNN Casera")
plt.show()

# ===== EVALUACIÓN Y REPORTE =====
def generate_report(y_true, y_pred, model_name):
    report = f"\n📊 Evaluación modelo {model_name}:\n"
    report += classification_report(y_true, y_pred, labels=class_labels, zero_division=1)
    report += f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}"
    report += f"\nF1-score (weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}"
    return report

# Evaluar y generar reportes
report_finetuned = generate_report(true_labels, preds_finetuned, "Fine-Tuneado")
report_cnn_casera = generate_report(true_labels, preds_cnn_casera, "CNN Casera")

# Evaluar DeepFace (filtrando etiquetas desconocidas)
filtered_true = [t for t, p in zip(true_labels, preds_deepface) if p in class_labels]
filtered_preds = [p for p in preds_deepface if p in class_labels]
report_deepface = generate_report(filtered_true, filtered_preds, "DeepFace (default)")

# ===== GUARDAR A TXT =====
with open("comparativa.txt", "w", encoding="utf-8") as f:
    f.write("=== COMPARATIVA DE MODELOS DE DETECCIÓN DE EMOCIONES ===\n")
    f.write(report_finetuned)
    f.write("\n" + "="*60 + "\n")
    f.write(report_cnn_casera)
    f.write("\n" + "="*60 + "\n")
    f.write(report_deepface)

print("✅ Evaluación completa. Resultados guardados en 'comparativa.txt'")
