import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# Cargar modelos
modelo_persona = YOLO("yolov8n.pt")  # Modelo general para detectar personas
modelo_ppe = YOLO("best.pt")        # Modelo personalizado para detectar PPE

# Interfaz Streamlit
st.set_page_config(page_title="Sistema PPE", layout="centered")

st.title("🛡️ Sistema inteligente de uso de PPE")
st.image("logo.png", use_column_width=True)

st.markdown("""
### 🧠 Introducción:
Esta aplicación permite detectar si los trabajadores cumplen con el uso correcto de elementos de protección personal (PPE) como casco, chaleco y botas en una imagen cargada o tomada con cámara.

### 📋 Instrucciones de uso:
1. Cargue una imagen o use la cámara para tomar una.
2. El sistema detectará personas en la imagen.
3. Se verificará si cada persona tiene casco, chaleco y botas.
4. Se mostrará un mensaje indicando si puede o no ingresar a la fábrica.
""")

# Subir imagen o tomarla desde cámara
image_file = st.file_uploader("📷 Cargar una imagen", type=["jpg", "jpeg", "png"])
if not image_file:
    image_file = st.camera_input("📸 O toma una foto")

if image_file:
    img = Image.open(image_file).convert('RGB')
    img_np = np.array(img)

    # Detección de personas con modelo general
    detections_person = modelo_persona(img_np)[0]
    persons = [det for det in detections_person.boxes.data if int(det[5]) == 0]  # clase 0 en COCO = persona

    st.subheader(f"👥 Personas detectadas: {len(persons)}")

    for i, det in enumerate(persons):
        x1, y1, x2, y2 = map(int, det[:4])
        person_crop = img_np[y1:y2, x1:x2]

        # Detección de objetos PPE con el modelo personalizado
        ppe_detections = modelo_ppe(person_crop)[0]
        labels = [ppe_detections.names[int(cls)] for cls in ppe_detections.boxes.cls]

        # Verificación de requisitos
        requisitos = {"casco", "chaleco", "botas"}
        encontrados = set([label.lower() for label in labels])
        cumple = requisitos.issubset(encontrados)

        # Dibujar bounding boxes
        for box, cls in zip(ppe_detections.boxes.xyxy, ppe_detections.boxes.cls):
            x1_o, y1_o, x2_o, y2_o = map(int, box)
            label = ppe_detections.names[int(cls)]
            cv2.rectangle(person_crop, (x1_o, y1_o), (x2_o, y2_o), (0, 255, 0), 2)
            cv2.putText(person_crop, label, (x1_o, y1_o - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        st.image(person_crop, caption=f"👤 Persona {i+1}", channels="RGB")

        if cumple:
            st.success("✅ Cumple con los requisitos para el ingreso a la fábrica 😎")
        else:
            st.error("🚨 ALERTA - NO CUMPLE CON LOS REQUISITOS DE PPE. NO PUEDE INGRESAR A LA FÁBRICA ⚠️")

        st.markdown("---")

    st.markdown("© autor yo we  · unab 2025 · © Derechos reservados")
