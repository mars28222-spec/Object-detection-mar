import streamlit as st
import os
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time

# ==========================
# 🔹 Setup Environment
# ==========================
os.system("apt-get update -y && apt-get install -y libgl1 libglib2.0-0")

# ==========================
# 🔹 Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Siti Marlina_Laporan 4.pt")  
    classifier = tf.keras.models.load_model("model/Siti Marlina_laporan 2 (1).h5")  
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# 🎨 Custom CSS untuk Background & Warna
# ==========================
st.markdown(
    """
    <style>
    /* Background seluruh dashboard */
    .stApp {
        background-color: #FFFDD0;
    }

    /* Judul utama */
    .custom-title {
        background-color: #99d6ff;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: white;
    }

    /* Sidebar section */
    .custom-sidebar {
        background-color: #cceeff;
        padding: 10px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 10px;
        color: #003366;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# 🎨 UI Utama
# ==========================
st.set_page_config(page_title="SmartVision AI Dashboard", page_icon="🍴", layout="wide")
st.markdown('<div class="custom-title">🍴🔍 SmartVision: Deteksi & Klasifikasi Gambar Cerdas</div>', unsafe_allow_html=True)
st.markdown(
    """
Selamat datang di SmartVision! Unggah gambar di sidebar dan pilih mode analisis.  
Preview gambar kecil akan tampil di atas, hasil prediksi/deteksi di bawah.
""",
    unsafe_allow_html=True
)

# ==========================
# 🧩 Sidebar dengan warna & background
# ==========================
st.sidebar.markdown('<div class="custom-sidebar">Pilih Mode Analisis</div>', unsafe_allow_html=True)
menu = st.sidebar.selectbox("", ["Deteksi Sendok & Garpu (YOLO)", "Klasifikasi Retakan (CNN)"])

st.sidebar.markdown('<div class="custom-sidebar">Unggah Gambar (Bisa lebih dari 1)</div>', unsafe_allow_html=True)
uploaded_files = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# ==========================
# ⏳ Fungsi Loading Interaktif
# ==========================
def loading_animation(task_name="Memproses"):
    with st.spinner(f"{task_name}... Mohon tunggu! ⏳"):
        time.sleep(1.5)

# ==========================
# 🖼 Tampilkan Gambar & Proses Analisis
# ==========================
MAX_PREVIEW = 250
MAX_RESULT = 600

if uploaded_files:
    preview_imgs = []
    result_imgs = []
    result_labels = []

    # Proses semua gambar
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file).convert("RGB")

        # Preview kecil
        preview_img = img.copy()
        preview_img.thumbnail((MAX_PREVIEW, MAX_PREVIEW))
        preview_imgs.append(preview_img)

        # Proses sesuai mode
        if menu == "Deteksi Sendok & Garpu (YOLO)":
            loading_animation("Mendeteksi objek")
            results = yolo_model(img)
            result_img = results[0].plot()
            result_display = Image.fromarray(result_img)
            result_display.thumbnail((MAX_RESULT, MAX_RESULT))
            result_imgs.append(result_display)

            # Simpan label deteksi
            detections = results[0].boxes
            labels = []
            if len(detections) > 0:
                for i, box in enumerate(detections):
                    cls = int(box.cls)
                    label = yolo_model.names[cls] if hasattr(yolo_model, 'names') else f"Kelas {cls}"
                    labels.append(f"{label} (Conf: {box.conf:.2f})")
            else:
                labels.append("Tidak ada objek terdeteksi")
            result_labels.append(labels)

        elif menu == "Klasifikasi Retakan (CNN)":
            loading_animation("Memprediksi gambar")
            target_size = classifier.input_shape[1:3]
            img_resized = img.resize(target_size)
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)/255.0
            prediction = classifier.predict(img_array)[0][0]
            predicted_label = "Retakan" if prediction>=0.5 else "Bukan Retakan"
            confidence = prediction if prediction >=0.5 else 1 - prediction

            display_resized = img_resized.copy()
            display_resized.thumbnail((MAX_RESULT, MAX_RESULT))
            result_imgs.append(display_resized)
            result_labels.append([f"{predicted_label} ({confidence*100:.2f}%)"])

    # ==========================
    # Baris Preview Gambar (horizontal)
    # ==========================
    st.subheader("Preview Gambar Upload")
    cols_preview = st.columns(len(preview_imgs))
    for i, col in enumerate(cols_preview):
        col.image(preview_imgs[i], caption=f"Gambar {i+1}", use_column_width=False)

    st.divider()

    # ==========================
    # Baris Hasil Prediksi / Deteksi (horizontal)
    # ==========================
    st.subheader("Hasil Prediksi / Deteksi")
    cols_result = st.columns(len(result_imgs))
    for i, col in enumerate(cols_result):
        col.image(result_imgs[i], caption=f"Hasil Gambar {i+1}", use_column_width=False)
        # Tampilkan label / feedback
        for label_text in result_labels[i]:
            col.markdown(f"**{label_text}**")

else:
    st.info("📸 Silakan unggah gambar di sidebar untuk memulai analisis.")
