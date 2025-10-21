import streamlit as st
from PIL import Image
import numpy as np
import time
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# ==========================
# Setup environment
# ==========================
os.system("apt-get update -y && apt-get install -y libgl1 libglib2.0-0")

# ==========================
# Load models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Siti Marlina_Laporan 4.pt")  
    classifier = tf.keras.models.load_model("model/Siti Marlina_laporan 2 (1).h5")  
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Session state
# ==========================
if 'preview_imgs' not in st.session_state:
    st.session_state.preview_imgs = []
if 'result_imgs' not in st.session_state:
    st.session_state.result_imgs = []
if 'result_labels' not in st.session_state:
    st.session_state.result_labels = []

# ==========================
# Fungsi refresh
# ==========================
def refresh_dashboard():
    st.session_state.preview_imgs = []
    st.session_state.result_imgs = []
    st.session_state.result_labels = []

# ==========================
# UI
# ==========================
st.set_page_config(page_title="SmartVision", layout="wide")
st.markdown('<h2 style="background-color:#1E90FF;color:white;padding:10px;border-radius:10px;text-align:center;">🍴🔍 SmartVision</h2>', unsafe_allow_html=True)
st.sidebar.markdown('<div style="background-color:#87CEFA;padding:10px;border-radius:10px;text-align:center;font-weight:bold;">Pilih Mode Analisis</div>', unsafe_allow_html=True)
menu = st.sidebar.selectbox("", ["Deteksi Sendok & Garpu (YOLO)", "Klasifikasi Retakan (CNN)"])
st.sidebar.markdown('<div style="background-color:#87CEFA;padding:10px;border-radius:10px;text-align:center;font-weight:bold;">Unggah Gambar (1-2)</div>', unsafe_allow_html=True)

# ==========================
# Tombol Refresh
# ==========================
if st.sidebar.button("🔄 Refresh"):
    refresh_dashboard()
    st.sidebar.info("Silakan refresh untuk memprediksi gambar baru.")

# ==========================
# Upload 1 atau 2 gambar
# ==========================
uploaded_files = st.sidebar.file_uploader("", type=["jpg","jpeg","png"], accept_multiple_files=True)

# ==========================
# Fungsi loading
# ==========================
def loading_animation(task_name="Memproses"):
    with st.spinner(f"{task_name}... Mohon tunggu! ⏳"):
        time.sleep(1.5)

# ==========================
# Ukuran gambar
# ==========================
MAX_PREVIEW = 250
RESULT_WIDTH = 800
RESULT_HEIGHT = 600

# ==========================
# Proses upload & prediksi
# ==========================
if uploaded_files:
    # Batasi maksimal 2 gambar
    files_to_process = uploaded_files[:2]

    # Reset otomatis jika tombol refresh sudah ditekan
    st.session_state.preview_imgs = []
    st.session_state.result_imgs = []
    st.session_state.result_labels = []

    for uploaded_file in files_to_process:
        img = Image.open(uploaded_file).convert("RGB")

        # Preview kecil
        preview_img = img.copy()
        preview_img.thumbnail((MAX_PREVIEW, MAX_PREVIEW))
        st.session_state.preview_imgs.append(preview_img)

        # Proses prediksi
        if menu == "Deteksi Sendok & Garpu (YOLO)":
            loading_animation("Mendeteksi objek")
            results = yolo_model(img)
            result_img = results[0].plot()
            result_display = Image.fromarray(result_img)
            result_display = result_display.resize((RESULT_WIDTH, RESULT_HEIGHT))
            st.session_state.result_imgs.append(result_display)

            labels = []
            for box in results[0].boxes:
                cls = int(box.cls)
                label = yolo_model.names[cls] if hasattr(yolo_model,'names') else f"Kelas {cls}"
                labels.append(f"{label} (Conf: {box.conf:.2f})")
            if not labels:
                labels.append("Tidak ada objek terdeteksi")
            st.session_state.result_labels.append(labels)

        elif menu == "Klasifikasi Retakan (CNN)":
            loading_animation("Memprediksi gambar")
            target_size = classifier.input_shape[1:3]
            img_resized = img.resize(target_size)
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)/255.0
            prediction = classifier.predict(img_array)[0][0]
            predicted_label = "Retakan" if prediction>=0.5 else "Bukan Retakan"
            confidence = prediction if prediction>=0.5 else 1 - prediction

            display_resized = img_resized.copy()
            display_resized = display_resized.resize((RESULT_WIDTH, RESULT_HEIGHT))
            st.session_state.result_imgs.append(display_resized)
            st.session_state.result_labels.append([f"{predicted_label} ({confidence*100:.2f}%)"])

# ==========================
# Tampilkan Preview
# ==========================
if st.session_state.preview_imgs:
    st.subheader("Preview Gambar Upload")
    cols_preview = st.columns(len(st.session_state.preview_imgs))
    for i, col in enumerate(cols_preview):
        col.image(st.session_state.preview_imgs[i], caption=f"Gambar {i+1}", use_container_width=False)

# ==========================
# Tampilkan Hasil
# ==========================
if st.session_state.result_imgs:
    st.divider()
    st.subheader("Hasil Prediksi / Deteksi")
    cols_result = st.columns(len(st.session_state.result_imgs))
    for i, col in enumerate(cols_result):
        col.image(st.session_state.result_imgs[i], caption=f"Hasil Gambar {i+1}", use_container_width=False)
        for label_text in st.session_state.result_labels[i]:
            col.markdown(f"**{label_text}**")

# ==========================
# Pesan jika belum upload
# ==========================
if not uploaded_files:
    st.info("📸 Silakan unggah gambar di sidebar untuk memulai analisis.")
