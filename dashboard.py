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
# Inisialisasi session_state
# ==========================
if 'preview_imgs' not in st.session_state:
    st.session_state.preview_imgs = []
if 'result_imgs' not in st.session_state:
    st.session_state.result_imgs = []
if 'result_labels' not in st.session_state:
    st.session_state.result_labels = []

# ==========================
# Fungsi reset session_state
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
st.sidebar.markdown('<div style="background-color:#87CEFA;padding:10px;border-radius:10px;text-align:center;font-weight:bold;">Unggah Gambar</div>', unsafe_allow_html=True)

# ==========================
# Tombol refresh
# ==========================
if st.sidebar.button("🔄 Refresh"):
    refresh_dashboard()
    st.sidebar.info("Silakan refresh untuk memprediksi gambar baru.")

# ==========================
# Upload satu gambar
# ==========================
uploaded_file = st.sidebar.file_uploader("", type=["jpg","jpeg","png"], accept_multiple_files=False)

# ==========================
# Fungsi loading
# ==========================
def loading_animation(task_name="Memproses"):
    with st.spinner(f"{task_name}... Mohon tunggu! ⏳"):
        time.sleep(1.5)

# ==========================
# Konfigurasi ukuran
# ==========================
MAX_PREVIEW = 250
RESULT_WIDTH = 800
RESULT_HEIGHT = 600

# ==========================
# Proses upload & prediksi
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    # Preview kecil
    preview_img = img.copy()
    preview_img.thumbnail((MAX_PREVIEW, MAX_PREVIEW))
    st.session_state.preview_imgs = [preview_img]

    # Proses prediksi
    if menu == "Deteksi Sendok & Garpu (YOLO)":
        loading_animation("Mendeteksi objek")
        results = yolo_model(img)
        result_img = results[0].plot()
        result_display = Image.fromarray(result_img)
        result_display = result_display.resize((RESULT_WIDTH, RESULT_HEIGHT))
        st.session_state.result_imgs = [result_display]

        labels = []
        for box in results[0].boxes:
            cls = int(box.cls)
            label = yolo_model.names[cls] if hasattr(yolo_model,'names') else f"Kelas {cls}"
            labels.append(f"{label} (Conf: {box.conf:.2f})")
        if not labels:
            labels.append("Tidak ada objek terdeteksi")
        st.session_state.result_labels = [labels]

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
        st.session_state.result_imgs = [display_resized]
        st.session_state.result_labels = [[f"{predicted_label} ({confidence*100:.2f}%)"]]

# ==========================
# Tampilkan Preview & Hasil
# ==========================
if st.session_state.preview_imgs:
    st.subheader("Preview Gambar Upload")
    st.columns([1])[0].image(st.session_state.preview_imgs[0], caption="Preview", use_container_width=False)

if st.session_state.result_imgs:
    st.subheader("Hasil Prediksi / Deteksi")
    st.columns([1])[0].image(st.session_state.result_imgs[0], caption="Hasil Prediksi", use_container_width=False)
    for label_text in st.session_state.result_labels[0]:
        st.markdown(f"**{label_text}**")

# ==========================
# Pesan jika belum upload
# ==========================
if uploaded_file is None:
    st.info("📸 Silakan unggah gambar di sidebar untuk memulai analisis.")
