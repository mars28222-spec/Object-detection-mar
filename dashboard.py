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
# 🎨 Custom CSS untuk Background
# ==========================
st.markdown(
    """
    <style>
    /* BG untuk judul utama */
    .stApp > header, .stApp > div.block-container {
        background-color: #e6f7ff;
    }
    /* Card BG untuk judul dan menu sidebar */
    .custom-title {
        background-color: #99d6ff;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
    }
    .custom-sidebar {
        background-color: #cceeff;
        padding: 10px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# 🎨 UI Utama
# ==========================
st.markdown('<div class="custom-title"><h1>🍴🔍 SmartVision AI Dashboard</h1></div>', unsafe_allow_html=True)
st.markdown(
    """
Selamat datang di *SmartVision*! Aplikasi ini dirancang untuk memudahkan analisis gambar dengan dua fitur utama:  
1️⃣ **Deteksi Objek (YOLO)** → mengenali sendok dan garpu.  
2️⃣ **Klasifikasi Retakan (CNN)** → membedakan permukaan normal vs retakan.
""",
    unsafe_allow_html=True
)

# ==========================
# 🧩 Sidebar & Menu dengan BG
# ==========================
st.sidebar.markdown('<div class="custom-sidebar">', unsafe_allow_html=True)
menu = st.sidebar.selectbox("Pilih Mode Analisis:", 
                            ["Deteksi Sendok & Garpu (YOLO)", "Klasifikasi Retakan (CNN)"])
uploaded_file = st.sidebar.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# ==========================
# ⏳ Fungsi Loading Interaktif
# ==========================
def loading_animation(task_name="Memproses"):
    with st.spinner(f"{task_name}... Mohon tunggu! ⏳"):
        time.sleep(1.5)

# ==========================
# 🖼 Tampilkan Gambar & Proses Analisis
# ==========================
MAX_DISPLAY_WIDTH = 500  # Batas lebar gambar

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    
    # Resize gambar untuk tampilan agar tidak terlalu besar
    display_img = img.copy()
    display_img.thumbnail((MAX_DISPLAY_WIDTH, MAX_DISPLAY_WIDTH))
    
    st.image(display_img, caption="Gambar yang Diupload", use_column_width=False)
    st.divider()
    
    # =======================================
    # 🍴 Mode 1 - Deteksi Sendok & Garpu
    # =======================================
    if menu == "Deteksi Sendok & Garpu (YOLO)":
        st.subheader("🔎 Hasil Deteksi Objek")
        loading_animation("Mendeteksi objek")
        results = yolo_model(img)
        result_img = results[0].plot()
        
        # Resize hasil deteksi juga agar tidak terlalu besar
        result_display = Image.fromarray(result_img)
        result_display.thumbnail((MAX_DISPLAY_WIDTH, MAX_DISPLAY_WIDTH))
        
        st.image(result_display, caption="Hasil Deteksi YOLO", use_column_width=False)
        
        detections = results[0].boxes
        if len(detections) > 0:
            st.success(f"✅ Terdeteksi {len(detections)} objek dalam gambar!")
            for i, box in enumerate(detections):
                cls = int(box.cls)
                conf = float(box.conf)
                label = yolo_model.names[cls] if hasattr(yolo_model, 'names') else f"Kelas {cls}"
                st.write(f"*Objek {i+1}:* {label} (Confidence: {conf:.2f})")
        else:
            st.warning("⚠️ Tidak ada objek yang terdeteksi. Coba unggah gambar lain.")

    # =======================================
    # 🧱 Mode 2 - Klasifikasi Retakan
    # =======================================
    elif menu == "Klasifikasi Retakan (CNN)":
        st.subheader("🧠 Hasil Klasifikasi Gambar")
        loading_animation("Memprediksi gambar")
        
        target_size = classifier.input_shape[1:3]
        img_resized = img.resize(target_size)
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = classifier.predict(img_array)[0][0]
        confidence = prediction if prediction >= 0.5 else 1 - prediction
        predicted_label = "Retakan" if prediction >= 0.5 else "Bukan Retakan"

        display_resized = img_resized.copy()
        display_resized.thumbnail((MAX_DISPLAY_WIDTH, MAX_DISPLAY_WIDTH))
        
        st.image(display_resized, caption="🖼 Gambar yang Diprediksi", use_column_width=False)
        st.success(f"*Prediksi:* {predicted_label}")
        st.write(f"*Tingkat Keyakinan Model:* {confidence*100:.2f}%")
