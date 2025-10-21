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
# 🎨 Custom CSS & Background Gambar
# ==========================
HEADER_BG = "https://www.canva.com/templates/EAGQFcE0rJ0//header-bg.png"  # Ganti URL dengan gambar Canva
SIDEBAR_BG = "https://www.canva.com/templates/EAGQFcE0rJ0//sidebar-bg.png"  # Ganti URL dengan gambar Canva

st.markdown(
    f"""
    <style>
    /* Header dengan BG gambar */
    .stApp > header {{
        background-image: url({HEADER_BG});
        background-size: cover;
        background-position: center;
        height: 150px;
    }}
    .custom-title {{
        background-image: url({HEADER_BG});
        background-size: cover;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 30px;
        font-weight: bold;
    }}
    /* Sidebar dengan BG gambar */
    .custom-sidebar {{
        background-image: url({SIDEBAR_BG});
        background-size: cover;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 10px;
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# 🎨 UI Utama
# ==========================
st.markdown('<div class="custom-title">🍴🔍 SmartVision AI Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    """
Selamat datang di *SmartVision*! Aplikasi ini dirancang untuk memudahkan analisis gambar dengan dua fitur utama:  
1️⃣ **Deteksi Objek (YOLO)** → mengenali sendok dan garpu.  
2️⃣ **Klasifikasi Retakan (CNN)** → membedakan permukaan normal vs retakan.
""",
    unsafe_allow_html=True
)

# ==========================
# 🧩 Sidebar dengan BG gambar
# ==========================
st.sidebar.markdown('<div class="custom-sidebar"><h3 style="text-align:center;">Pilih Metode Analisis</h3></div>', unsafe_allow_html=True)
menu = st.sidebar.selectbox("", ["Deteksi Sendok & Garpu (YOLO)", "Klasifikasi Retakan (CNN)"])

st.sidebar.markdown('<div class="custom-sidebar"><h3 style="text-align:center;">Unggah Gambar</h3></div>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png"])

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
    
    # Resize gambar untuk tampilan
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
else:
    st.info("📸 Silakan unggah gambar di sidebar untuk memulai analisis.")
