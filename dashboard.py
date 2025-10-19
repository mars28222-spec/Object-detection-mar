def add_bg_from_url(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Contoh pakai gambar daun muda
add_bg_from_url("https://images.unsplash.com/photo-1501004318641-b39e6451bec6?auto=format&fit=crop&w=1200&q=80")



import os
os.system("apt-get update -y && apt-get install -y libgl1 libglib2.0-0")

import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# 🔹 Load Models
# ==========================
@st.cache_resource
def load_models():
    # YOLO untuk deteksi sendok dan garpu
    yolo_model = YOLO("model/Siti Marlina_Laporan 4.pt")  
    
    # Model CNN untuk klasifikasi retakan vs bukan retakan
    classifier = tf.keras.models.load_model("model/Siti Marlina_laporan 2 (1).h5")  
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# 🎨 UI
# ==========================
st.title("🍴🔍 SmartVision: Deteksi & Klasifikasi Gambar Cerdas")
st.markdown(
    """
    Selamat datang di **SmartVision**, aplikasi berbasis kecerdasan buatan yang siap membantu kamu menganalisis gambar secara otomatis! 🤖  
    Aplikasi ini memiliki dua fitur unggulan:
    
    - 🍽️ **Deteksi Objek (YOLO)** → Mengenali keberadaan **sendok** dan **garpu** dalam gambar secara cepat dan akurat.  
    - 🧱 **Klasifikasi Gambar (CNN)** → Membedakan antara **retakan** dan **permukaan normal** menggunakan teknologi *deep learning*.
    
    Unggah gambar favoritmu dan biarkan AI bekerja! 🚀
    """
)

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Sendok & Garpu (YOLO)", "Klasifikasi Retakan (CNN)"])

uploaded_file = st.file_uploader("📤 Unggah gambar di sini", type=["jpg", "jpeg", "png"])

# ==========================
# 🖼️ Tampilkan gambar
# ==========================
if uploaded_file is not None:
    # 🔹 Pastikan gambar hanya punya 3 channel (RGB)
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)
    st.divider()

    # =======================================
    # 🍴 Mode 1 - Deteksi Sendok & Garpu
    # =======================================
    if menu == "Deteksi Sendok & Garpu (YOLO)":
        st.subheader("🔎 Hasil Deteksi Objek")
        results = yolo_model(img)
        result_img = results[0].plot()
        st.image(result_img, caption="Hasil Deteksi YOLO", use_container_width=True)

        detections = results[0].boxes
        if len(detections) > 0:
            for i, box in enumerate(detections):
                cls = int(box.cls)
                conf = float(box.conf)
                label = yolo_model.names[cls] if hasattr(yolo_model, 'names') else f"Kelas {cls}"
                st.write(f"**Objek {i+1}:** {label} (Confidence: {conf:.2f})")
        else:
            st.info("Tidak ada objek yang terdeteksi dalam gambar ini.")

    # =======================================
    # 🧱 Mode 2 - Klasifikasi Retakan
    # =======================================
    elif menu == "Klasifikasi Retakan (CNN)":
        st.subheader("🧠 Hasil Klasifikasi Gambar")

        # 🔹 Preprocessing gambar (otomatis sesuai ukuran input model)
        target_size = classifier.input_shape[1:3]
        img_resized = img.resize(target_size)
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # 🔹 Prediksi (untuk model biner sigmoid)
        prediction = classifier.predict(img_array)[0][0]
        confidence = prediction if prediction >= 0.5 else 1 - prediction

        if prediction >= 0.5:
            predicted_label = "Retakan"
        else:
            predicted_label = "Bukan Retakan"

        # 🔹 Tampilkan hasil
        st.image(img_resized, caption="🖼️ Gambar yang Diprediksi", use_column_width=True)
        st.success(f"**Prediksi:** {predicted_label}")
        st.write(f"**Tingkat Keyakinan Model:** {confidence*100:.2f}%")
        st.write(f"**Nilai Probabilitas (Sigmoid Output):** {prediction:.4f}")

        # 🔹 Penjelasan tambahan
        if predicted_label == "Retakan":
            st.markdown("🧱 Gambar ini **terdeteksi mengandung retakan**. Perlu diperiksa lebih lanjut.")
        else:
            st.markdown("✅ Gambar ini **tidak menunjukkan adanya retakan yang signifikan.**")

# Jika belum upload
else:
    st.info("📸 Silakan unggah gambar terlebih dahulu untuk memulai analisis.")
