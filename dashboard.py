import streamlit as st
import os
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# 🌿 Tambahkan Background
# ==========================
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

# ==========================
# 🔹 Setup environment (opsional di Streamlit Cloud)
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

                # 🎨 Tambahkan teks kreatif
                if "sendok" in label.lower():
                    st.markdown("🥄 Wah, ada **sendok elegan** di sini! Siap menyendok makanan lezat 🍜")
                elif "garpu" in label.lower():
                    st.markdown("🍴 Terlihat **garpu tajam nan gagah** siap menemani sendoknya ✨")
        else:
            st.info("Tidak ada objek yang terdeteksi dalam gambar ini.")

    # =======================================
    # 🧱 Mode 2 - Klasifikasi Retakan
    # =======================================
    elif menu == "Klasifikasi Retakan (CNN)":
        st.subheader("🧠 Hasil Klasifikasi Gambar")

        target_size = classifier.input_shape[1:3]
        img_resized = img.resize(target_size)
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = classifier.predict(img_array)[0][0]
        confidence = prediction if prediction >= 0.5 else 1 - prediction

        if prediction >= 0.5:
            predicted_label = "Retakan"
        else:
            predicted_label = "Bukan Retakan"

        st.image(img_resized, caption="🖼️ Gambar yang Diprediksi", use_column_width=True)
        st.success(f"**Prediksi:** {predicted_label}")
        st.write(f"**Tingkat Keyakinan Model:** {confidence*100:.2f}%")

        if predicted_label == "Retakan":
            st.markdown("🧱 Terlihat ada **retakan!** Mungkin waktunya perbaikan 💥")
        else:
            st.markdown("✅ Permukaannya **halus dan kuat**, tidak ada retakan berarti 💪")

else:
    st.info("📸 Silakan unggah gambar terlebih dahulu untuk memulai analisis.")
