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
# 🎨 Custom CSS untuk Warna & Background
# ==========================
st.markdown(
    """
    <style>
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
st.set_page_config(page_title="Mars AI Dashboard", page_icon="🍴", layout="wide")
st.markdown('<div class="custom-title">🍴🔍 Mars AI: Deteksi & Klasifikasi Gambar Cerdas</div>', unsafe_allow_html=True)
st.markdown(
    """
Selamat datang di Mars AI! Aplikasi ini memudahkan analisis gambar dengan dua fitur utama:  
1️⃣ *Deteksi Objek (YOLO)* → mengenali sendok dan garpu.  
2️⃣ *Klasifikasi Retakan (CNN)* → membedakan permukaan normal vs retakan.  
Unggah gambar di sidebar dan pilih mode analisis! 🚀
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
MAX_PREVIEW = 300  # preview kecil
MAX_RESULT = 800   # hasil prediksi/deteksi besar

if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        img = Image.open(uploaded_file).convert("RGB")

        # Preview kecil
        preview_img = img.copy()
        preview_img.thumbnail((MAX_PREVIEW, MAX_PREVIEW))
        st.image(preview_img, caption=f"Preview Gambar {idx+1}", use_column_width=False)
        st.divider()

        # =======================================
        # 🍴 Mode 1 - Deteksi Sendok & Garpu
        # =======================================
        if menu == "Deteksi Sendok & Garpu (YOLO)":
            st.subheader(f"🔎 Hasil Deteksi Objek Gambar {idx+1}")
            loading_animation("Mendeteksi objek")
            results = yolo_model(img)
            result_img = results[0].plot()
            result_display = Image.fromarray(result_img)
            result_display.thumbnail((MAX_RESULT, MAX_RESULT))
            st.image(result_display, caption=f"Hasil Deteksi YOLO Gambar {idx+1}", use_column_width=False)
            
            detections = results[0].boxes
            if len(detections) > 0:
                st.success(f"✅ Terdeteksi {len(detections)} objek!")
                for i, box in enumerate(detections):
                    cls = int(box.cls)
                    conf = float(box.conf)
                    label = yolo_model.names[cls] if hasattr(yolo_model, 'names') else f"Kelas {cls}"
                    st.write(f"Objek {i+1}: {label} (Confidence: {conf:.2f})")
                    if "sendok" in label.lower():
                        st.markdown("🥄 Wah, ada sendok elegan siap menyendok hidangan!")
                    elif "garpu" in label.lower():
                        st.markdown("🍴 Terlihat garpu gagah menemani sendoknya ✨")
            else:
                st.warning("⚠ Tidak ada objek yang terdeteksi.")

        # =======================================
        # 🧱 Mode 2 - Klasifikasi Retakan
        # =======================================
        elif menu == "Klasifikasi Retakan (CNN)":
            st.subheader(f"🧠 Hasil Klasifikasi Gambar {idx+1}")
            loading_animation("Memprediksi gambar")
            
            target_size = classifier.input_shape[1:3]
            img_resized = img.resize(target_size)
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)/255.0

            prediction = classifier.predict(img_array)[0][0]
            confidence = prediction if prediction >= 0.5 else 1 - prediction
            predicted_label = "Retakan" if prediction >= 0.5 else "Bukan Retakan"

            # Hasil prediksi besar
            display_resized = img_resized.copy()
            display_resized.thumbnail((MAX_RESULT, MAX_RESULT))
            st.image(display_resized, caption=f"Hasil Prediksi Gambar {idx+1}", use_column_width=False)
            st.success(f"Prediksi: {predicted_label}")
            st.write(f"Tingkat Keyakinan Model: {confidence*100:.2f}%")

            if predicted_label == "Retakan":
                st.markdown("🧱 Terlihat ada retakan! Perlu diperhatikan 💥")
            else:
                st.markdown("✅ Permukaannya halus dan kuat, aman 💪")

# ==========================
# ⚠ Jika Belum Upload Gambar
# ==========================
else:
    st.info("📸 Silakan unggah gambar di sidebar untuk memulai analisis.")

# ==========================
# 🎯 Footer & Tips UX
# ==========================
st.divider()
st.markdown(
    """
💡 Tips Penggunaan:  
- Gunakan gambar resolusi jelas untuk hasil terbaik.  
- Pilih mode sesuai kebutuhan: deteksi objek atau klasifikasi retakan.  
- Bersabar saat model memproses gambar, terutama YOLO.
"""
)
