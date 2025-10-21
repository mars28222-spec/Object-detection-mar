import streamlit as st
import os
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from io import BytesIO
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
# 🎨 UI Utama
# ==========================
st.set_page_config(page_title="SmartVision AI Dashboard", page_icon="🍴", layout="wide")
st.title("🍴🔍 SmartVision: Deteksi & Klasifikasi Gambar Cerdas")
st.markdown(
    """
Selamat datang di *SmartVision*! Aplikasi ini dirancang untuk memudahkan analisis gambar dengan dua fitur utama:  
1️⃣ **Deteksi Objek (YOLO)** → mengenali sendok dan garpu.  
2️⃣ **Klasifikasi Retakan (CNN)** → membedakan permukaan normal vs retakan.  
Unggah gambar di bawah dan pilih mode analisis! 🚀
"""
)

# ==========================
# 🧩 Sidebar & Menu
# ==========================
menu = st.sidebar.selectbox("Pilih Mode Analisis:", 
                            ["Deteksi Sendok & Garpu (YOLO)", "Klasifikasi Retakan (CNN)"])
uploaded_file = st.sidebar.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# ⏳ Fungsi Loading Interaktif
# ==========================
def loading_animation(task_name="Memproses"):
    with st.spinner(f"{task_name}... Mohon tunggu! ⏳"):
        time.sleep(1.5)

# ==========================
# 🖼 Tampilkan Gambar & Proses Analisis
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang Diupload", use_column_width=True)
    st.divider()
    
    # =======================================
    # 🍴 Mode 1 - Deteksi Sendok & Garpu
    # =======================================
    if menu == "Deteksi Sendok & Garpu (YOLO)":
        st.subheader("🔎 Hasil Deteksi Objek")
        loading_animation("Mendeteksi objek")
        results = yolo_model(img)
        result_img = results[0].plot()
        st.image(result_img, caption="Hasil Deteksi YOLO", use_column_width=True)
        
        detections = results[0].boxes
        if len(detections) > 0:
            st.success(f"✅ Terdeteksi {len(detections)} objek dalam gambar!")
            for i, box in enumerate(detections):
                cls = int(box.cls)
                conf = float(box.conf)
                label = yolo_model.names[cls] if hasattr(yolo_model, 'names') else f"Kelas {cls}"

                st.write(f"*Objek {i+1}:* {label} (Confidence: {conf:.2f})")

                # 🎨 Feedback kreatif berdasarkan label
                if "sendok" in label.lower():
                    st.markdown("🥄 Wah, ada *sendok elegan* siap menyendok hidangan!")
                elif "garpu" in label.lower():
                    st.markdown("🍴 Terlihat *garpu gagah* menemani sendoknya ✨")
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

        st.image(img_resized, caption="🖼 Gambar yang Diprediksi", use_column_width=True)
        st.success(f"*Prediksi:* {predicted_label}")
        st.write(f"*Tingkat Keyakinan Model:* {confidence*100:.2f}%")

        if predicted_label == "Retakan":
            st.markdown("🧱 Terlihat ada *retakan!* Perlu diperhatikan 💥")
        else:
            st.markdown("✅ Permukaannya *halus dan kuat*, aman 💪")

# ==========================
# ⚠️ Jika Belum Upload Gambar
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
- Gunakan gambar dengan resolusi jelas untuk hasil terbaik.  
- Pilih mode sesuai kebutuhan: deteksi objek atau klasifikasi retakan.  
- Bersabar sebentar saat model memproses gambar, terutama YOLO yang membutuhkan resource lebih besar.
"""
)
