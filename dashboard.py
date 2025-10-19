import streamlit as st

# ==============================
# 🌿 Background seluruh halaman
# ==============================
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://www.canva.com/design/DAFxxxxx/view?utm_content=DAFxxxxx&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()


# ==============================
# 💛 Styling area tertentu
# ==============================
st.markdown(
    """
    <style>
    /* ====== Judul ====== */
    .title-container {
        background-image: url("https://www.canva.com/design/DAFxxxxx/view?utm_content=DAFxxxxx&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton");
        background-size: cover;
        background-position: center;
        padding: 15px;
        border-radius: 15px;
        text-align: center;
        color: black;
        font-weight: bold;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 25px;
    }

    /* ====== Sidebar ====== */
    section[data-testid="stSidebar"] {
        background-image: url("https://www.canva.com/design/DAFxxxxx/view?utm_content=DAFxxxxx&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton");
        background-size: cover;
        background-position: center;
        border-right: 2px solid #dcdcdc;
    }

    /* ====== Area Upload ====== */
    [data-testid="stFileUploaderDropzone"] {
        background-image: url("https://www.canva.com/design/DAFxxxxx/view?utm_content=DAFxxxxx&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton");
        background-size: cover;
        background-position: center;
        border: 2px dashed #ffcc00;
        border-radius: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==============================
# 🎯 Judul
# ==============================
st.markdown(
    """
    <div class="title-container">
        <h1>🍴 SmartVision: Deteksi Garpu, Sendok, dan Retakan</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================
# 📸 Upload Gambar
# ==============================
uploaded_file = st.file_uploader("Upload gambar di sini...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Gambar yang diunggah', use_column_width=True)
    st.success("✅ Gambar berhasil diunggah!")

    # Contoh logika klasifikasi (dummy)
    prediksi = "garpu"  # ini nanti dari model kamu

    # ✨ Pesan kreatif sesuai hasil
    if prediksi == "garpu":
        st.markdown("🥄 **Hebat! Ini garpu tajam siap menemani makan malam eleganmu!**")
    elif prediksi == "sendok":
        st.markdown("🍽️ **Sendok lembut siap menyendok kelezatan setiap hidangan!**")
    elif prediksi == "retak":
        st.markdown("⚠️ **Hmm... sepertinya ada retakan kecil, hati-hati ya saat digunakan!**")
    else:
        st.markdown("✅ **Tidak retak — sempurna dan siap digunakan!**")
