import streamlit as st


st.set_page_config(
    page_title="Jasen Theodorus | Motor Insurance Pricing",
    page_icon="🚗",
    layout="wide",
)

st.sidebar.title("🚗 Motor Insurance Pricing")
st.sidebar.caption("Portfolio · Jasen Theodorus")

st.title("Motor Insurance Risk Pricing")
st.subheader("Mengubah data klaim menjadi estimasi premi risiko")

st.markdown(
    """
    Halo, saya **Jasen Theodorus**.

    Saya memiliki latar belakang **Ilmu Aktuaria** dan pengalaman
    dalam pengembangan produk serta analisis di industri perbankan.

    Melalui proyek ini, saya menggabungkan pendekatan aktuaria
    dan machine learning untuk mengestimasi **premi risiko tahunan
    asuransi kendaraan bermotor**.
    """
)

st.divider()

st.header("Apa yang dipelajari dalam proyek ini?")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 Frekuensi")
    st.write(
        "Mengestimasi jumlah klaim per tahun menggunakan "
        "model dengan loss Poisson."
    )

with col2:
    st.subheader("💶 Severity")
    st.write(
        "Mengestimasi rata-rata biaya per klaim menggunakan "
        "model dengan loss Gamma."
    )

with col3:
    st.subheader("🚗 Premi Risiko")
    st.write(
        "Membandingkan hasil Frequency × Severity dengan "
        "prediksi langsung dari model Tweedie."
    )

st.latex(
    r"\text{Premi Risiko Tahunan}"
    r" = \text{Frekuensi Tahunan}"
    r" \times \text{Severity}"
)

st.header("Jelajahi proyek")

st.markdown(
    """
    - **Tentang Proyek:** tujuan bisnis, dataset, dan definisi variabel.
    - **EDA:** eksplorasi karakteristik polis dan pengalaman klaim.
    - **Model dan Evaluasi:** proses pemodelan serta hasil pengujian.
    - **Simulasi Premi:** masukkan profil untuk melihat estimasi model.
    """
)

st.info(
    "Proyek menggunakan data historis motor third-party liability "
    "Prancis (freMTPL2), dengan nilai biaya dalam EUR. "
    "Model dilatih pada polis yang lolos rekonsiliasi data. "
    "Hasil merupakan estimasi premi risiko untuk demonstrasi, "
    "belum termasuk biaya operasional, komisi, dan margin keuntungan."
)

st.caption("Jasen Theodorus · Actuarial & Data Science Portfolio")