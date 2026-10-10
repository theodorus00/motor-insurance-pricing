import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Tentang Proyek | Motor Insurance Pricing",
    page_icon="📘",
    layout="wide",
)

st.title("📘 Tentang Proyek")
st.subheader("Estimasi Premi Risiko Tahunan Asuransi Kendaraan")

st.header("Latar belakang")

st.write(
    "Perusahaan asuransi memerlukan estimasi biaya klaim untuk "
    "menentukan premi yang sesuai dengan risiko peserta. "
    "Risiko tersebut dipengaruhi oleh seberapa sering klaim "
    "terjadi dan seberapa besar biaya setiap klaim."
)

st.write(
    "Proyek ini menggunakan data historis polis dan klaim "
    "untuk mempelajari hubungan karakteristik pengemudi, kendaraan, "
    "serta wilayah dengan ekspektasi biaya klaim."
)

st.header("Tujuan")

st.markdown(
    """
    - Mengestimasi frekuensi klaim per tahun pertanggungan.
    - Mengestimasi rata-rata biaya per klaim.
    - Menghitung premi risiko tahunan melalui Frequency × Severity.
    - Membandingkan hasilnya dengan prediksi langsung model Tweedie.
    - Menyediakan simulasi interaktif untuk mengeksplorasi hasil model.
    """
)

st.header("Apa yang dimaksud premi risiko?")

st.latex(
    r"\text{Premi Risiko Tahunan}"
    r"=\mathbb{E}[\text{Biaya Klaim Selama Satu Tahun}"
    r"\mid \text{Profil Risiko}]"
)

st.write(
    "Premi risiko merupakan estimasi rata-rata biaya klaim "
    "untuk profil tertentu. Nilai ini belum mencakup biaya "
    "operasional, komisi, dan margin keuntungan."
)

st.info(
    "Ekspektasi biaya bukan jaminan bahwa premi akan menutup "
    "seluruh klaim aktual. Klaim yang terjadi tetap dapat "
    "berbeda dari estimasi."
)

st.header("Dataset")

st.write(
    "Proyek menggunakan freMTPL2, yaitu data historis "
    "motor third-party liability di Prancis. Cakupannya "
    "berkaitan dengan tanggung jawab kepada pihak ketiga, "
    "bukan otomatis kerusakan kendaraan milik tertanggung."
)

st.markdown(
    """
    | File | Isi |
    |---|---|
    | `freMTPL2freq.csv` | Karakteristik polis, exposure, dan jumlah klaim |
    | `freMTPL2sev.csv` | Nilai biaya pada setiap catatan klaim |

    Kedua tabel dihubungkan menggunakan **IDpol**.
    Nilai biaya dalam proyek ini menggunakan **EUR**.
    """
)

st.header("Penjelasan variabel")

dictionary = pd.DataFrame([
    {
        "Variabel": "IDpol",
        "Penjelasan": "Identitas polis untuk menghubungkan data.",
        "Peran": "Identitas, bukan fitur prediksi",
    },
    {
        "Variabel": "ClaimNb",
        "Penjelasan": "Jumlah klaim selama periode exposure.",
        "Peran": "Pembentukan target frekuensi",
    },
    {
        "Variabel": "Exposure",
        "Penjelasan": (
            "Lama pertanggungan dalam tahun; "
            "0,5 berarti sekitar setengah tahun."
        ),
        "Peran": "Normalisasi target dan bobot",
    },
    {
        "Variabel": "Area",
        "Penjelasan": "Kode kategori area pada dataset.",
        "Peran": "Fitur kategori",
    },
    {
        "Variabel": "VehPower",
        "Penjelasan": (
            "Kode tingkat tenaga kendaraan; "
            "bukan nilai langsung dalam HP atau kW."
        ),
        "Peran": "Fitur numerik",
    },
    {
        "Variabel": "VehAge",
        "Penjelasan": "Usia kendaraan dalam tahun.",
        "Peran": "Fitur numerik",
    },
    {
        "Variabel": "DrivAge",
        "Penjelasan": "Usia pengemudi dalam tahun.",
        "Peran": "Fitur numerik",
    },
    {
        "Variabel": "BonusMalus",
        "Penjelasan": (
            "Indeks bonus–malus: di bawah 100 menunjukkan bonus, "
            "di atas 100 menunjukkan malus."
        ),
        "Peran": "Fitur, harus tersedia sebelum periode prediksi",
    },
    {
        "Variabel": "VehBrand",
        "Penjelasan": "Kode merek kendaraan yang disamarkan.",
        "Peran": "Fitur kategori",
    },
    {
        "Variabel": "VehGas",
        "Penjelasan": "Jenis bahan bakar kendaraan.",
        "Peran": "Fitur kategori",
    },
    {
        "Variabel": "Density",
        "Penjelasan": "Kepadatan penduduk per kilometer persegi.",
        "Peran": "Fitur numerik, ditransformasi dengan log1p",
    },
    {
        "Variabel": "Region",
        "Penjelasan": "Kode wilayah di Prancis.",
        "Peran": "Fitur kategori",
    },
    {
        "Variabel": "ClaimAmount",
        "Penjelasan": "Biaya satu catatan klaim dalam EUR.",
        "Peran": "Pembentukan target severity dan biaya total",
    },
])

st.dataframe(dictionary, hide_index=True)

st.header("Bagaimana data disiapkan?")

st.markdown(
    """
    1. **Audit validitas:** memeriksa ID, exposure, jumlah klaim,
       biaya klaim, dan nilai fitur.
    2. **Rekonsiliasi:** mencocokkan jumlah klaim pada data polis
       dengan jumlah catatan pada data severity.
    3. **Pembentukan cohort:** menggunakan polis tanpa klaim yang
       konsisten serta polis berklaim yang berhasil direkonsiliasi.
    4. **Pembagian data:** memisahkan train, validation, dan test
       berdasarkan polis.
    5. **Pemodelan dan evaluasi:** memilih pendekatan pada validation,
       lalu mengevaluasi model final pada test.
    """
)

st.warning(
    "Polis yang memiliki klaim tetapi tidak memiliki catatan severity "
    "tidak dianggap memiliki biaya nol. Polis yang tidak berhasil "
    "direkonsiliasi dikeluarkan dari cohort pemodelan. "
    "Keputusan ini dapat menimbulkan selection bias."
)

st.header("Cara menggunakan website")

st.markdown(
    """
    - Buka **EDA** untuk memahami karakteristik data train.
    - Buka **Model dan Evaluasi** untuk melihat metode dan performanya.
    - Buka **Simulasi Premi** untuk memasukkan profil dan membandingkan
      hasil Frequency × Severity dengan Tweedie.
    """
)

st.caption(
    "Proyek ini berfokus pada estimasi premi risiko. "
    "Perhitungan cadangan klaim atau cadangan premi "
    "belum menjadi bagian dari aplikasi."
)