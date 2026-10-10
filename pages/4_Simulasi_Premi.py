import pandas as pd
import streamlit as st

from src.prediction import (
    get_category_options,
    predict_premium,
)


st.set_page_config(
    page_title="Simulasi Premi | Jasen Theodorus",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Simulasi Premi Risiko")
st.write(
    "Masukkan profil pengemudi dan kendaraan untuk mengestimasi "
    "biaya klaim selama satu tahun pertanggungan."
)

st.info(
    "Simulasi menggunakan data historis asuransi tanggung jawab "
    "pihak ketiga kendaraan di Prancis. Nilai dalam EUR, "
    "belum termasuk biaya operasional dan margin keuntungan."
)

options = get_category_options()

with st.form("premium_simulation"):
    left, right = st.columns(2)

    with left:
        st.subheader("Pengemudi dan kendaraan")

        driver_age = st.number_input(
            "Usia pengemudi (tahun)",
            min_value=18,
            max_value=100,
            value=35,
            step=1,
        )

        vehicle_age = st.number_input(
            "Usia kendaraan (tahun)",
            min_value=0,
            max_value=100,
            value=5,
            step=1,
        )

        vehicle_power = st.number_input(
            "Kode tenaga kendaraan (VehPower)",
            min_value=1,
            max_value=20,
            value=6,
            step=1,
            help="Kode pada dataset, bukan tenaga dalam HP atau kW.",
        )

        bonus_malus = st.number_input(
            "Indeks Bonus–Malus",
            min_value=1,
            max_value=350,
            value=100,
            step=1,
            help=(
                "Di bawah 100 menunjukkan bonus; "
                "di atas 100 menunjukkan malus. "
                "Gunakan nilai sebelum periode prediksi."
            ),
        )

        vehicle_brand = st.selectbox(
            "Kode merek kendaraan",
            options["VehBrand"],
            help="Merek pada dataset disamarkan menjadi kode.",
        )

    with right:
        st.subheader("Wilayah dan bahan bakar")

        fuel = st.selectbox(
            "Bahan bakar",
            options["VehGas"],
        )

        area = st.selectbox(
            "Kode area",
            options["Area"],
            help="Kategori area pada dataset Prancis.",
        )

        region = st.selectbox(
            "Kode wilayah Prancis",
            options["Region"],
        )

        density = st.number_input(
            "Kepadatan penduduk (penduduk/km²)",
            min_value=0.0,
            max_value=100000.0,
            value=1000.0,
            step=100.0,
        )

    st.caption(
        "Batas angka pada form adalah batas demonstrasi, "
        "bukan jaminan bahwa semua kombinasi profil "
        "terwakili dalam data training."
    )

    submitted = st.form_submit_button(
        "Hitung Premi Risiko",
        type="primary",
    )

if submitted:
    profile = {
        "DrivAge": driver_age,
        "VehAge": vehicle_age,
        "VehPower": vehicle_power,
        "BonusMalus": bonus_malus,
        "VehBrand": vehicle_brand,
        "VehGas": fuel,
        "Area": area,
        "Region": region,
        "Density": density,
    }

    try:
        result = predict_premium(profile)
    except ValueError as error:
        st.error(str(error))
    else:
        st.divider()
        st.subheader("Hasil simulasi")

        st.caption(
            f"Pendekatan terpilih dari validation: "
            f"{result['selected_approach']}"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Frekuensi (klaim/tahun)",
            f"{result['frequency']:.4f}",
        )

        col2.metric(
            "Severity (EUR/klaim)",
            f"{result['severity']:,.2f}",
        )

        col3.metric(
            "Premi risiko terpilih (EUR/tahun)",
            f"{result['selected_premium']:,.2f}",
        )

        comparison = pd.DataFrame({
            "Pendekatan": [
                "Frequency × Severity",
                "Direct Tweedie",
            ],
            "Premi risiko tahunan (EUR)": [
                result["frequency_severity_premium"],
                result["tweedie_premium"],
            ],
        })

        st.subheader("Perbandingan pendekatan")
        st.dataframe(
            comparison,
            hide_index=True,
        )

        st.caption(
            "Frekuensi merupakan ekspektasi jumlah klaim, "
            "bukan langsung probabilitas mengalami klaim. "
            "Model Tweedie memprediksi premi risiko secara langsung."
        )

        with st.expander("Lihat profil yang dihitung"):
            st.json(profile)