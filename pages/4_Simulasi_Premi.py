import math
import os
from urllib.parse import urlsplit

import requests
import pandas as pd
import streamlit as st

# Set MOTOR_PRICING_API_URL pada environment untuk memakai server lain.
API_URL = os.getenv("MOTOR_PRICING_API_URL", "http://127.0.0.1:8000").strip().rstrip("/")

# Kode kategori dari dataset freMTPL2. API tetap memvalidasi kategori
# terhadap encoder model. Halaman ini tidak memuat joblib lokal.
CATEGORY_OPTIONS = {
    "VehBrand": ["B1", "B10", "B11", "B12", "B13", "B14", "B2", "B3", "B4", "B5", "B6"],
    "VehGas": ["Diesel", "Regular"],
    "Area": ["A", "B", "C", "D", "E", "F"],
    "Region": ["R11", "R21", "R22", "R23", "R24", "R25", "R26", "R31", "R41", "R42", "R43", "R52", "R53", "R54", "R72", "R73", "R74", "R82", "R83", "R91", "R93", "R94"],
}


def predict_premium_api(profile):
    """Panggil API, validasi respons, lalu petakan field ke tampilan lama."""
    parsed = urlsplit(API_URL)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("Konfigurasi MOTOR_PRICING_API_URL tidak valid. Gunakan URL dasar server API.")

    try:
        response = requests.post(
            f"{API_URL}/predict", json=profile, timeout=(5, 30),
            allow_redirects=False,
        )
    except requests.exceptions.Timeout as error:
        raise ValueError("API terlalu lama merespons. Silakan coba lagi.") from error
    except requests.exceptions.ConnectionError as error:
        raise ValueError("Tidak dapat terhubung ke API. Pastikan server FastAPI aktif dan alamat API benar.") from error
    except requests.exceptions.RequestException as error:
        raise ValueError("Permintaan ke API gagal. Periksa koneksi dan konfigurasi server.") from error

    if response.status_code == 422:
        try:
            detail = response.json().get("detail", "Input tidak valid.")
            if isinstance(detail, list):
                detail = "; ".join(
                    f"{'.'.join(str(part) for part in item.get('loc', []) if part != 'body')}: {item.get('msg', 'Input tidak valid')}"
                    for item in detail if isinstance(item, dict)
                )
            if not isinstance(detail, str):
                detail = "Periksa profil kendaraan yang dimasukkan."
        except (ValueError, AttributeError):
            detail = "Periksa profil kendaraan yang dimasukkan."
        raise ValueError(f"Input ditolak API (422): {detail}")
    if response.status_code == 503:
        raise ValueError("Model pada API belum siap. Silakan coba lagi nanti.")
    if response.status_code != 200:
        raise ValueError(f"API mengembalikan HTTP {response.status_code}. Periksa log server FastAPI.")

    try:
        data = response.json()
    except ValueError as error:
        raise ValueError("Respons API bukan JSON yang valid.") from error
    if not isinstance(data, dict):
        raise ValueError("Format respons API tidak sesuai.")
    mapping = {
        "frequency": "expected_claims_per_year",
        "severity": "expected_cost_per_claim",
        "selected_premium": "selected_annual_risk_premium",
        "frequency_severity_premium": "frequency_severity_premium",
        "tweedie_premium": "tweedie_premium",
    }
    result = {}
    for target, source in mapping.items():
        value = data.get(source)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Respons API tidak valid pada field {source}.")
        result[target] = float(value)
    if data.get("currency") != "EUR":
        raise ValueError("Mata uang respons API tidak sesuai dengan tampilan EUR.")
    if data.get("selected_model") not in {"Baseline", "FrequencySeverity", "DirectTweedie"}:
        raise ValueError("Pendekatan model pada respons API tidak dikenal.")
    if not isinstance(data.get("population"), str) or not data["population"].strip():
        raise ValueError("Informasi populasi model tidak tersedia pada respons API.")
    result["selected_approach"] = data["selected_model"]
    result["population"] = data["population"]
    result["api_response"] = data
    return result


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

options = CATEGORY_OPTIONS

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
        with st.spinner("Menghitung premi..."):
            result = predict_premium_api(profile)
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

        st.caption(f"Populasi evaluasi model: {result['population']}")
        with st.expander("Lihat respons API"):
            st.json(result["api_response"])
