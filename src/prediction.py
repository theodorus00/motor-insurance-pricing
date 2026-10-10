from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd


# Lokasi dihitung dari file ini, bukan lokasi terminal.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "motor_pricing.joblib"

NUMERIC_COLUMNS = [
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "Density",
]

CATEGORY_COLUMNS = [
    "Area",
    "VehBrand",
    "VehGas",
    "Region",
]


@lru_cache(maxsize=1)
def load_model():
    """Muat model sekali selama proses aplikasi berjalan."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def prepare_features(profile, bundle):
    """Ubah satu profil pengguna menjadi input model."""
    required = NUMERIC_COLUMNS + CATEGORY_COLUMNS
    missing = [column for column in required if column not in profile]

    if missing:
        raise ValueError(f"Input belum lengkap: {', '.join(missing)}")

    data = pd.DataFrame([{column: profile[column] for column in required}])

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="raise")

        if not np.isfinite(data[column].iloc[0]):
            raise ValueError(f"{column} harus berupa angka yang valid.")

        if data[column].iloc[0] < 0:
            raise ValueError(f"{column} tidak boleh negatif.")

    if data["DrivAge"].iloc[0] < 18:
        raise ValueError("Usia pengemudi minimal 18 tahun.")

    if (data[["VehPower", "BonusMalus"]].iloc[0] <= 0).any():
        raise ValueError("VehPower dan BonusMalus harus lebih besar dari nol.")

    for column in CATEGORY_COLUMNS:
        value = data[column].iloc[0]

        if pd.isna(value):
            raise ValueError(f"{column} wajib diisi.")

        data[column] = str(value).strip().strip("'")

        if data[column].iloc[0] in ("", "?"):
            raise ValueError(f"{column} wajib diisi.")

    data["LogDensity"] = np.log1p(data["Density"])

    return data[bundle["features"]]


def predict_premium(profile):
    """Prediksi frekuensi, severity, dan premi risiko tahunan dalam EUR."""
    bundle = load_model()
    features = prepare_features(profile, bundle)
    models = bundle["models"]

    frequency = float(
        models[bundle["frequency_name"]].predict(features)[0]
    )
    severity = float(
        models[bundle["severity_name"]].predict(features)[0]
    )
    tweedie_premium = float(
        models[bundle["direct_name"]].predict(features)[0]
    )

    estimates = [frequency, severity, tweedie_premium]

    if not all(np.isfinite(value) and value > 0 for value in estimates):
        raise ValueError("Model menghasilkan prediksi tidak valid.")

    frequency_severity_premium = frequency * severity

    premiums = {
        "Baseline": float(bundle["baseline_rate"]),
        "FrequencySeverity": frequency_severity_premium,
        "DirectTweedie": tweedie_premium,
    }

    winner = bundle["winner"]

    if winner not in premiums:
        raise ValueError(f"Pendekatan model tidak dikenali: {winner}")

    return {
        "frequency": frequency,
        "severity": severity,
        "frequency_severity_premium": frequency_severity_premium,
        "tweedie_premium": tweedie_premium,
        "selected_premium": premiums[winner],
        "selected_approach": winner,
        "currency": bundle.get("currency", "EUR"),
    }
    
def get_category_options():
    bundle = load_model()

    model_names = [
        bundle["frequency_name"],
        bundle["severity_name"],
        bundle["direct_name"],
    ]

    options = {}

    for name in model_names:
        pipeline = bundle["models"][name]
        preprocess = pipeline.named_steps["preprocess"]
        encoder = (
            preprocess.named_transformers_["cat"]
            .named_steps["encode"]
        )

        for column, categories in zip(
            CATEGORY_COLUMNS, encoder.categories_
        ):
            values = {str(value) for value in categories}

            if column not in options:
                options[column] = values
            else:
                options[column] &= values

    if any(not values for values in options.values()):
        raise ValueError("Tidak ada kategori bersama antar-model.")

    return {
        column: sorted(values)
        for column, values in options.items()
    }