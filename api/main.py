from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import logging

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)

# Lokasi model dihitung dari lokasi file, bukan lokasi terminal.
PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_DIR
    / "outputs_motor_pricing"
    / "motor_pricing.joblib"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)

    required_keys = {
        "winner",
        "baseline_rate",
        "frequency_name",
        "severity_name",
        "direct_name",
        "models",
        "features",
        "currency",
        "population",
    }

    if not isinstance(bundle, dict):
        raise ValueError("Artefak model harus berupa dictionary.")

    missing = required_keys - bundle.keys()
    if missing:
        raise ValueError(f"Komponen model belum lengkap: {missing}")

    if bundle["winner"] not in {
        "Baseline",
        "FrequencySeverity",
        "DirectTweedie",
    }:
        raise ValueError(f"Winner tidak dikenal: {bundle['winner']}")

    for key in ["frequency_name", "severity_name", "direct_name"]:
        if bundle[key] not in bundle["models"]:
            raise ValueError(f"Estimator tidak ditemukan: {bundle[key]}")

    app.state.bundle = bundle

    yield

    app.state.bundle = None


app = FastAPI(
    title="Motor Insurance Pricing API",
    description="Estimasi premi risiko tahunan asuransi kendaraan.",
    version="1.0.0",
    lifespan=lifespan,
)


class VehicleInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )

    VehPower: int = Field(gt=0)
    VehAge: int = Field(ge=0)
    DrivAge: int = Field(ge=18)
    BonusMalus: int = Field(gt=0)
    Density: float = Field(ge=0)

    Area: Literal["A", "B", "C", "D", "E", "F"]
    VehBrand: str = Field(min_length=1)
    VehGas: Literal["Diesel", "Regular"]
    Region: str = Field(min_length=1)


class PredictionOutput(BaseModel):
    selected_model: str
    currency: str
    population: str
    expected_claims_per_year: float
    expected_cost_per_claim: float
    frequency_severity_premium: float
    tweedie_premium: float
    selected_annual_risk_premium: float


def get_bundle():
    bundle = getattr(app.state, "bundle", None)

    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model belum siap.",
        )

    return bundle


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    bundle = get_bundle()

    return {
        "status": "ready",
        "selected_model": bundle["winner"],
        "currency": bundle["currency"],
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: VehicleInput):
    bundle = get_bundle()
    raw = pd.DataFrame([payload.model_dump()])

    try:
        # Feature engineering yang sama dengan notebook.
        raw["LogDensity"] = np.log1p(raw["Density"])

        # Urutan kolom mengikuti artefak model.
        x = raw.loc[:, bundle["features"]]

        models = bundle["models"]

        # Validasi kategori terhadap encoder yang dilatih.
        # Struktur pipeline mengikuti notebook kita.
        for model_name in {
            bundle["frequency_name"],
            bundle["severity_name"],
            bundle["direct_name"],
        }:
            preprocessor = models[model_name].named_steps["preprocess"]
            cat_pipeline = preprocessor.named_transformers_["cat"]
            encoder = cat_pipeline.named_steps["encode"]

            cat_columns = next(
                columns
                for name, transformer, columns
                in preprocessor.transformers_
                if name == "cat"
            )

            for column, categories in zip(
                cat_columns,
                encoder.categories_,
            ):
                if raw.at[0, column] not in categories:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Kategori {column} tidak dikenal model.",
                    )

        frequency = float(
            models[bundle["frequency_name"]].predict(x)[0]
        )

        severity = float(
            models[bundle["severity_name"]].predict(x)[0]
        )

        tweedie = float(
            models[bundle["direct_name"]].predict(x)[0]
        )

        frequency_severity = frequency * severity

        premiums = {
            "Baseline": float(bundle["baseline_rate"]),
            "FrequencySeverity": frequency_severity,
            "DirectTweedie": tweedie,
        }

        selected = premiums[bundle["winner"]]

        values = [
            frequency,
            severity,
            tweedie,
            frequency_severity,
            selected,
        ]

        if not all(np.isfinite(v) and v >= 0 for v in values):
            raise ValueError("Model menghasilkan nilai tidak valid.")

        return PredictionOutput(
            selected_model=bundle["winner"],
            currency=bundle["currency"],
            population=bundle["population"],
            expected_claims_per_year=frequency,
            expected_cost_per_claim=severity,
            frequency_severity_premium=frequency_severity,
            tweedie_premium=tweedie,
            selected_annual_risk_premium=selected,
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception("Prediksi premi gagal.")
        raise HTTPException(
            status_code=500,
            detail="Prediksi gagal. Periksa log server.",
        )