import json
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@st.cache_data
def load_eda_data():
    summary = json.loads(
        (PROCESSED_DIR / "eda_summary.json").read_text(
            encoding="utf-8"
        )
    )

    claims = pd.read_csv(
        PROCESSED_DIR / "eda_claim_distribution.csv"
    )

    segments = pd.read_csv(
        PROCESSED_DIR / "eda_segments.csv"
    )

    histograms = pd.read_csv(
        PROCESSED_DIR / "eda_histograms.csv"
    )

    return summary, claims, segments, histograms

REPORTS_DIR = PROJECT_ROOT / "reports"


@st.cache_data
def load_model_reports():
    metadata = json.loads(
        (REPORTS_DIR / "experiment_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    tables = {
        name: pd.read_csv(REPORTS_DIR / filename)
        for name, filename in {
            "validation": "model_selection_validation.csv",
            "test": "test_metrics.csv",
            "calibration": "calibration_test.csv",
            "importance": "permutation_importance.csv",
        }.items()
    }

    return metadata, tables