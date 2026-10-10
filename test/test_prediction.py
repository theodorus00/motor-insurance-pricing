from pathlib import Path

import pandas as pd

from src.prediction import predict_premium


PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main():
    sample = pd.read_csv(
        PROJECT_ROOT / "data" / "raw" / "freMTPL2freq.csv",
        nrows=1,
    )

    profile = sample.iloc[0].to_dict()
    result = predict_premium(profile)

    print("Pendekatan terpilih:", result["selected_approach"])
    print(f"Frekuensi: {result['frequency']:.4f} klaim/tahun")
    print(f"Severity: EUR {result['severity']:,.2f}/klaim")
    print(
        "Premi Frequency × Severity: "
        f"EUR {result['frequency_severity_premium']:,.2f}/tahun"
    )
    print(
        f"Premi Tweedie: EUR {result['tweedie_premium']:,.2f}/tahun"
    )
    print(
        f"Premi terpilih: EUR {result['selected_premium']:,.2f}/tahun"
    )


if __name__ == "__main__":
    main()