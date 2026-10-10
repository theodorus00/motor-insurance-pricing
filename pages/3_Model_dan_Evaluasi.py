import plotly.express as px
import streamlit as st

from src.data_loader import load_model_reports


st.set_page_config(
    page_title="Model dan Evaluasi | Motor Insurance Pricing",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Model dan Evaluasi")

metadata, reports = load_model_reports()

approach_labels = {
    "Baseline": "Baseline konstan",
    "FrequencySeverity": "Frequency × Severity",
    "DirectTweedie": "Direct Tweedie",
}

winner = metadata["winner"]
winner_label = approach_labels.get(winner, winner)

st.success(
    f"Pendekatan terpilih: {winner_label}. "
    "Pemilihan dilakukan berdasarkan deviance pada validation, "
    "sebelum mengevaluasi data test."
)

tab1, tab2, tab3 = st.tabs([
    "Metodologi",
    "Performa Model",
    "Kalibrasi dan Interpretasi",
])

with tab1:
    st.subheader("Dua pendekatan estimasi premi risiko")

    st.markdown(
        """
        **1. Frequency × Severity**

        Model frekuensi mengestimasi jumlah klaim per tahun.
        Model severity mengestimasi rata-rata biaya per klaim.
        Hasil keduanya dikalikan untuk memperoleh premi risiko tahunan.
        """
    )

    st.latex(
        r"\widehat{\text{Premi Risiko}}"
        r"=\widehat{\text{Frekuensi}}"
        r"\times\widehat{\text{Severity}}"
    )

    st.markdown(
        """
        Kandidat frekuensi mencakup **Poisson GLM** dan
        **Histogram Gradient Boosting dengan loss Poisson**.
        Kandidat severity mencakup **Gamma GLM** dan
        **Histogram Gradient Boosting dengan loss Gamma**.

        **2. Direct Tweedie**

        Tweedie GLM memprediksi biaya klaim per tahun exposure
        secara langsung. Target dapat bernilai nol atau positif.
        """
    )

    st.write(
        "Power Tweedie untuk pemodelan langsung dan evaluasi:",
        metadata["tweedie_evaluation_power"],
    )

    st.subheader("Model komponen yang terpilih")

    st.write({
        "Frekuensi": metadata["frequency"],
        "Severity": metadata["severity"],
        "Direct Tweedie": metadata["direct"],
    })

    st.subheader("Target dan pembobotan")

    st.markdown(
        """
        | Model | Target | Bobot training |
        |---|---|---|
        | Frekuensi | Jumlah klaim ÷ exposure | Exposure |
        | Severity | Total biaya ÷ jumlah klaim, pada polis dengan klaim | Jumlah klaim |
        | Direct Tweedie | Total biaya ÷ exposure | Exposure |
        """
    )

    st.subheader("Persiapan data dan evaluasi")

    st.markdown(
        """
        - Model menggunakan polis yang lolos rekonsiliasi antara
          data frekuensi dan severity.
        - Data dibagi menjadi **60% train, 20% validation,
          dan 20% test**, dengan stratifikasi ada/tidaknya klaim.
        - Fitur numerik diimputasi menggunakan median dan
          distandardisasi. Density ditransformasi dengan `log1p`.
        - Fitur kategori diimputasi dan diubah dengan one-hot encoding.
        - Saat seleksi model, preprocessing dipelajari dari train.
        - Setelah pemilihan, model di-fit ulang pada train + validation.
        - Data test digunakan untuk evaluasi akhir,
          bukan mengganti pemenang.
        """
    )

with tab2:
    st.subheader("Pemilihan pendekatan pada validation")

    validation = reports["validation"].copy()
    validation["Approach"] = validation["Approach"].replace(
        approach_labels
    )

    st.dataframe(
        validation[[
            "Approach",
            "TweedieDeviance",
            "MAE",
            "RMSE",
            "Actual_over_Predicted",
        ]],
        hide_index=True,
    )

    st.subheader("Hasil pada data test")

    test = reports["test"].copy()
    indexed = test.set_index("Approach")

    selected = indexed.loc[winner]
    baseline = indexed.loc["Baseline"]

    improvement = (
        100
        * (
            baseline["TweedieDeviance"]
            - selected["TweedieDeviance"]
        )
        / baseline["TweedieDeviance"]
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Tweedie deviance — model terpilih",
        f"{selected['TweedieDeviance']:.3f}",
    )

    col2.metric(
        "Penurunan deviance vs baseline",
        f"{improvement:.2f}%",
    )

    col3.metric(
        "Actual / Predicted — model terpilih",
        f"{selected['Actual_over_Predicted']:.3f}",
    )

    test["Approach"] = test["Approach"].replace(approach_labels)

    fig = px.bar(
        test,
        x="Approach",
        y="TweedieDeviance",
        text_auto=".2f",
        labels={
            "Approach": "Pendekatan",
            "TweedieDeviance": "Tweedie deviance",
        },
        title="Perbandingan deviance pada data test",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(test, hide_index=True)

    st.caption(
        "Deviance lebih rendah berarti lebih baik pada metrik ini. "
        "Seluruh pendekatan dibandingkan dengan power yang sama "
        "dan bobot exposure. Penurunan deviance negatif berarti "
        "model lebih buruk daripada baseline. "
        "MAE dan RMSE dihitung pada biaya tahunan dengan bobot exposure."
    )

    ratio = float(selected["Actual_over_Predicted"])

    if ratio > 1:
        st.write(
            f"Total biaya klaim aktual pada test adalah "
            f"**{ratio:.3f} kali** prediksi model terpilih. "
            "Model memprediksi biaya agregat lebih rendah "
            "daripada aktual pada sampel ini."
        )
    elif ratio < 1:
        st.write(
            f"Total biaya klaim aktual pada test adalah "
            f"**{ratio:.3f} kali** prediksi model terpilih. "
            "Model memprediksi biaya agregat lebih tinggi "
            "daripada aktual pada sampel ini."
        )
    else:
        st.write(
            "Total biaya klaim aktual dan prediksi sama "
            "pada sampel test ini."
        )

    st.caption(
        "Rasio agregat mendekati 1 belum menjamin prediksi "
        "akurat untuk setiap polis atau segmen."
    )

with tab3:
    st.subheader("Kalibrasi per kelompok risiko")

    calibration = reports["calibration"].copy()

    fig = px.bar(
        calibration,
        x="RiskBand",
        y=["ActualAnnual", "PredictedAnnual"],
        barmode="group",
        labels={
            "RiskBand": "Kelompok berdasarkan prediksi risiko",
            "value": "EUR per tahun exposure",
            "variable": "Jenis nilai",
        },
        title="Biaya aktual versus prediksi pada data test",
    )

    fig.for_each_trace(
        lambda trace: trace.update(
            name={
                "ActualAnnual": "Aktual",
                "PredictedAnnual": "Prediksi",
            }.get(trace.name, trace.name)
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Kelompok dibentuk berdasarkan prediksi model terpilih. "
        "Nilai aktual dan prediksi setiap kelompok dinormalisasi "
        "dengan total exposure."
    )

    st.subheader("Permutation importance")

    if winner == "Baseline":
        st.info(
            "Baseline menghasilkan prediksi konstan "
            "sehingga tidak memiliki importance fitur."
        )
    else:
        importance = reports["importance"].sort_values(
            "MeanIncreaseDeviance"
        )

        fig = px.bar(
            importance,
            x="MeanIncreaseDeviance",
            y="Feature",
            orientation="h",
            error_x="Std",
            labels={
                "MeanIncreaseDeviance": (
                    "Rata-rata kenaikan deviance setelah pengacakan"
                ),
                "Feature": "Fitur",
            },
            title="Pengaruh pengacakan fitur terhadap performa",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Dihitung menggunakan model train-only pada sampel "
            "validation, dengan dua pengacakan per fitur. "
            "Garis menunjukkan simpangan baku dua pengacakan, "
            "bukan interval kepercayaan. Nilai lebih besar menunjukkan "
            "performa lebih terganggu saat fitur diacak. "
            "Nilai negatif dapat muncul karena variasi sampel. "
            "Importance bukan bukti hubungan sebab-akibat."
        )

    st.subheader("Batas penggunaan")

    st.markdown(
        """
        - Hasil berlaku pada cohort yang lolos rekonsiliasi;
          pengecualian polis bermasalah dapat menimbulkan selection bias.
        - Random split belum menguji perubahan risiko antarperiode.
        - Klaim besar dapat sangat memengaruhi hasil evaluasi.
        - Data historis Prancis tidak langsung mewakili risiko
          kendaraan di Indonesia.
        - Estimasi belum mencakup expense, komisi, inflasi,
          reasuransi, biaya modal, dan margin keuntungan.
        """
    )