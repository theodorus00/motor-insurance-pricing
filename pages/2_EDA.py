import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_eda_data


st.set_page_config(
    page_title="EDA | Motor Insurance Pricing",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Eksplorasi Data Asuransi Kendaraan")

st.write(
    "Kenali karakteristik polis, distribusi klaim, "
    "dan perbedaan pengalaman klaim antarsegmen."
)

summary, claims, segments, histograms = load_eda_data()

st.info(
    f"Populasi: {summary['population']}. "
    "Seluruh grafik menggunakan data train, "
    "bukan keseluruhan dataset atau data test. "
    "Nilai biaya ditampilkan dalam EUR."
)

# Ringkasan portofolio
col1, col2, col3 = st.columns(3)

col1.metric(
    "Jumlah polis",
    f"{summary['policies']:,.0f}",
)
col2.metric(
    "Total exposure (tahun)",
    f"{summary['exposure_years']:,.2f}",
)
col3.metric(
    "Jumlah klaim",
    f"{summary['claims']:,.0f}",
)

col4, col5, col6 = st.columns(3)

col4.metric(
    "Frekuensi (klaim/tahun)",
    f"{summary['frequency']:.4f}",
)
col5.metric(
    "Severity (EUR/klaim)",
    (
        f"{summary['severity_eur']:,.2f}"
        if summary["severity_eur"] is not None
        else "Tidak tersedia"
    ),
)
col6.metric(
    "Premi risiko historis (EUR/tahun)",
    f"{summary['pure_premium_eur']:,.2f}",
)

st.caption(
    "Frekuensi = total klaim ÷ total exposure. "
    "Severity = total biaya klaim ÷ total klaim. "
    "Premi risiko historis = total biaya klaim ÷ total exposure. "
    "Angka ini merupakan pengalaman aktual data train, "
    "bukan prediksi model."
)

tab1, tab2, tab3 = st.tabs([
    "Distribusi Klaim",
    "Karakteristik Polis",
    "Analisis Segmen",
])

with tab1:
    st.subheader("Berapa banyak polis yang memiliki klaim?")

    st.write(
        f"**{summary['zero_claim_pct']:.2f}%** polis "
        "pada data train tidak memiliki klaim."
    )

    log_y = st.checkbox(
        "Gunakan skala log pada jumlah polis",
        value=True,
    )

    claim_plot = claims.copy()
    claim_plot["Jumlah klaim"] = (
        claim_plot["ClaimNb"].astype(int).astype(str)
    )

    fig = px.bar(
        claim_plot,
        x="Jumlah klaim",
        y="Policies",
        log_y=log_y,
        labels={"Policies": "Jumlah polis"},
        title="Distribusi jumlah klaim per polis",
    )

    fig.update_xaxes(type="category")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Skala log membantu menampilkan kelompok dengan sedikit "
        "polis. Grafik ini menghitung polis berdasarkan jumlah "
        "klaim selama exposure masing-masing; durasinya bisa berbeda."
    )

with tab2:
    st.subheader("Distribusi karakteristik polis")

    variables = {
        "Usia pengemudi": (
            "DrivAge",
            "Usia pengemudi (tahun)",
        ),
        "Lama pertanggungan": (
            "Exposure",
            "Exposure (tahun)",
        ),
        "Severity rata-rata per polis — log10": (
            "Log10AvgSeverity",
            "Log10 rata-rata biaya per klaim dalam polis",
        ),
    }

    selected = st.selectbox(
        "Pilih karakteristik",
        list(variables),
    )

    variable, axis_label = variables[selected]

    histogram = histograms.loc[
        histograms["Variable"].eq(variable)
    ].copy()

    histogram["Center"] = (
        histogram["Left"] + histogram["Right"]
    ) / 2

    histogram["Width"] = (
        histogram["Right"] - histogram["Left"]
    )

    fig = go.Figure(go.Bar(
        x=histogram["Center"],
        y=histogram["Count"],
        width=histogram["Width"],
        customdata=histogram[["Left", "Right"]].to_numpy(),
        hovertemplate=(
            "Batas interval: %{customdata[0]:.2f}"
            " – %{customdata[1]:.2f}"
            "<br>Jumlah polis: %{y:,.0f}"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=selected,
        xaxis_title=axis_label,
        yaxis_title="Jumlah polis",
        bargap=0,
    )

    st.plotly_chart(fig, use_container_width=True)

    if variable == "Log10AvgSeverity":
        st.caption(
            "Hanya polis dengan klaim positif. "
            "Setiap polis menyumbang satu nilai rata-rata severity, "
            "sehingga grafik ini bukan distribusi individual klaim. "
            "Pada sumbu log10, 3 berarti EUR 1.000 dan "
            "4 berarti EUR 10.000."
        )

with tab3:
    st.subheader("Pengalaman klaim antarsegmen")

    dimensions = {
        "Kelompok usia pengemudi": "DriverAgeBand",
        "Area": "Area",
        "Bahan bakar": "VehGas",
        "Wilayah Prancis": "Region",
    }

    metrics = {
        "Frekuensi klaim": (
            "Frequency",
            "Klaim per tahun exposure",
        ),
        "Severity": (
            "Severity",
            "EUR per klaim",
        ),
        "Premi risiko historis": (
            "PurePremium",
            "EUR per tahun exposure",
        ),
    }

    left, right = st.columns(2)

    with left:
        dimension_label = st.selectbox(
            "Kelompokkan berdasarkan",
            list(dimensions),
        )

    with right:
        metric_label = st.selectbox(
            "Metrik yang ditampilkan",
            list(metrics),
        )

    dimension = dimensions[dimension_label]
    metric, unit = metrics[metric_label]

    table = segments.loc[
        segments["Dimension"].eq(dimension)
    ].copy()

    table["Segment"] = table["Segment"].astype(str)

    fig = px.bar(
        table,
        x="Segment",
        y=metric,
        hover_data={
            "Policies": ":,.0f",
            "Exposure": ":,.2f",
            "Claims": ":,.0f",
        },
        labels={
            "Segment": dimension_label,
            metric: unit,
            "Policies": "Jumlah polis",
            "Exposure": "Exposure (tahun)",
            "Claims": "Jumlah klaim",
        },
        title=f"{metric_label} menurut {dimension_label}",
    )

    fig.update_xaxes(type="category")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        table[[
            "Segment",
            "Policies",
            "Exposure",
            "Claims",
            "Frequency",
            "Severity",
            "PurePremium",
        ]].rename(columns={
            "Segment": "Segmen",
            "Policies": "Jumlah polis",
            "Exposure": "Exposure (tahun)",
            "Claims": "Jumlah klaim",
            "Frequency": "Frekuensi",
            "Severity": "Severity (EUR/klaim)",
            "PurePremium": "Premi risiko (EUR/tahun)",
        }),
        hide_index=True,
    )

    st.caption(
        "Perhatikan jumlah klaim dan exposure saat membandingkan "
        "segmen. Segmen kecil dapat memiliki rata-rata yang "
        "sangat dipengaruhi satu klaim besar. Severity tidak "
        "tersedia jika segmen tidak memiliki klaim. "
        "Perbedaan antarsegmen tidak membuktikan hubungan sebab-akibat."
    )