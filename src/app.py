import json
from pathlib import Path

import joblib
import pandas as pd
import pydeck as pdk
import streamlit as st

from features import add_features

ROOT         = Path(__file__).parent.parent
MODEL_PATH   = ROOT / "models" / "best_model.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"


@st.cache_resource
def carregar_modelo():
    pipe = joblib.load(MODEL_PATH)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return pipe, metrics


def nivel_risco(p: float) -> tuple[str, str]:
    if p < 0.25:
        return "BAIXO", "🟢"
    if p < 0.50:
        return "MODERADO", "🟡"
    if p < 0.75:
        return "ALTO", "🟠"
    return "CRÍTICO", "🔴"


def main():
    st.set_page_config(page_title="OrbitalFire", page_icon="🛰️", layout="wide")
    st.title("🛰️ OrbitalFire — Risco de Queimada via Dados Orbitais")
    st.caption("GAIE · Global Solution 2026 · Indústria Espacial · ODS 13")

    pipe, metrics = carregar_modelo()

    with st.sidebar:
        st.header("Parâmetros da observação")
        mes                  = st.slider("Mês", 1, 12, 8)
        temperatura_c        = st.slider("Temperatura (°C)", 10.0, 48.0, 34.0)
        umidade_relativa     = st.slider("Umidade relativa (%)", 8.0, 100.0, 30.0)
        velocidade_vento_kmh = st.slider("Vento (km/h)", 0.0, 60.0, 18.0)
        precipitacao_mm      = st.slider("Precipitação 24h (mm)", 0.0, 120.0, 2.0)
        dias_sem_chuva       = st.slider("Dias sem chuva", 0, 60, 18)
        ndvi                 = st.slider("NDVI (vegetação)", 0.05, 0.95, 0.45)
        indice_fwi           = st.slider("Índice FWI", 0.0, 120.0, 40.0)
        altitude_m           = st.slider("Altitude (m)", 0.0, 2800.0, 500.0)
        latitude             = st.number_input("Latitude", -33.0, 5.0, -15.0)
        longitude            = st.number_input("Longitude", -74.0, -34.0, -52.0)
        tipo_cobertura       = st.selectbox(
            "Cobertura do solo",
            ["floresta", "cerrado", "pastagem", "agricultura", "urbano"],
            index=1,
        )

    entrada = {
        "mes": mes, "temperatura_c": temperatura_c,
        "umidade_relativa": umidade_relativa,
        "velocidade_vento_kmh": velocidade_vento_kmh,
        "precipitacao_mm": precipitacao_mm,
        "dias_sem_chuva": dias_sem_chuva, "ndvi": ndvi,
        "indice_fwi": indice_fwi, "latitude": latitude,
        "longitude": longitude, "altitude_m": altitude_m,
        "tipo_cobertura": tipo_cobertura,
    }

    X    = add_features(pd.DataFrame([entrada]))
    prob = float(pipe.predict_proba(X)[:, 1][0])
    nivel, emoji = nivel_risco(prob)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Resultado")
        st.metric("Probabilidade de foco", f"{prob:.1%}")
        st.markdown(f"### {emoji} Risco **{nivel}**")
        st.progress(prob)
    with col2:
        st.subheader("Modelo em produção")
        best = metrics["best_model"]
        r    = metrics["results"][best]
        st.write(f"**{best}**")
        st.write(f"- Acurácia (teste): {r['test_accuracy']:.1%}")
        st.write(f"- ROC-AUC (teste):  {r['test_roc_auc']:.3f}")
        st.write(f"- F1 (teste):       {r['test_f1']:.3f}")

    st.divider()
    st.subheader("Localização da observação")

    _cor_risco = {
        "BAIXO":    [0, 200, 0, 210],
        "MODERADO": [255, 200, 0, 210],
        "ALTO":     [255, 120, 0, 210],
        "CRÍTICO":  [220, 0, 0, 210],
    }
    mapa_df = pd.DataFrame([{"lat": latitude, "lon": longitude}])
    camada = pdk.Layer(
        "ScatterplotLayer",
        data=mapa_df,
        get_position="[lon, lat]",
        get_color=_cor_risco[nivel],
        get_radius=80_000,
        pickable=True,
    )
    st.pydeck_chart(pdk.Deck(
        layers=[camada],
        initial_view_state=pdk.ViewState(
            latitude=latitude, longitude=longitude, zoom=4, pitch=0
        ),
        tooltip={"text": f"Lat: {latitude:.2f}, Lon: {longitude:.2f}\nRisco: {nivel} ({prob:.1%})"},
    ))

    st.divider()
    st.subheader("Comparação de modelos")
    df_cmp = pd.DataFrame(metrics["results"]).T[
        ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]
    ]
    st.dataframe(df_cmp.style.format("{:.3f}"))


if __name__ == "__main__":
    main()
