import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_STATE = 42
N_ROWS = 5000

# Propensões relativas a queimada por cobertura do solo, calibradas para o Brasil
COBERTURAS = {
    "floresta":    0.6,
    "cerrado":     1.4,
    "pastagem":    1.5,
    "agricultura": 1.1,
    "urbano":      0.2,
}


def gerar(n=N_ROWS, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)

    mes = rng.integers(1, 13, size=n)
    # Estação seca no Brasil central: maio–setembro
    fator_seca = np.where(np.isin(mes, [5, 6, 7, 8, 9]), 1.0, 0.0)

    temperatura_c         = rng.normal(28 + 5 * fator_seca, 4, n).clip(10, 48)
    umidade_relativa      = rng.normal(60 - 22 * fator_seca, 12, n).clip(8, 100)
    velocidade_vento_kmh  = rng.gamma(2.0, 6.0, n).clip(0, 60)
    precipitacao_mm       = (rng.gamma(1.2, 6.0, n) * (1 - 0.7 * fator_seca)).clip(0, 120)
    dias_sem_chuva        = rng.poisson(4 + 9 * fator_seca, n).clip(0, 60)

    cobertura = rng.choice(list(COBERTURAS.keys()), size=n, p=[0.22, 0.26, 0.24, 0.20, 0.08])
    ndvi_base = {"floresta": 0.78, "cerrado": 0.55, "pastagem": 0.45, "agricultura": 0.50, "urbano": 0.20}
    ndvi      = np.array([rng.normal(ndvi_base[c], 0.08) for c in cobertura]).clip(0.05, 0.95)

    latitude   = rng.uniform(-33, 5, n)
    longitude  = rng.uniform(-74, -34, n)
    altitude_m = rng.gamma(2.0, 250, n).clip(0, 2800)

    # FWI proxy: sobe com calor, vento e dias secos; cai com umidade e precipitação
    indice_fwi = (
        0.45 * (temperatura_c - 15)
        + 0.35 * (100 - umidade_relativa) / 5
        + 0.30 * velocidade_vento_kmh
        + 0.50 * dias_sem_chuva
        - 0.20 * precipitacao_mm
    ).clip(0, None)

    prop_cob = np.array([COBERTURAS[c] for c in cobertura])
    z = (
        0.060 * (temperatura_c - 28)
        - 0.045 * (umidade_relativa - 50)
        + 0.030 * velocidade_vento_kmh
        - 0.040 * precipitacao_mm
        + 0.070 * dias_sem_chuva
        + 0.018 * indice_fwi
        - 1.20  * (ndvi - 0.5)
        + 0.90  * (prop_cob - 1.0)
        - 1.6
    )
    # Ruído para manter acurácia realista (~80%) em vez de separabilidade perfeita
    z    = 1.7 * z + rng.normal(0, 0.5, n)
    prob = 1 / (1 + np.exp(-z))
    ocorrencia_foco = (rng.uniform(0, 1, n) < prob).astype(int)

    return pd.DataFrame({
        "mes":                  mes,
        "temperatura_c":        temperatura_c.round(1),
        "umidade_relativa":     umidade_relativa.round(1),
        "velocidade_vento_kmh": velocidade_vento_kmh.round(1),
        "precipitacao_mm":      precipitacao_mm.round(1),
        "dias_sem_chuva":       dias_sem_chuva,
        "ndvi":                 ndvi.round(3),
        "indice_fwi":           indice_fwi.round(1),
        "latitude":             latitude.round(4),
        "longitude":            longitude.round(4),
        "altitude_m":           altitude_m.round(0),
        "tipo_cobertura":       cobertura,
        "ocorrencia_foco":      ocorrencia_foco,
    })


if __name__ == "__main__":
    out = Path(__file__).parent.parent / "data" / "fire_risk_dataset.csv"
    df  = gerar()
    df.to_csv(out, index=False)
    print(f"Dataset gerado: {df.shape[0]} linhas x {df.shape[1]} colunas → {out}")
    print(f"Taxa de focos (target=1): {df['ocorrencia_foco'].mean():.1%}")
    print(df.head())
