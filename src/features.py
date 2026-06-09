import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["severidade_seca"]  = df["dias_sem_chuva"] * (1 - df["umidade_relativa"] / 100)
    df["estresse_termico"] = df["temperatura_c"]  * (1 - df["umidade_relativa"] / 100)
    df["estacao_seca"]     = df["mes"].isin([5, 6, 7, 8, 9]).astype(int)
    df["combustivel_seco"] = (1 - df["ndvi"]) * df["dias_sem_chuva"]
    return df
