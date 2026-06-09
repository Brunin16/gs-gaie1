import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import add_features

warnings.filterwarnings("ignore")

ROOT         = Path(__file__).parent.parent
DATA_PATH    = ROOT / "data" / "fire_risk_dataset.csv"
MODEL_PATH   = ROOT / "models" / "best_model.joblib"
ARTIFACTS    = ROOT / "artifacts"
RANDOM_STATE = 42
TARGET       = "ocorrencia_foco"


def carregar(path=DATA_PATH):
    return add_features(pd.read_csv(path))


def split_features(df):
    X        = df.drop(columns=[TARGET])
    y        = df[TARGET]
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    return X, y, num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols):
    return ColumnTransformer([
        ("num", StandardScaler(),                      num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


def main():
    df = carregar()
    X, y, num_cols, cat_cols = split_features(df)
    print(f"Amostras: {len(df)} | Numéricas: {len(num_cols)} | Categóricas: {len(cat_cols)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pre = build_preprocessor(num_cols, cat_cols)
    modelos = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Random Forest":       RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=4,
            n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting":   GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            random_state=RANDOM_STATE),
    }

    cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    resultados = {}
    pipelines  = {}

    for nome, clf in modelos.items():
        pipe   = Pipeline([("pre", pre), ("clf", clf)])
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        resultados[nome] = {
            "cv_auc_mean":    float(cv_auc.mean()),
            "cv_auc_std":     float(cv_auc.std()),
            "test_accuracy":  float(accuracy_score(y_test, y_pred)),
            "test_precision": float(precision_score(y_test, y_pred)),
            "test_recall":    float(recall_score(y_test, y_pred)),
            "test_f1":        float(f1_score(y_test, y_pred)),
            "test_roc_auc":   float(roc_auc_score(y_test, y_proba)),
        }
        pipelines[nome] = pipe
        print(f"\n=== {nome} ===")
        print(f"CV ROC-AUC: {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})")
        print(classification_report(y_test, y_pred, digits=3))

    melhor_nome = max(resultados, key=lambda k: resultados[k]["test_roc_auc"])
    melhor_pipe = pipelines[melhor_nome]
    print(f"\n>>> MELHOR MODELO: {melhor_nome} "
          f"(ROC-AUC={resultados[melhor_nome]['test_roc_auc']:.4f})")

    _plot_comparacao(resultados)
    _plot_confusion(melhor_pipe, X_test, y_test, melhor_nome)
    _plot_roc(pipelines, X_test, y_test)
    _shap_analysis(melhor_pipe, X_train, X_test, num_cols, cat_cols, melhor_nome)

    joblib.dump(melhor_pipe, MODEL_PATH)
    with open(ARTIFACTS / "metrics.json", "w") as f:
        json.dump({
            "best_model":          melhor_nome,
            "results":             resultados,
            "feature_columns":     list(X.columns),
            "numeric_columns":     num_cols,
            "categorical_columns": cat_cols,
        }, f, indent=2)
    print("\nArtefatos salvos.")


def _plot_comparacao(resultados):
    metr   = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    nomes  = list(resultados.keys())
    x, w   = np.arange(len(metr)), 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, nome in enumerate(nomes):
        ax.bar(x + i * w, [resultados[nome][m] for m in metr], w, label=nome)
    ax.set_xticks(x + w)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Comparação de modelos — conjunto de teste")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "model_comparison.png", dpi=130)
    plt.close(fig)


def _plot_confusion(pipe, X_test, y_test, nome):
    disp = ConfusionMatrixDisplay(
        confusion_matrix(y_test, pipe.predict(X_test)),
        display_labels=["Sem foco", "Foco"],
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Oranges", colorbar=False)
    ax.set_title(f"Matriz de confusão — {nome}")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "confusion_matrix.png", dpi=130)
    plt.close(fig)


def _plot_roc(pipelines, X_test, y_test):
    fig, ax = plt.subplots(figsize=(6, 5))
    for nome, pipe in pipelines.items():
        proba       = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{nome} (AUC={roc_auc_score(y_test, proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("Falso positivo")
    ax.set_ylabel("Verdadeiro positivo")
    ax.set_title("Curvas ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "roc_curves.png", dpi=130)
    plt.close(fig)


def _shap_analysis(pipe, X_train, X_test, num_cols, cat_cols, nome):
    pre        = pipe.named_steps["pre"]
    clf        = pipe.named_steps["clf"]
    feat_names = pre.get_feature_names_out()

    X_test_t = pre.transform(X_test)
    if hasattr(X_test_t, "toarray"):
        X_test_t = X_test_t.toarray()

    # 500 amostras são suficientes para SHAP ser estável sem custo excessivo
    sample = X_test_t[:500]

    if nome in ("Random Forest", "Gradient Boosting"):
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(sample)
        if isinstance(sv, list):
            sv = sv[1]       # RF devolve lista [classe_0, classe_1]
        elif sv.ndim == 3:
            sv = sv[:, :, 1]
    else:
        bg = shap.sample(pre.transform(X_train), 100, random_state=RANDOM_STATE)
        if hasattr(bg, "toarray"):
            bg = bg.toarray()
        explainer = shap.LinearExplainer(clf, bg)
        sv = explainer.shap_values(sample)

    plt.figure()
    shap.summary_plot(sv, sample, feature_names=feat_names, show=False, max_display=12)
    plt.title(f"SHAP — impacto das variáveis ({nome})")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(sv, sample, feature_names=feat_names, plot_type="bar",
                      show=False, max_display=12)
    plt.title(f"SHAP — importância média ({nome})")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "shap_bar.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("SHAP gerado.")


if __name__ == "__main__":
    main()
