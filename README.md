# 🛰️ OrbitalFire — Predição de Risco de Queimada via Dados Orbitais

**Disciplina:** Generative AI For Engineering (GAIE)
**Global Solution 2026 · 1º Semestre · Indústria Espacial**
**FIAP · Engenharia de Software · 4º Ano**
**ODS conectado:** ODS 13 — Ação Climática (com interface para ODS 9 e 11)

> Integrantes:
> - Bruno Eduardo Caputo Paulino — RM 558303

---

## 1. Contexto do problema

Queimadas e incêndios florestais causam perdas ambientais, econômicas e humanas
em escala continental — especialmente no Brasil (Amazônia, Cerrado, Pantanal).
Dados de **sensoriamento remoto** (focos de calor da NASA FIRMS, NDVI do
Copernicus/Sentinel, índices meteorológicos do INPE) permitem **antecipar o risco**
de ocorrência de focos antes que se tornem incêndios incontroláveis.

O **OrbitalFire** é o módulo de IA da solução integrada: dado um conjunto de
variáveis ambientais e orbitais de uma região (temperatura, umidade, vento,
precipitação, dias sem chuva, NDVI, índice FWI, cobertura do solo, etc.), o modelo
estima a **probabilidade de ocorrência de foco de queimada** e classifica a área
em níveis de risco (Baixo / Moderado / Alto / Crítico), permitindo alerta precoce
e priorização de recursos de combate.

**Quem sofre / por que importa:** órgãos de defesa civil, brigadistas, produtores
rurais e populações próximas a áreas de risco. O monitoramento orbital cobre
regiões remotas sem infraestrutura terrestre de sensoriamento.

## 2. Fonte dos dados

Para garantir reprodutibilidade e volume controlado, o dataset é **sintético**,
gerado por `src/generate_dataset.py` simulando o comportamento dos dados públicos
de FIRMS/INPE/Copernicus. O gerador codifica relações físicas realistas
(estação seca, NDVI por cobertura, índice FWI) somadas a ruído estocástico.

- **5.000 linhas × 13 colunas** (atende ao mínimo de 1.000 linhas e 10 colunas)
- Alvo: `ocorrencia_foco` (0 = sem foco, 1 = foco) — ~40% positivos
- O mesmo dataset alimenta a disciplina **BDDI** (pipeline Airflow → Oracle)

| Variável | Descrição |
|---|---|
| `mes` | Mês da observação (1–12) |
| `temperatura_c` | Temperatura do ar (°C) |
| `umidade_relativa` | Umidade relativa (%) |
| `velocidade_vento_kmh` | Velocidade do vento (km/h) |
| `precipitacao_mm` | Precipitação nas últimas 24h (mm) |
| `dias_sem_chuva` | Dias consecutivos sem chuva |
| `ndvi` | Índice de vegetação por diferença normalizada (0–1) |
| `indice_fwi` | Proxy do Fire Weather Index |
| `latitude` / `longitude` | Coordenadas (território brasileiro) |
| `altitude_m` | Altitude (m) |
| `tipo_cobertura` | floresta / cerrado / pastagem / agricultura / urbano |
| `ocorrencia_foco` | **alvo** (0/1) |

## 3. Metodologia

Pipeline completo em `src/train_pipeline.py`:

1. **Carregamento + engenharia de atributos** — 4 features derivadas:
   `severidade_seca`, `estresse_termico`, `estacao_seca`, `combustivel_seco`.
2. **Pré-processamento** via `ColumnTransformer`: `StandardScaler` nas numéricas
   e `OneHotEncoder` na categórica (`tipo_cobertura`).
3. **Treinamento** de 3 modelos dentro de `Pipeline` do scikit-learn.
4. **Validação** com `StratifiedKFold` (5 folds) + avaliação em holdout de 20%.
5. **Comparação** por Acurácia, Precisão, Recall, F1 e ROC-AUC.
6. **Escolha do melhor modelo** (maior ROC-AUC no teste) e persistência.
7. **Interpretabilidade** com SHAP.

## 4. Modelos testados

Técnica de classificação supervisionada (3 algoritmos, atende ao mínimo de 2):

- **Logistic Regression** (modelo linear de referência)
- **Random Forest** (ensemble de árvores — bagging)
- **Gradient Boosting** (ensemble sequencial — boosting)

## 5. Resultados obtidos

| Modelo | Acurácia | Precisão | Recall | F1 | ROC-AUC | CV ROC-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** ⭐ | **0.824** | 0.797 | 0.744 | **0.770** | **0.897** | 0.904 |
| Random Forest | 0.823 | 0.804 | 0.729 | 0.765 | 0.885 | 0.895 |
| Gradient Boosting | 0.815 | 0.782 | 0.737 | 0.759 | 0.890 | 0.895 |

**Melhor modelo: Logistic Regression** (ROC-AUC = 0.897 no teste).
Os três modelos ficam muito próximos — coerente, já que o processo gerador dos
dados é fundamentalmente logístico, favorecendo o modelo linear. Os gráficos estão
em `artifacts/`: `model_comparison.png`, `confusion_matrix.png`, `roc_curves.png`.

## 6. Interpretação com SHAP

Análise em `artifacts/shap_summary.png` (beeswarm) e `artifacts/shap_bar.png`
(importância média). As variáveis de maior impacto na previsão são, tipicamente:

- **`dias_sem_chuva`** e **`severidade_seca`** — quanto maior, maior o risco;
- **`umidade_relativa`** — relação inversa (mais úmido, menor risco);
- **`ndvi`** e **`combustivel_seco`** — vegetação seca aumenta o combustível;
- **`indice_fwi`** e **`temperatura_c`** — reforçam o risco;
- **`tipo_cobertura`** (cerrado/pastagem) — maior propensão que floresta/urbano.

Isso confirma que o modelo aprendeu relações **fisicamente plausíveis**, e não
correlações espúrias — requisito central da interpretabilidade.

## 7. Como executar

```bash
# 1. Ambiente
python -m venv venv && source venv/bin/activate   # (Windows: venv\Scripts\activate)
pip install -r requirements.txt

# 2. Gerar o dataset
python src/generate_dataset.py

# 3. Treinar, comparar, gerar SHAP e salvar o melhor modelo
python src/train_pipeline.py

# 4. Subir a aplicação (deploy)
streamlit run src/app.py
```

## 8. Deploy

A aplicação interativa (`src/app.py`, **Streamlit**) carrega o melhor pipeline
treinado e estima o risco de queimada em tempo real a partir dos parâmetros
informados, exibindo probabilidade, nível de risco e a comparação dos modelos.

**Sugestão de deploy público (gratuito):** [Streamlit Community Cloud](https://streamlit.io/cloud)
— conectar o repositório GitHub e apontar para `src/app.py`.

🔗 **Link da aplicação em funcionamento:** _preencher após o deploy_

## 9. Estrutura do repositório

```
orbitalfire-gaie/
├── data/
│   └── fire_risk_dataset.csv      # dataset sintético (gerado)
├── models/
│   └── best_model.joblib          # melhor pipeline serializado
├── artifacts/
│   ├── metrics.json               # métricas de todos os modelos
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── shap_summary.png
│   └── shap_bar.png
├── src/
│   ├── generate_dataset.py        # geração dos dados
│   ├── train_pipeline.py          # pipeline de ML + SHAP
│   └── app.py                     # deploy Streamlit
├── requirements.txt
└── README.md
```
