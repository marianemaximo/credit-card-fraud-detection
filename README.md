# Projeto 01 — Detecção de Fraude em Cartão de Crédito

Modelo de Machine Learning para identificar transações potencialmente fraudulentas com **dados extremamente desbalanceados (~0,17%)**, avaliando o **impacto financeiro de falsos positivos e falsos negativos** — e não apenas métricas de classificação.

> ⚠️ **Sobre os dados.** O dataset real (ULB / Kaggle, `creditcard.csv`) não pôde ser baixado no ambiente em que o projeto foi construído. A execução entregue usa **dados sintéticos** com a mesma estrutura (`Time`, `V1–V28`, `Amount`, `Class`), gerados por `src/synthetic.py`. A metodologia é a mesma, mas **os números abaixo são dos dados sintéticos e não valem para o dataset real**.
> Para usar o real: baixe `creditcard.csv` em <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>, salve em `data/raw/creditcard.csv` e rode o notebook (ele detecta o arquivo sozinho e imprime a fonte usada). Depois, revise as seções "Leitura" do notebook, que comentam os resultados da execução sintética.

## O diferencial: investigar, justificar e medir utilidade

| Etapa | O que foi feito |
|---|---|
| **EDA com inferência** | Mann-Whitney, KS, χ² + V de Cramér, tamanho de efeito, correção de Bonferroni, checagem de multicolinearidade |
| **Validação sem vazamento** | Split **temporal** (treino = passado, teste = futuro), teste aberto uma única vez, tudo (scaler, SMOTE, Isolation Forest) dentro do `Pipeline`, validação adversarial para deriva |
| **Desbalanceamento** | Nenhum tratamento × pesos de classe × RUS × SMOTE, em modelo linear e em *boosting*, com média ± desvio por fold |
| **Detecção de anomalias** | Isolation Forest (com e sem rótulos), Mahalanobis, LOF, autoencoder; e o escore de anomalia como *feature* de um modelo supervisionado |
| **Modelos** | Regressão Logística, Random Forest, XGBoost, LightGBM, LightGBM + IF; busca aleatória de hiperparâmetros |
| **Rigor estatístico** | IC 95% por bootstrap estratificado e **bootstrap pareado** entre modelos |
| **Negócio** | Calibração (Platt), limiar de decisão que minimiza **custo em dinheiro**, regra de Bayes por valor, precision/recall@k, sensibilidade às premissas de custo |
| **Explicabilidade** | SHAP + análise de erros por faixa de valor |

## Resultados (dados sintéticos; 200 mil transações, 360 fraudes, teste = 80 fraudes)

| Modelo | PR-AUC no teste (IC 95%) | ROC-AUC |
|---|---|---|
| LightGBM + escore IF (final) | **0,793** (0,70–0,87) | 0,981 |
| XGBoost | 0,782 (0,69–0,87) | 0,987 |
| LightGBM | 0,774 (0,68–0,86) | 0,984 |
| Random Forest | 0,771 (0,68–0,85) | 0,969 |
| Regressão Logística | 0,722 (0,63–0,81) | 0,965 |

**Impacto financeiro** (FP = 5; FN = valor da transação + 20; sem modelo, a perda é 10.383):

| Política | FP | FN | Precisão | Recall | Custo | Perda evitada |
|---|---|---|---|---|---|---|
| Limiar global — modelo final | 23 | 18 | 73% | 77,5% | 1.636 | **84%** (IC 72–92%) |
| Limiar global — Reg. Logística | 48 | 23 | 54% | 71% | 2.157 | 79% |
| Regra de Bayes por valor | 82 | 19 | 43% | 76% | 1.956 | 81% |
| "Bloquear os maiores valores" | 143 | 80 | 0% | 0% | 11.098 | −7% |

**Conclusões honestas**
- Acurácia (99,8% para "tudo legítimo") e ROC-AUC (0,92–0,99 para modelos muito diferentes) **não discriminam**; PR-AUC e custo, sim.
- SMOTE, undersampling e pesos de classe **não** superaram o modelo sem tratamento em PR-AUC; o desbalanceamento foi resolvido por métrica + calibração + limiar guiado por custo.
- Árvores superam o modelo linear de forma demonstrável (+0,07; IC95% [+0,04; +0,11]), mas **XGBoost, LightGBM e RF são estatisticamente indistinguíveis** com 80 fraudes no teste. O escore do Isolation Forest como feature adiciona pouco (dentro do ruído).
- O Isolation Forest, **sem usar rótulos**, recupera ~91% do PR-AUC da regressão logística supervisionada.
- A regra de decisão ótima na teoria (Bayes por valor) perdeu para um limiar global simples, porque a cauda de probabilidades baixas é mal calibrada e não verificável com tão poucas fraudes.
- O modelo evita ≥ 79% da perda em toda a grade de custos testada (FP de 1 a 50; custo fixo por fraude de 0 a 100).

## Estrutura

```
projeto_fraude/
├── notebooks/01_fraud_detection.ipynb   # análise completa, executada, com comentários
├── notebooks/01_fraud_detection.html    # mesma análise para leitura sem Jupyter
├── src/
│   ├── synthetic.py   # gerador de dados sintéticos (fallback)
│   ├── data.py        # carga + split temporal
│   ├── features.py    # FraudFeatures, IsolationScore (transformadores sklearn)
│   ├── models.py      # fábrica de pipelines (modelo × estratégia de desbalanceamento)
│   └── evaluation.py  # PR@k, custo, limiar ótimo, regra de Bayes, bootstrap
├── tests/test_core.py # custo vs. força bruta, ausência de vazamento, split temporal
├── reports/figures/   # 14 figuras;  reports/metrics.json
├── models/fraud_model.joblib  # pipeline final + calibração + limiar
├── data/raw/          # coloque aqui creditcard.csv
└── requirements.txt
```

## Como rodar

```bash
pip install -r requirements.txt
# opcional: data/raw/creditcard.csv (dataset real)
python -m pytest -q tests
jupyter nbconvert --to notebook --execute --inplace notebooks/01_fraud_detection.ipynb   # ~10 min em 2 CPUs
```

Usar o modelo salvo:

```python
import joblib, numpy as np
art = joblib.load("models/fraud_model.joblib")
raw = art["pipeline"].predict_proba(df_novo[["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]])[:, 1]
p = art["platt"].predict_proba(np.log(np.clip(raw, 1e-6, 1-1e-6) / (1 - np.clip(raw, 1e-6, 1-1e-6))).reshape(-1, 1))[:, 1]
bloquear = p >= art["threshold_global"]
```

## Limitações e próximos passos
Dados sintéticos nesta execução; janela de 48 h; poucas fraudes (intervalos largos); custos de FP/FN são premissas; sem identificador de cartão não há features comportamentais (velocidade, desvio do padrão do cliente). Em produção: monitorar deriva e recalibrar, tratar atraso de rótulos e *feedback loop*, features por cartão/comerciante, explicabilidade ao cliente e *champion/challenger* antes de bloquear de fato.
