"""Engenharia de atributos como transformadores sklearn (sem vazamento: tudo dentro do Pipeline)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest


class FraudFeatures(BaseEstimator, TransformerMixin):
    """Recebe o DataFrame bruto (Time, V1..V28, Amount) e devolve as features do modelo.

    - `Time` bruto é descartado (segundos desde a 1ª transação não generalizam);
      mantemos a hora do dia de forma cíclica (sin/cos) + flag de madrugada.
    - `Amount` é muito assimétrico -> log1p. Flags de valor baixo (card testing) e redondo.
    """

    def fit(self, X, y=None):
        self.v_cols_ = [c for c in X.columns if c.startswith("V")]
        return self

    def transform(self, X):
        hour = (X["Time"] // 3600) % 24
        out = X[self.v_cols_].copy()
        out["log_amount"] = np.log1p(X["Amount"])
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        out["is_night"] = (hour < 6).astype(int)
        out["amount_small"] = (X["Amount"] < 2).astype(int)
        out["amount_round"] = ((X["Amount"] > 0) & (X["Amount"] % 10 == 0)).astype(int)
        return out


class IsolationScore(BaseEstimator, TransformerMixin):
    """Acrescenta o escore de anomalia (Isolation Forest, NÃO supervisionado) como feature.

    O IF é ajustado só nos dados de treino de cada fold (dentro do Pipeline),
    então não há vazamento do rótulo nem do fold de validação.
    """

    def __init__(self, n_estimators=200, random_state=42, n_jobs=2):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        self.iso_ = IsolationForest(
            n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=self.n_jobs
        ).fit(X)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X["iso_score"] = -self.iso_.score_samples(X.drop(columns=["iso_score"], errors="ignore"))
        return X
