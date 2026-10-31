"""Fábrica de pipelines. Todo pré-processamento e reamostragem ficam DENTRO do Pipeline,
então em validação cruzada o ajuste (scaler, SMOTE, Isolation Forest...) usa apenas o fold de treino.
"""
from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .features import FraudFeatures, IsolationScore

SEED = 42
IMBALANCE = ("none", "class_weight", "rus", "smote")
MODELS = ("logreg", "rf", "xgb", "lgbm", "lgbm_iso")


def _classifier(model: str, imbalance: str):
    cw = imbalance == "class_weight"
    if model == "logreg":
        return LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced" if cw else None)
    if model == "rf":
        return RandomForestClassifier(
            n_estimators=120, max_depth=10, min_samples_leaf=10, max_samples=0.5,
            class_weight="balanced_subsample" if cw else None, n_jobs=2, random_state=SEED,
        )
    if model == "xgb":
        return XGBClassifier(
            n_estimators=250, learning_rate=0.05, max_depth=3, min_child_weight=5, subsample=0.8,
            colsample_bytree=0.7, reg_lambda=5, scale_pos_weight=10 if cw else 1,
            tree_method="hist", n_jobs=2, random_state=SEED,
        )
    if model in ("lgbm", "lgbm_iso"):
        return LGBMClassifier(
            n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=40, reg_lambda=5,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            scale_pos_weight=10 if cw else 1, n_jobs=2, verbose=-1, random_state=SEED,
        )
    raise ValueError(model)


def _sampler(imbalance: str):
    # razão 0.1 => 1 fraude para cada 10 legítimas após a reamostragem (balancear 1:1 é agressivo demais)
    if imbalance == "rus":
        return RandomUnderSampler(sampling_strategy=0.1, random_state=SEED)
    if imbalance == "smote":
        return SMOTE(sampling_strategy=0.1, k_neighbors=5, random_state=SEED)
    return None


def build(model: str = "lgbm", imbalance: str = "none") -> Pipeline:
    if imbalance not in IMBALANCE:
        raise ValueError(imbalance)
    steps = [("feat", FraudFeatures())]
    if model == "logreg":
        steps.append(("scale", StandardScaler()))
    if model == "lgbm_iso":
        steps.append(("iso", IsolationScore(random_state=SEED)))
    sampler = _sampler(imbalance)
    if sampler is not None:
        steps.append(("samp", sampler))
    steps.append(("clf", _classifier(model, imbalance)))
    return Pipeline(steps)
