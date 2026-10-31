import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluation as ev
from src.data import temporal_split
from src.models import build
from src.synthetic import make_synthetic


def test_synthetic_schema_and_rate():
    df = make_synthetic(n=20_000, fraud_rate=0.002, seed=1)
    assert list(df.columns) == ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    assert df.Class.sum() == 40
    assert df.Time.is_monotonic_increasing and (df.Amount >= 0).all()


def test_temporal_split_has_no_overlap_in_time():
    df = make_synthetic(n=10_000, fraud_rate=0.003, seed=2)
    tr, te = temporal_split(df, 0.7)
    assert tr.Time.max() <= te.Time.min()


def test_cost_curve_matches_bruteforce():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.1).astype(int)
    amt = rng.lognormal(3, 1, 300)
    s = rng.random(300) + 0.5 * y
    _, cost = ev.cost_curve(y, amt, s, cost_fp=5, cost_fn_fixed=20)
    order = np.argsort(-s, kind="stable")
    for j in (0, 7, 50, 300):
        flag = np.zeros(300, bool)
        flag[order[:j]] = True
        assert np.isclose(cost[j], ev.evaluate_flags(y, amt, flag, 5, 20)["custo"])


def test_bayes_rule_is_cost_optimal_for_calibrated_probs():
    # com probabilidades verdadeiras conhecidas, a regra de Bayes deve ter custo esperado <= qualquer limiar global
    rng = np.random.default_rng(1)
    n = 200_000
    p = rng.beta(0.2, 20, n)
    y = (rng.random(n) < p).astype(int)
    amt = rng.lognormal(3, 1.5, n)
    bayes = ev.evaluate_flags(y, amt, ev.decide_by_expected_cost(p, amt))["custo"]
    best_global = min(ev.evaluate_flags(y, amt, p >= t)["custo"] for t in np.quantile(p, np.linspace(0.5, 0.9995, 60)))
    assert bayes <= best_global * 1.02


def test_pipeline_fit_predict_and_no_label_leak():
    df = make_synthetic(n=6_000, fraud_rate=0.02, seed=3)
    X, y = df.drop(columns="Class"), df.Class.values
    for name in ("logreg", "lgbm_iso"):
        for imb in ("none", "smote"):
            p = build(name, imb).fit(X, y)
            out = p.predict_proba(X.head(50))[:, 1]
            assert out.shape == (50,) and ((0 <= out) & (out <= 1)).all()
    # a feature de saída não pode conter a coluna alvo
    assert "Class" not in build("lgbm").fit(X, y)[:-1].transform(X).columns
