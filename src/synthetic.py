"""Gerador de dados sintéticos com a MESMA estrutura do dataset ULB/Kaggle
"Credit Card Fraud Detection" (Time, V1..V28, Amount, Class).

Serve apenas como *fallback* para quando `data/raw/creditcard.csv` não existe.
Ele NÃO reproduz o dataset real: os números do projeto mudarão ao usar o CSV original.

Decisões de desenho (para que o problema seja realista e não trivial):
- Legítimas: V_i ~ t-Student (caudas pesadas) com desvio decrescente, como componentes PCA.
- Fraudes em 3 subtipos: "evidente", "moderada" e "furtiva" (quase indistinguível).
  Isso impõe um teto de recall realista (nem toda fraude é detectável).
- Valor: fraudes têm muitos valores pequenos ("card testing") além de valores altos.
- Horário: fraudes são relativamente mais frequentes de madrugada.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Desvio-padrão aproximado das componentes no dataset real (decrescente, como em PCA)
_V_STD = np.array([
    1.96, 1.65, 1.52, 1.42, 1.38, 1.33, 1.24, 1.19, 1.10, 1.09, 1.02, 1.00, 1.00, 0.96,
    0.92, 0.88, 0.85, 0.84, 0.81, 0.77, 0.73, 0.73, 0.62, 0.61, 0.52, 0.48, 0.40, 0.33,
])

# Deslocamento médio das fraudes (inspirado no padrão do dataset real: V14, V12, V10, V17 ↓; V4, V11 ↑)
_FRAUD_SHIFT = {
    1: -4.8, 2: 3.6, 3: -7.0, 4: 4.5, 5: -3.0, 6: -1.4, 7: -5.6, 9: -2.6,
    10: -5.6, 11: 3.8, 12: -6.3, 14: -6.9, 16: -4.1, 17: -6.6, 18: -2.2,
}


def _sample_times(rng: np.random.Generator, n: int, peak_hour: float, amp: float) -> np.ndarray:
    """Amostra instantes em 48h com ciclo diário: densidade ∝ 1 + amp*cos(2π(h-peak)/24)."""
    grid = np.arange(0, 172_800, 30)
    hour = (grid / 3600.0) % 24
    w = 1.0 + amp * np.cos(2 * np.pi * (hour - peak_hour) / 24.0)
    w /= w.sum()
    idx = rng.choice(len(grid), size=n, p=w)
    return grid[idx] + rng.uniform(0, 30, size=n)


def make_synthetic(n: int = 200_000, fraud_rate: float = 0.0018, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(round(n * fraud_rate))
    n_legit = n - n_fraud

    # ---------------- legítimas ----------------
    df_free = 5.0
    t_scale = np.sqrt((df_free - 2) / df_free)  # t(df) padronizada para variância 1
    V_legit = rng.standard_t(df_free, size=(n_legit, 28)) * t_scale * _V_STD
    amt_legit = rng.lognormal(mean=3.1, sigma=1.45, size=n_legit)
    big = rng.random(n_legit) < 0.01
    amt_legit[big] = rng.lognormal(mean=6.5, sigma=0.8, size=big.sum())
    amt_legit = np.round(np.clip(amt_legit, 0, 25_000), 2)
    # ~4% de valores redondos (pagamentos de contas, etc.)
    rnd = rng.random(n_legit) < 0.04
    amt_legit[rnd] = np.round(amt_legit[rnd] / 10) * 10
    time_legit = _sample_times(rng, n_legit, peak_hour=15, amp=0.85)

    # ---------------- fraudes ----------------
    kind = rng.choice(["evidente", "moderada", "furtiva"], size=n_fraud, p=[0.50, 0.30, 0.20])
    strength = np.select([kind == "evidente", kind == "moderada"], [0.65, 0.30], default=0.08)
    spread = np.select([kind == "evidente", kind == "moderada"], [1.5, 1.3], default=1.1)
    shift = np.zeros(28)
    for i, s in _FRAUD_SHIFT.items():
        shift[i - 1] = s
    latent = 1.0 + 0.25 * rng.standard_normal((n_fraud, 1))  # intensidade variável por transação
    V_fraud = (
        rng.standard_t(df_free, size=(n_fraud, 28)) * t_scale * _V_STD * spread[:, None]
        + shift[None, :] * strength[:, None] * latent
    )
    # não linearidade: nas fraudes moderadas, V14 só "aparece" quando V10 também é extremo
    mod = kind == "moderada"
    V_fraud[mod, 13] *= (V_fraud[mod, 9] < -1.5) * 1.0 + 0.3

    u = rng.random(n_fraud)
    amt_fraud = np.where(
        u < 0.35, rng.lognormal(0.3, 0.8, n_fraud),            # testes de cartão (valores baixos)
        np.where(u < 0.80, rng.lognormal(4.0, 1.3, n_fraud),   # valores médios
                 rng.lognormal(5.5, 1.0, n_fraud)),             # valores altos
    )
    amt_fraud = np.round(np.clip(amt_fraud, 0, 25_000), 2)
    time_fraud = _sample_times(rng, n_fraud, peak_hour=3, amp=0.35)

    # ---------------- montagem ----------------
    cols = [f"V{i}" for i in range(1, 29)]
    legit = pd.DataFrame(V_legit, columns=cols)
    legit.insert(0, "Time", time_legit)
    legit["Amount"] = amt_legit
    legit["Class"] = 0
    fraud = pd.DataFrame(V_fraud, columns=cols)
    fraud.insert(0, "Time", time_fraud)
    fraud["Amount"] = amt_fraud
    fraud["Class"] = 1

    df = pd.concat([legit, fraud], ignore_index=True).sort_values("Time").reset_index(drop=True)
    return df
