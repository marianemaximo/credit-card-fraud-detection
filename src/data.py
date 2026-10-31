"""Carregamento e divisão temporal dos dados."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .synthetic import make_synthetic

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "creditcard.csv"
SYN_PATH = ROOT / "data" / "processed" / "creditcard_sintetico.csv"


def load_data() -> tuple[pd.DataFrame, str]:
    """Retorna (df, fonte). Usa o CSV real se existir; senão, gera dados sintéticos."""
    if RAW_PATH.exists():
        return pd.read_csv(RAW_PATH), "real (ULB/Kaggle)"
    if SYN_PATH.exists():
        return pd.read_csv(SYN_PATH), "SINTÉTICO (fallback)"
    df = make_synthetic()
    SYN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SYN_PATH, index=False)
    return df, "SINTÉTICO (fallback)"


def temporal_split(df: pd.DataFrame, train_frac: float = 0.70):
    """Divide por tempo: treino = passado, teste = futuro (evita vazamento temporal).

    Em fraude, o embaralhamento aleatório superestima o desempenho porque o modelo
    "vê o futuro" e padrões de campanhas de fraude vazam entre treino e teste.
    """
    df = df.sort_values("Time").reset_index(drop=True)
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
