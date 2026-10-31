"""Métricas e avaliação orientada a negócio para detecção de fraude."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# Premissas de custo (configuráveis; ver sensibilidade no notebook)
COST_FP = 5.0          # custo de bloquear/revisar uma transação legítima (analista + atrito com cliente)
COST_FN_FIXED = 20.0   # custo fixo por fraude não detectada (chargeback fee + investigação), além do valor


def precision_recall_at_k(y, score, k_frac):
    """Capacidade operacional: se só dá para revisar o top-k% dos alertas."""
    y = np.asarray(y)
    k = max(1, int(round(len(y) * k_frac)))
    top = np.argsort(-np.asarray(score), kind="stable")[:k]
    tp = y[top].sum()
    return tp / k, tp / y.sum()


def cost_curve(y, amount, score, cost_fp=COST_FP, cost_fn_fixed=COST_FN_FIXED):
    """Custo total ao bloquear as top-j transações, para todo j (vetorizado).

    Custo = cost_fp * (#legítimas bloqueadas) + soma(valor + custo fixo) das fraudes NÃO bloqueadas.
    Retorna (scores_ordenados, custo_para_cada_j) com j = 0..n.
    """
    y, amount, score = map(np.asarray, (y, amount, score))
    order = np.argsort(-score, kind="stable")
    y_s, amt_s = y[order], amount[order]
    fn_cost_each = (amt_s + cost_fn_fixed) * y_s
    total_fn = fn_cost_each.sum()
    fp_cum = np.concatenate([[0], np.cumsum(1 - y_s)])
    fn_cum = np.concatenate([[0], np.cumsum(fn_cost_each)])
    cost = cost_fp * fp_cum + (total_fn - fn_cum)
    return score[order], cost


def best_threshold(y, amount, proba, cost_fp=COST_FP, cost_fn_fixed=COST_FN_FIXED):
    """Limiar de probabilidade que minimiza o custo (escolhido em dados de VALIDAÇÃO)."""
    s_sorted, cost = cost_curve(y, amount, proba, cost_fp, cost_fn_fixed)
    j = int(np.argmin(cost))
    thr = 1.0 + 1e-9 if j == 0 else s_sorted[j - 1]  # bloqueia scores >= thr
    return float(thr), float(cost[j])


def evaluate_flags(y, amount, flag, cost_fp=COST_FP, cost_fn_fixed=COST_FN_FIXED):
    """Resultado de negócio para um vetor booleano de alertas (bloqueios)."""
    y, amount, flag = np.asarray(y), np.asarray(amount), np.asarray(flag).astype(bool)
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    tn = int((~flag & (y == 0)).sum())
    fraud_total = float(amount[y == 1].sum())
    fraud_caught = float(amount[flag & (y == 1)].sum())
    cost = cost_fp * fp + float((amount[~flag & (y == 1)] + cost_fn_fixed).sum())
    cost_no_model = float((amount[y == 1] + cost_fn_fixed).sum())
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
        "alert_rate": float(flag.mean()),
        "valor_fraude_total": fraud_total, "valor_fraude_barrado": fraud_caught,
        "recall_valor": fraud_caught / max(fraud_total, 1e-9),
        "custo": cost, "custo_sem_modelo": cost_no_model,
        "economia_liquida": cost_no_model - cost,
    }


def evaluate_at_threshold(y, amount, proba, thr, **kw):
    out = evaluate_flags(y, amount, np.asarray(proba) >= thr, **kw)
    out["threshold"] = thr
    return out


def decide_by_expected_cost(proba, amount, cost_fp=COST_FP, cost_fn_fixed=COST_FN_FIXED):
    """Regra de Bayes por transação: bloqueia se p*(valor+custo_fixo) > (1-p)*cost_fp.

    Exige probabilidades CALIBRADAS. Equivale a um limiar que varia com o valor da transação:
    transações caras são bloqueadas com menos evidência; as baratas exigem mais.
    """
    p = np.asarray(proba)
    return p * (np.asarray(amount) + cost_fn_fixed) > (1 - p) * cost_fp


def bootstrap_ci(y, score, metric=average_precision_score, n_boot=1000, seed=0, alpha=0.05):
    """IC percentil por bootstrap estratificado (fraudes e legítimas reamostradas separadamente)."""
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(metric(y[idx], score[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return metric(y, score), float(lo), float(hi)


def paired_bootstrap_diff(y, score_a, score_b, metric=average_precision_score, n_boot=1000, seed=0):
    """Diferença pareada metric(A)-metric(B) com IC 95% e P(A>B) por bootstrap."""
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y), np.asarray(score_a), np.asarray(score_b)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d.append(metric(y[idx], a[idx]) - metric(y[idx], b[idx]))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


def summarize(y, score):
    return {"PR-AUC": average_precision_score(y, score), "ROC-AUC": roc_auc_score(y, score)}
