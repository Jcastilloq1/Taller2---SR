"""
utils/evaluation.py
===================
Métricas de evaluación offline del sistema de recomendación.

Métricas implementadas:
  - Precisión@K, Recall@K, F1@K
  - NDCG@K (Normalized Discounted Cumulative Gain)
  - MAP@K  (Mean Average Precision)
  - MRR    (Mean Reciprocal Rank)
  - RMSE / MAE (para predicción de rating)
  - Intra-List Diversity (ILD)
  - Catalog Coverage

Todas las funciones están diseñadas para ser independientes del modelo
y reciben solo listas/arrays de IDs o scores.
"""

import numpy as np
import pandas as pd
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────────────────────────────────────
# Métricas de ranking (Top-N)
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fracción de los top-k recomendados que son relevantes."""
    top_k = recommended[:k]
    hits  = sum(1 for bid in top_k if bid in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fracción de relevantes que aparecen en los top-k."""
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits  = sum(1 for bid in top_k if bid in relevant)
    return hits / len(relevant)


def f1_at_k(recommended: list, relevant: set, k: int) -> float:
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def ndcg_at_k(recommended: list,
              relevant: set,
              k: int,
              ratings: Optional[dict] = None) -> float:
    """
    NDCG@K.
    Si se pasa ratings (dict business_id→stars), usa gain = stars - 1.
    Si no, gain binario (1 si relevante, 0 si no).
    """
    top_k = recommended[:k]
    dcg   = 0.0
    for i, bid in enumerate(top_k):
        if bid in relevant:
            gain = (ratings[bid] - 1) if (ratings and bid in ratings) else 1.0
            dcg += gain / np.log2(i + 2)

    # IDCG: si se usan ratings, ordena por rating descendente
    if ratings:
        ideal_gains = sorted(
            [ratings[b] - 1 for b in relevant if b in ratings], reverse=True
        )[:k]
    else:
        ideal_gains = [1.0] * min(len(relevant), k)

    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """AP@K para un solo usuario."""
    score, hits = 0.0, 0
    for i, bid in enumerate(recommended[:k]):
        if bid in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k) if relevant else 0.0


def mean_average_precision(recommendations: dict[str, list],
                            ground_truth: dict[str, set],
                            k: int = 10) -> float:
    """MAP@K sobre todos los usuarios."""
    aps = [
        average_precision_at_k(recommendations[u], ground_truth.get(u, set()), k)
        for u in recommendations
        if u in ground_truth
    ]
    return float(np.mean(aps)) if aps else 0.0


def mean_reciprocal_rank(recommendations: dict[str, list],
                          ground_truth: dict[str, set],
                          k: int = 10) -> float:
    """MRR: promedio del recíproco del rango del primer ítem relevante."""
    rrs = []
    for user_id, recs in recommendations.items():
        relevant = ground_truth.get(user_id, set())
        for i, bid in enumerate(recs[:k]):
            if bid in relevant:
                rrs.append(1.0 / (i + 1))
                break
        else:
            rrs.append(0.0)
    return float(np.mean(rrs)) if rrs else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Métricas de predicción de rating
# ─────────────────────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


# ─────────────────────────────────────────────────────────────────────────────
# Métricas adicionales de diversidad y cobertura
# ─────────────────────────────────────────────────────────────────────────────

def intra_list_diversity(recommended: list,
                          item_matrix,
                          biz2row: dict) -> float:
    """
    ILD: 1 − similitud promedio entre pares en la lista recomendada.
    Mide qué tan diversa es la lista (1=máx diversidad, 0=todos iguales).
    """
    rows = [biz2row[b] for b in recommended if b in biz2row]
    if len(rows) < 2:
        return 0.0
    vecs  = item_matrix[rows]
    if hasattr(vecs, "toarray"):
        vecs = vecs.toarray()
    vecs = np.asarray(vecs)
    sims = cosine_similarity(vecs)
    # Promedio de pares (sin diagonal)
    n    = len(rows)
    total_sim = (sims.sum() - n) / (n * (n - 1))
    return round(1.0 - float(total_sim), 4)


def catalog_coverage(all_recommendations: list[list[str]],
                      total_items: int) -> float:
    """
    Fracción del catálogo total que aparece en alguna lista recomendada.
    """
    unique_recommended = set(bid for recs in all_recommendations for bid in recs)
    return len(unique_recommended) / total_items if total_items > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Evaluación completa del sistema
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(recommender,
                   test_df: pd.DataFrame,
                   k: int = 10,
                   rating_threshold: float = 4.0,
                   sample_users: Optional[int] = 500) -> dict:
    """
    Evalúa el sistema híbrido sobre el conjunto de test.

    test_df:          reviews de test (user_id, business_id, stars)
    k:                longitud de las listas de recomendación
    rating_threshold: rating mínimo para considerar un ítem "relevante"
    sample_users:     evaluar sobre una muestra de usuarios (None = todos)

    Retorna dict con todas las métricas.
    """
    # Ground truth: para cada usuario, set de negocios con rating ≥ threshold
    ground_truth: dict[str, set] = {}
    user_ratings: dict[str, dict] = {}
    for _, row in test_df.iterrows():
        uid = row["user_id"]
        bid = row["business_id"]
        stars = float(row["stars"])
        if stars >= rating_threshold:
            ground_truth.setdefault(uid, set()).add(bid)
        user_ratings.setdefault(uid, {})[bid] = stars

    users_to_eval = list(ground_truth.keys())
    if sample_users and len(users_to_eval) > sample_users:
        rng = np.random.default_rng(42)
        users_to_eval = rng.choice(users_to_eval, sample_users, replace=False).tolist()

    # Generar recomendaciones
    all_recommendations: dict[str, list] = {}
    for user_id in users_to_eval:
        try:
            recs = recommender.recommend(user_id, top_n=k, exclude_seen=True)
            all_recommendations[user_id] = [r["business_id"] for r in recs]
        except Exception as e:
            all_recommendations[user_id] = []

    # Métricas de ranking
    p_k, r_k, f1_k, ndcg_k = [], [], [], []
    for user_id, recs in all_recommendations.items():
        relevant = ground_truth.get(user_id, set())
        ratings  = user_ratings.get(user_id, {})
        p_k.append(precision_at_k(recs, relevant, k))
        r_k.append(recall_at_k(recs, relevant, k))
        f1_k.append(f1_at_k(recs, relevant, k))
        ndcg_k.append(ndcg_at_k(recs, relevant, k, ratings))

    map_k = mean_average_precision(all_recommendations, ground_truth, k)
    mrr   = mean_reciprocal_rank(all_recommendations, ground_truth, k)

    # Cobertura del catálogo
    total_biz = len(recommender.biz2idx) if hasattr(recommender, "biz2idx") else 1
    coverage  = catalog_coverage(list(all_recommendations.values()), total_biz)

    # ILD promedio (requiere cb_model con item_matrix)
    ild_vals = []
    if hasattr(recommender, "cb_model") and recommender.cb_model.item_matrix is not None:
        for recs in all_recommendations.values():
            if recs:
                ild_vals.append(intra_list_diversity(
                    recs,
                    recommender.cb_model.item_matrix,
                    recommender.cb_model.biz2row,
                ))

    results = {
        f"precision@{k}":  round(float(np.mean(p_k)), 4),
        f"recall@{k}":     round(float(np.mean(r_k)), 4),
        f"f1@{k}":         round(float(np.mean(f1_k)), 4),
        f"ndcg@{k}":       round(float(np.mean(ndcg_k)), 4),
        f"map@{k}":        round(map_k, 4),
        "mrr":             round(mrr, 4),
        "catalog_coverage": round(coverage, 4),
        "ild":             round(float(np.mean(ild_vals)), 4) if ild_vals else None,
        "n_users_evaluated": len(users_to_eval),
    }
    return results