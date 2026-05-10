"""
tests/test_recommender.py
=========================
Tests unitarios y de integración del sistema de recomendación.

Usa datos sintéticos (sin necesitar los archivos JSON de Yelp)
para que los tests sean rápidos y ejecutables en cualquier entorno.

Ejecutar:
    pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scipy.sparse import csr_matrix
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: datos sintéticos
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_reviews():
    """500 reseñas sintéticas con 50 usuarios y 100 negocios."""
    rng = np.random.default_rng(42)
    n   = 500
    users = [f"user_{i}" for i in rng.integers(0, 50, n)]
    bizs  = [f"biz_{i}"  for i in rng.integers(0, 100, n)]
    stars = rng.integers(1, 6, n).astype(float)
    dates = pd.date_range("2020-01-01", periods=n, freq="6h")
    return pd.DataFrame({
        "review_id":   [f"rev_{i}" for i in range(n)],
        "user_id":     users,
        "business_id": bizs,
        "stars":       stars,
        "date":        dates,
        "text":        [f"review text sample {i}" for i in range(n)],
        "useful":      rng.integers(0, 5, n),
        "funny":       rng.integers(0, 3, n),
        "cool":        rng.integers(0, 3, n),
    })


@pytest.fixture
def synthetic_businesses():
    """100 negocios con lat/lon/categorías."""
    rng  = np.random.default_rng(42)
    n    = 100
    cats = [["Restaurant"], ["Bar"], ["Coffee"], ["Shop"], ["Hotel"]]
    return pd.DataFrame({
        "business_id":  [f"biz_{i}" for i in range(n)],
        "name":         [f"Business {i}" for i in range(n)],
        "city":         rng.choice(["Las Vegas", "Phoenix", "Toronto"], n).tolist(),
        "state":        rng.choice(["NV", "AZ", "ON"], n).tolist(),
        "latitude":     rng.uniform(33.0, 45.0, n).tolist(),
        "longitude":    rng.uniform(-122.0, -73.0, n).tolist(),
        "stars":        rng.uniform(1.0, 5.0, n).round(1).tolist(),
        "review_count": rng.integers(5, 500, n).tolist(),
        "categories":   [cats[i % 5] for i in range(n)],
        "wifi":         rng.choice([True, False, None], n).tolist(),
        "outdoor":      rng.choice([True, False, None], n).tolist(),
        "price_range":  rng.choice(["1", "2", "3", "4", None], n).tolist(),
        "is_open":      rng.integers(0, 2, n).tolist(),
    })


@pytest.fixture
def synthetic_tips(synthetic_reviews):
    """50 tips basados en el mismo universo de usuarios/negocios."""
    rng = np.random.default_rng(99)
    n   = 50
    df  = synthetic_reviews.sample(n, random_state=42).reset_index(drop=True)
    return pd.DataFrame({
        "user_id":          df["user_id"],
        "business_id":      df["business_id"],
        "text":             [f"tip text {i}" for i in range(n)],
        "date":             df["date"],
        "compliment_count": rng.integers(0, 10, n),
    })


@pytest.fixture
def synthetic_checkins(synthetic_businesses):
    """Checkins sintéticos."""
    rng  = np.random.default_rng(7)
    rows = []
    for bid in synthetic_businesses["business_id"]:
        n_cks = rng.integers(2, 10)
        for _ in range(n_cks):
            dt = pd.Timestamp("2021-01-01") + pd.Timedelta(hours=int(rng.integers(0, 8760)))
            rows.append({"business_id": bid, "checkin_datetime": dt,
                         "hour": dt.hour, "day_of_week": dt.weekday(),
                         "day_type": "fin_semana" if dt.weekday() >= 5 else "laborable"})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: data_loader
# ─────────────────────────────────────────────────────────────────────────────

class TestDataLoader:
    def test_filter_interactions(self, synthetic_reviews):
        from utils.data_loader import filter_interactions
        import config as cfg
        cfg.MIN_USER_REVIEWS     = 2
        cfg.MIN_BUSINESS_REVIEWS = 2
        filtered = filter_interactions(synthetic_reviews)
        assert len(filtered) > 0
        assert filtered["user_id"].value_counts().min() >= 2

    def test_build_id_maps(self, synthetic_reviews):
        from utils.data_loader import build_id_maps, filter_interactions
        import config as cfg
        cfg.MIN_USER_REVIEWS = cfg.MIN_BUSINESS_REVIEWS = 2
        df = filter_interactions(synthetic_reviews)
        u2i, i2u, b2i, i2b = build_id_maps(df)
        assert len(u2i) == df["user_id"].nunique()
        assert len(b2i) == df["business_id"].nunique()

    def test_temporal_split(self, synthetic_reviews):
        from utils.data_loader import temporal_split
        train, val, test = temporal_split(synthetic_reviews)
        total = len(train) + len(val) + len(test)
        assert total == len(synthetic_reviews)
        assert train["date"].max() <= val["date"].min()
        assert val["date"].max() <= test["date"].min()

    def test_extract_request_context(self):
        from utils.data_loader import extract_request_context
        from datetime import datetime
        ctx = extract_request_context(datetime(2024, 6, 15, 14, 30))   # sábado 14:30
        assert ctx["franja"] == "tarde"
        assert ctx["day_type"] == "fin_semana"
        assert ctx["hour"] == 14


# ─────────────────────────────────────────────────────────────────────────────
# Tests: context_aware
# ─────────────────────────────────────────────────────────────────────────────

class TestContextAwareModel:
    def test_fit_and_boost(self, synthetic_checkins, synthetic_businesses):
        from models.context_aware import ContextAwareModel
        model = ContextAwareModel()
        model.fit(synthetic_checkins, synthetic_businesses)
        assert len(model.popularity) > 0
        bid = synthetic_businesses["business_id"].iloc[0]
        boost = model.get_context_boost(bid, "tarde", "laborable")
        assert 0.0 <= boost <= 1.0

    def test_adjust_scores(self, synthetic_checkins, synthetic_businesses):
        from models.context_aware import ContextAwareModel
        model = ContextAwareModel()
        model.fit(synthetic_checkins, synthetic_businesses)
        scores = {"biz_0": 0.8, "biz_1": 0.5, "biz_2": 0.3}
        adjusted = model.adjust_scores(scores, {"franja": "noche", "day_type": "laborable"})
        assert set(adjusted.keys()) == set(scores.keys())
        for v in adjusted.values():
            assert v >= 0

    def test_filter_by_distance(self, synthetic_checkins, synthetic_businesses):
        from models.context_aware import ContextAwareModel
        model = ContextAwareModel()
        model.fit(synthetic_checkins, synthetic_businesses)
        bids = synthetic_businesses["business_id"].tolist()[:20]
        filtered = model.filter_by_distance(bids, 37.0, -100.0, max_km=5000.0)
        assert len(filtered) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: content_based
# ─────────────────────────────────────────────────────────────────────────────

class TestContentBasedModel:
    def test_fit(self, synthetic_businesses, synthetic_reviews, synthetic_tips):
        from models.content_based import ContentBasedModel
        model = ContentBasedModel(use_sbert=False)
        model.fit(synthetic_businesses, synthetic_reviews, synthetic_tips)
        assert model.item_matrix is not None
        assert len(model.biz_ids) == len(synthetic_businesses)

    def test_get_similar(self, synthetic_businesses, synthetic_reviews, synthetic_tips):
        from models.content_based import ContentBasedModel
        model = ContentBasedModel(use_sbert=False)
        model.fit(synthetic_businesses, synthetic_reviews, synthetic_tips)
        similar = model.get_similar("biz_0", n=5)
        assert len(similar) == 5
        for bid, score in similar:
            assert 0.0 <= score <= 1.0

    def test_score_for_user(self, synthetic_businesses, synthetic_reviews, synthetic_tips):
        from models.content_based import ContentBasedModel
        model = ContentBasedModel(use_sbert=False)
        model.fit(synthetic_businesses, synthetic_reviews, synthetic_tips,
                  user_reviews=synthetic_reviews)
        uid = synthetic_reviews["user_id"].iloc[0]
        candidates = synthetic_businesses["business_id"].tolist()[:20]
        scores = model.score_for_user(uid, candidates)
        assert isinstance(scores, dict)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluationMetrics:
    def test_precision_recall(self):
        from utils.evaluation import precision_at_k, recall_at_k
        recs     = ["a", "b", "c", "d", "e"]
        relevant = {"b", "d", "f"}
        assert precision_at_k(recs, relevant, 5) == pytest.approx(2 / 5)
        assert recall_at_k(recs, relevant, 5)    == pytest.approx(2 / 3)

    def test_ndcg(self):
        from utils.evaluation import ndcg_at_k
        recs     = ["a", "b", "c"]
        relevant = {"a", "b"}
        ndcg     = ndcg_at_k(recs, relevant, 3)
        assert 0.0 < ndcg <= 1.0

    def test_map(self):
        from utils.evaluation import mean_average_precision
        recs = {"u1": ["a", "b", "c"], "u2": ["x", "a", "b"]}
        gt   = {"u1": {"a", "c"}, "u2": {"a"}}
        ap   = mean_average_precision(recs, gt, k=3)
        assert 0.0 < ap <= 1.0

    def test_catalog_coverage(self):
        from utils.evaluation import catalog_coverage
        recs = [["a", "b"], ["b", "c"], ["d"]]
        cov  = catalog_coverage(recs, total_items=10)
        assert cov == pytest.approx(4 / 10)

    def test_rmse_mae(self):
        from utils.evaluation import rmse, mae
        y_true = np.array([4.0, 3.0, 5.0, 2.0])
        y_pred = np.array([3.5, 3.0, 4.5, 2.5])
        assert rmse(y_true, y_pred) < 1.0
        assert mae(y_true, y_pred)  < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Test de integración: pipeline completo con datos sintéticos
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline(self,
                           synthetic_reviews,
                           synthetic_businesses,
                           synthetic_tips,
                           synthetic_checkins):
        """
        Verifica que el pipeline completo de entrenamiento e inferencia
        funciona end-to-end con datos sintéticos.
        """
        import config as cfg
        cfg.MIN_USER_REVIEWS = cfg.MIN_BUSINESS_REVIEWS = 2
        cfg.TOP_N_CANDIDATES = 20
        cfg.TOP_N_FINAL      = 5
        cfg.ALS_FACTORS      = 8
        cfg.ALS_ITERATIONS   = 3
        cfg.SVD_N_FACTORS    = 8
        cfg.SVD_N_EPOCHS     = 2
        cfg.TFIDF_MAX_FEATURES = 500

        from utils.data_loader import (
            filter_interactions, build_id_maps,
            build_implicit_matrix, temporal_split
        )
        from models.hybrid_recommender import HybridRecommender

        reviews_f = filter_interactions(synthetic_reviews)
        train, val, test = temporal_split(reviews_f)
        u2i, i2u, b2i, i2b = build_id_maps(train)
        imp_mat = build_implicit_matrix(train, synthetic_tips, u2i, b2i)

        rec = HybridRecommender()
        rec.fit(
            train_df        = train,
            implicit_matrix = imp_mat,
            businesses_df   = synthetic_businesses,
            checkins_df     = synthetic_checkins,
            reviews_df      = reviews_f,
            tips_df         = synthetic_tips,
            user2idx        = u2i,
            biz2idx         = b2i,
        )
        rec.businesses_df = synthetic_businesses

        # Inferencia para un usuario conocido
        uid = train["user_id"].iloc[0]
        results = rec.recommend(uid, top_n=5)
        assert isinstance(results, list)
        assert len(results) <= 5
        for r in results:
            assert "business_id" in r
            assert "hybrid_score" in r
            assert 0.0 <= r["hybrid_score"] <= 1.0 + 1e-9

    def test_cold_start_user(self,
                             synthetic_reviews,
                             synthetic_businesses,
                             synthetic_tips,
                             synthetic_checkins):
        """Un usuario desconocido debe recibir recomendaciones populares."""
        import config as cfg
        cfg.MIN_USER_REVIEWS = cfg.MIN_BUSINESS_REVIEWS = 2
        cfg.TOP_N_CANDIDATES = 20
        cfg.TOP_N_FINAL      = 5
        cfg.ALS_FACTORS      = 8
        cfg.ALS_ITERATIONS   = 3
        cfg.SVD_N_FACTORS    = 8
        cfg.SVD_N_EPOCHS     = 2
        cfg.TFIDF_MAX_FEATURES = 500

        from utils.data_loader import (
            filter_interactions, build_id_maps,
            build_implicit_matrix, temporal_split
        )
        from models.hybrid_recommender import HybridRecommender

        reviews_f = filter_interactions(synthetic_reviews)
        train, _, _ = temporal_split(reviews_f)
        u2i, i2u, b2i, i2b = build_id_maps(train)
        imp_mat = build_implicit_matrix(train, synthetic_tips, u2i, b2i)

        rec = HybridRecommender()
        rec.fit(train, imp_mat, synthetic_businesses, synthetic_checkins,
                reviews_f, synthetic_tips, u2i, b2i)
        rec.businesses_df = synthetic_businesses

        results = rec.recommend("UNKNOWN_USER_XYZ", top_n=5)
        assert isinstance(results, list)