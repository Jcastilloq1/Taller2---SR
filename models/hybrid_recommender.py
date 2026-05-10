"""
models/hybrid_recommender.py
============================
Orquestador del sistema híbrido de recomendación.

Combina los tres modelos componentes:
  - SVDModel / ALSModel  (filtrado colaborativo por factorización)
  - ContextAwareModel    (sensible al contexto)
  - ContentBasedModel    (basado en contenido)

Pipeline de inferencia:
  1. CF genera candidatos (top-K global)
  2. ContextAware filtra por distancia y ajusta scores
  3. ContentBased aporta su score de similitud semántica
  4. Combinador ponderado: ŝ = α·CF + β·CTX + γ·CB
  5. Re-ranking MMR para balancear relevancia y diversidad
  6. Devolver Top-N con metadatos enriquecidos

Este módulo es el único que la API necesita importar.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import pickle

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from models.collaborative_filtering import SVDModel, ALSModel
from models.context_aware import ContextAwareModel
from models.content_based import ContentBasedModel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Normalización min-max a [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return scores
    vals  = np.array(list(scores.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Re-ranking MMR (Maximal Marginal Relevance)
# ─────────────────────────────────────────────────────────────────────────────

def _mmr_rerank(
    scores: dict[str, float],
    item_matrix,            # np array (n_all_items, n_features)
    biz2row: dict,
    n: int = cfg.TOP_N_FINAL,
    lambda_: float = cfg.MMR_LAMBDA,
) -> list[tuple[str, float]]:
    """
    Maximal Marginal Relevance:
      MMR = argmax [ λ·rel(i) − (1−λ)·max_{j∈S} sim(i, j) ]

    lambda_=1 → solo relevancia (sin diversidad)
    lambda_=0 → solo diversidad
    """
    candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected: list[tuple[str, float]] = []
    remaining = [bid for bid, _ in candidates if bid in biz2row]

    if not remaining:
        return candidates[:n]

    while remaining and len(selected) < n:
        if not selected:
            # Primer ítem: el de mayor score
            best = remaining.pop(0)
            selected.append((best, scores[best]))
            continue

        # Vectores del conjunto ya seleccionado
        sel_rows   = np.array([biz2row[b] for b, _ in selected])
        sel_vecs   = item_matrix[sel_rows]   # (|S|, D)

        best_bid, best_mmr = None, -np.inf
        for bid in remaining:
            row  = biz2row[bid]
            vec  = item_matrix[[row]]         # (1, D)
            rel  = scores[bid]

            # Máxima similitud con ítems ya seleccionados
            if hasattr(sel_vecs, "toarray"):
                sv = sel_vecs.toarray()
                v  = vec.toarray()
            else:
                sv = np.asarray(sel_vecs)
                v  = np.asarray(vec)

            sims     = sv @ v.T              # (|S|, 1)
            max_sim  = float(sims.max())

            mmr_score = lambda_ * rel - (1 - lambda_) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_bid = bid

        if best_bid:
            remaining.remove(best_bid)
            selected.append((best_bid, scores[best_bid]))

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# HybridRecommender
# ─────────────────────────────────────────────────────────────────────────────

class HybridRecommender:
    """
    Sistema híbrido de recomendación para el dataset Yelp.

    Uso básico:
        rec = HybridRecommender()
        rec.fit(train_df, implicit_matrix, businesses_df,
                checkins_df, reviews_df, tips_df,
                user2idx, biz2idx)
        results = rec.recommend(user_id="abc123", top_n=10,
                                user_lat=37.78, user_lon=-122.4)
    """

    def __init__(self,
                 weight_cf:  float = cfg.HYBRID_WEIGHT_CF,
                 weight_ctx: float = cfg.HYBRID_WEIGHT_CTX,
                 weight_cb:  float = cfg.HYBRID_WEIGHT_CB):
        self.weight_cf  = weight_cf
        self.weight_ctx = weight_ctx
        self.weight_cb  = weight_cb

        self.svd_model  = SVDModel()
        self.als_model  = ALSModel()
        self.ctx_model  = ContextAwareModel()
        self.cb_model   = ContentBasedModel()

        self.businesses_df: Optional[pd.DataFrame] = None
        self.user2idx: dict = {}
        self.biz2idx:  dict = {}
        self.idx2biz:  dict = {}
        self._all_biz_ids: list = []
        self._user_history: dict[str, list] = {}   # user_id → [biz_ids visitados]

    # ── Entrenamiento completo ────────────────────────────────────────────────

    def fit(self,
            train_df: pd.DataFrame,
            implicit_matrix,
            businesses_df: pd.DataFrame,
            checkins_df: pd.DataFrame,
            reviews_df: pd.DataFrame,
            tips_df: pd.DataFrame,
            user2idx: dict,
            biz2idx: dict) -> "HybridRecommender":
        """
        Entrena los tres modelos componentes.

        train_df:        reviews de entrenamiento (user_id, business_id, stars, date)
        implicit_matrix: scipy csr_matrix (users × items) para ALS
        businesses_df:   DataFrame de negocios con lat/lon/categorías
        checkins_df:     DataFrame de checkins (salida de load_checkins)
        reviews_df:      Todas las reseñas (para historial de usuario)
        tips_df:         DataFrame de tips
        user2idx/biz2idx: mapeos de ID → índice entero
        """
        self.businesses_df = businesses_df
        self.user2idx = user2idx
        self.biz2idx  = biz2idx
        self.idx2biz  = {v: k for k, v in biz2idx.items()}
        self._all_biz_ids = list(biz2idx.keys())

        # Historial de usuario (para scores de CB y exclusión de vistos)
        self._user_history = (
            reviews_df.groupby("user_id")["business_id"]
            .apply(list).to_dict()
        )

        logger.info("Entrenando SVDModel...")
        self.svd_model.fit(train_df, user2idx, biz2idx)

        logger.info("Entrenando ALSModel...")
        self.als_model.fit(implicit_matrix, user2idx, biz2idx, train_df)

        logger.info("Entrenando ContextAwareModel...")
        self.ctx_model.fit(checkins_df, businesses_df)

        logger.info("Entrenando ContentBasedModel...")
        self.cb_model.fit(businesses_df, reviews_df, tips_df, train_df)

        logger.info("HybridRecommender listo.")
        return self

    # ── Inferencia ────────────────────────────────────────────────────────────

    def recommend(self,
                  user_id: str,
                  top_n: int = cfg.TOP_N_FINAL,
                  user_lat: Optional[float] = None,
                  user_lon: Optional[float] = None,
                  request_datetime: Optional[datetime] = None,
                  user_history: Optional[list[str]] = None,
                  city_filter: Optional[str] = None,
                  category_filter: Optional[list[str]] = None,
                  exclude_seen: bool = True) -> list[dict]:
        """
        Genera recomendaciones para un usuario.

        Parámetros de la petición (todos opcionales):
          user_lat/user_lon   : coordenadas GPS del usuario → activa filtro de distancia
          request_datetime    : momento de la petición     → activa contexto temporal
          user_history        : lista de business_id visitados (override del historial)
          city_filter         : solo recomendar negocios en esa ciudad
          category_filter     : lista de categorías deseadas
          exclude_seen        : excluir negocios ya visitados

        Retorna lista de dicts con los campos:
          business_id, name, city, stars, categories,
          hybrid_score, cf_score, ctx_score, cb_score,
          distance_km (si se proporcionaron coordenadas)
        """
        from utils.data_loader import extract_request_context
        context = extract_request_context(request_datetime)

        # ── 1. Obtener candidatos del modelo CF ──────────────────────────────
        # ALS genera top-K rápido
        cf_scores = self.als_model.top_n(
            user_id, n=cfg.TOP_N_CANDIDATES, exclude_seen=exclude_seen
        )

        # Si ALS no tiene historial (usuario nuevo), usar todos los negocios
        # filtrados por ciudad/categoría
        if not cf_scores:
            logger.debug("Cold-start para usuario '%s' → usando candidatos populares", user_id)
            candidates = self._get_popular_candidates(
                n=cfg.TOP_N_CANDIDATES,
                city=city_filter,
                categories=category_filter,
            )
            cf_scores = {b: 0.0 for b in candidates}
        else:
            candidates = list(cf_scores.keys())

        # ── 2. Filtros opcionales de ciudad y categoría ──────────────────────
        if city_filter or category_filter:
            candidates = self._apply_filters(candidates, city_filter, category_filter)
            cf_scores  = {k: v for k, v in cf_scores.items() if k in candidates}

        # ── 3. Filtro geoespacial opcional ───────────────────────────────────
        if user_lat is not None and user_lon is not None:
            candidates = self.ctx_model.filter_by_distance(
                candidates, user_lat, user_lon
            )
            cf_scores = {k: v for k, v in cf_scores.items() if k in candidates}

        if not candidates:
            return []

        # ── 4. Score contextual ──────────────────────────────────────────────
        ctx_scores_raw = self.ctx_model.adjust_scores(
            cf_scores, context, user_lat, user_lon
        )

        # ── 5. Score de contenido ────────────────────────────────────────────
        history = user_history or self._user_history.get(user_id, [])
        cb_scores_raw = self.cb_model.score_for_user(
            user_id, candidates, user_history=history,
            exclude_seen=exclude_seen
        )

        # ── 6. Normalizar a [0,1] ────────────────────────────────────────────
        cf_norm  = _minmax(cf_scores)
        ctx_norm = _minmax(ctx_scores_raw)
        cb_norm  = _minmax(cb_scores_raw)

        # ── 7. Combinador ponderado ──────────────────────────────────────────
        hybrid_scores: dict[str, float] = {}
        for bid in candidates:
            s_cf  = cf_norm.get(bid, 0.0)
            s_ctx = ctx_norm.get(bid, 0.0)
            s_cb  = cb_norm.get(bid, 0.0)
            hybrid_scores[bid] = (
                self.weight_cf  * s_cf +
                self.weight_ctx * s_ctx +
                self.weight_cb  * s_cb
            )

        # ── 8. Re-ranking MMR ────────────────────────────────────────────────
        reranked = _mmr_rerank(
            hybrid_scores,
            self.cb_model.item_matrix,
            self.cb_model.biz2row,
            n=top_n,
        )

        # ── 9. Enriquecer con metadatos del negocio ──────────────────────────
        biz_meta = {}
        if self.businesses_df is not None:
            biz_meta = self.businesses_df.set_index("business_id").to_dict("index")

        results = []
        for rank, (bid, hybrid_score) in enumerate(reranked, start=1):
            meta = biz_meta.get(bid, {})
            entry = {
                "rank":         rank,
                "business_id":  bid,
                "name":         meta.get("name", ""),
                "city":         meta.get("city", ""),
                "state":        meta.get("state", ""),
                "stars":        meta.get("stars", None),
                "review_count": meta.get("review_count", None),
                "categories":   meta.get("categories", []),
                "hybrid_score": round(hybrid_score, 4),
                "cf_score":     round(cf_norm.get(bid, 0.0), 4),
                "ctx_score":    round(ctx_norm.get(bid, 0.0), 4),
                "cb_score":     round(cb_norm.get(bid, 0.0), 4),
            }
            # Distancia si se proporcionaron coordenadas
            if user_lat is not None and user_lon is not None:
                coords = self.ctx_model._businesses_lat_lon.get(bid)
                if coords:
                    from models.context_aware import _haversine_km
                    import numpy as _np
                    dist = float(_haversine_km(
                        user_lat, user_lon,
                        _np.array([coords[0]]), _np.array([coords[1]])
                    )[0])
                    entry["distance_km"] = round(dist, 2)
            results.append(entry)

        return results

    # ── Helpers internos ──────────────────────────────────────────────────────

    def _get_popular_candidates(self,
                                 n: int,
                                 city: Optional[str] = None,
                                 categories: Optional[list[str]] = None) -> list[str]:
        """Fallback cold-start: negocios más populares filtrados."""
        if self.businesses_df is None:
            return self._all_biz_ids[:n]
        df = self.businesses_df.copy()
        if city:
            df = df[df["city"].str.lower() == city.lower()]
        if categories:
            cats_set = set(c.lower() for c in categories)
            df = df[df["categories"].apply(
                lambda cats: bool(set(c.lower() for c in cats) & cats_set)
                if isinstance(cats, list) else False
            )]
        df = df.sort_values("review_count", ascending=False)
        return df["business_id"].iloc[:n].tolist()

    def _apply_filters(self,
                        candidates: list[str],
                        city: Optional[str],
                        categories: Optional[list[str]]) -> list[str]:
        if self.businesses_df is None:
            return candidates
        df = self.businesses_df.set_index("business_id")
        result = []
        for bid in candidates:
            if bid not in df.index:
                continue
            row = df.loc[bid]
            if city and row.get("city", "").lower() != city.lower():
                continue
            if categories:
                biz_cats = set(c.lower() for c in (row.get("categories") or []))
                if not biz_cats.intersection(c.lower() for c in categories):
                    continue
            result.append(bid)
        return result

    # ── Persistencia ──────────────────────────────────────────────────────────

    def save(self, directory: Path) -> None:
        """Guarda todos los modelos en el directorio especificado."""
        directory.mkdir(exist_ok=True)
        self.svd_model.save(directory / "svd_model.pkl")
        self.als_model.save(directory / "als_model.pkl")
        self.ctx_model.save(directory / "ctx_model.pkl")
        self.cb_model.save(directory  / "cb_model.pkl")
        # Guardar metadatos del orquestador
        meta = {
            "weight_cf":  self.weight_cf,
            "weight_ctx": self.weight_ctx,
            "weight_cb":  self.weight_cb,
            "user2idx":   self.user2idx,
            "biz2idx":    self.biz2idx,
            "_user_history": self._user_history,
        }
        with open(directory / "hybrid_meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        logger.info("HybridRecommender guardado en %s", directory)

    @classmethod
    def load(cls, directory: Path) -> "HybridRecommender":
        """Reconstruye el orquestador desde los modelos guardados."""
        with open(directory / "hybrid_meta.pkl", "rb") as f:
            meta = pickle.load(f)

        rec = cls(
            weight_cf  = meta["weight_cf"],
            weight_ctx = meta["weight_ctx"],
            weight_cb  = meta["weight_cb"],
        )
        rec.user2idx        = meta["user2idx"]
        rec.biz2idx         = meta["biz2idx"]
        rec.idx2biz         = {v: k for k, v in meta["biz2idx"].items()}
        rec._all_biz_ids    = list(meta["biz2idx"].keys())
        rec._user_history   = meta["_user_history"]

        rec.svd_model = SVDModel.load(directory / "svd_model.pkl")
        rec.als_model = ALSModel.load(directory / "als_model.pkl")
        rec.ctx_model = ContextAwareModel.load(directory / "ctx_model.pkl")
        rec.cb_model  = ContentBasedModel.load(directory / "cb_model.pkl")

        logger.info("HybridRecommender cargado desde %s", directory)
        return rec