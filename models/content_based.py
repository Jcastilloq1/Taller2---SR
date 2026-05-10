"""
models/content_based.py
=======================
Modelo de recomendación basado en contenido.

Construye perfiles semánticos de negocios a partir de:
  - Categorías (business.json → categories)
  - Atributos (price_range, wifi, outdoor)
  - Texto agregado de reseñas (review.json → text)
  - Tips (tip.json → text)

Estrategia:
  1. TF-IDF sobre el texto combinado de cada negocio 
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

logger = logging.getLogger(__name__)


class ContentBasedModel:
    """
    Perfil de negocio basado en contenido textual + atributos.
    Usa coseno de similitud para recomendar negocios similares
    a los que el usuario valoró positivamente.
    """

    def __init__(self, use_sbert: bool = cfg.USE_SBERT):
        self.use_sbert      = use_sbert
        self.vectorizer     = None
        self.item_matrix    = None   # (n_businesses, n_features)
        self.biz_ids: list  = []     # orden de filas en item_matrix
        self.biz2row: dict  = {}     # business_id → fila en item_matrix
        self._sbert_model   = None
        self._seen: dict[str, set] = {}

    # ── Construcción del texto de cada negocio ───────────────────────────────

    @staticmethod
    def _build_business_corpus(
        businesses: pd.DataFrame,
        reviews: pd.DataFrame,
        tips: pd.DataFrame,
    ) -> pd.Series:
        """
        Para cada negocio construye un documento de texto que combina
        categorías, atributos y el texto de sus reseñas/tips.
        Devuelve una Serie indexada por business_id.
        """
        # Texto de categorías y atributos
        def biz_features(row) -> str:
            cats = " ".join(row["categories"]) if isinstance(row["categories"], list) else ""
            pr   = f"price{row['price_range']}" if pd.notna(row.get("price_range")) else ""
            wifi = "wifi" if row.get("wifi") else ""
            out  = "outdoor terrace" if row.get("outdoor") else ""
            city = row.get("city", "")
            return " ".join(filter(None, [cats, pr, wifi, out, city]))

        biz_text = businesses.set_index("business_id").apply(biz_features, axis=1)

        # Texto de reseñas (concatenar hasta 50 reseñas por negocio)
        rev_text = (
            reviews.groupby("business_id")["text"]
            .apply(lambda texts: " ".join(texts.dropna().astype(str).iloc[:50]))
        )

        # Texto de tips
        tip_text = (
            tips.groupby("business_id")["text"]
            .apply(lambda texts: " ".join(texts.dropna().astype(str).iloc[:30]))
        )

        corpus = biz_text.add(rev_text, fill_value="").add(tip_text, fill_value="")
        return corpus.fillna("")

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def fit(self,
            businesses: pd.DataFrame,
            reviews: pd.DataFrame,
            tips: pd.DataFrame,
            user_reviews: Optional[pd.DataFrame] = None) -> "ContentBasedModel":
        """
        Construye los vectores TF-IDF (y opcionalmente SBERT) para todos
        los negocios disponibles.

        user_reviews: para registrar ítems vistos por usuario.
        """
        logger.info("ContentBasedModel: construyendo corpus...")
        corpus = self._build_business_corpus(businesses, reviews, tips)
        self.biz_ids  = corpus.index.tolist()
        self.biz2row  = {bid: i for i, bid in enumerate(self.biz_ids)}

        if self.use_sbert:
            self._fit_sbert(corpus)
        else:
            self._fit_tfidf(corpus)

        if user_reviews is not None:
            self._seen = (
                user_reviews.groupby("user_id")["business_id"]
                .apply(set).to_dict()
            )
        logger.info("ContentBasedModel listo: %d negocios, matriz %s",
                    len(self.biz_ids), self.item_matrix.shape)
        return self

    def _fit_tfidf(self, corpus: pd.Series) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features = cfg.TFIDF_MAX_FEATURES,
            ngram_range  = cfg.TFIDF_NGRAM_RANGE,
            sublinear_tf = True,
            strip_accents = "unicode",
            min_df       = 2,
        )
        tfidf_matrix     = self.vectorizer.fit_transform(corpus.values)
        self.item_matrix = normalize(tfidf_matrix, norm="l2")

    def _fit_sbert(self, corpus: pd.Series) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Instala sentence-transformers: pip install sentence-transformers")

        logger.info("ContentBasedModel: codificando con Sentence-BERT '%s'...",
                    cfg.SBERT_MODEL_NAME)
        self._sbert_model = SentenceTransformer(cfg.SBERT_MODEL_NAME)

        # Limitar a 512 tokens truncando el texto
        texts = [t[:2000] for t in corpus.values]
        embeddings       = self._sbert_model.encode(texts, show_progress_bar=True,
                                                     batch_size=64, normalize_embeddings=True)
        self.item_matrix = embeddings   # ya normalizado

    # ── Inferencia ─────────────────────────────────────────────────────────

    def get_similar(self, business_id: str, n: int = 20) -> list[tuple[str, float]]:
        """
        Devuelve los n negocios más similares al dado.
        Retorna lista de (business_id, similarity_score).
        """
        row = self.biz2row.get(business_id)
        if row is None:
            return []
        vec  = self.item_matrix[row]
        sims = cosine_similarity(vec, self.item_matrix).flatten()
        sims[row] = -1   # excluir el propio negocio
        top_rows  = np.argsort(sims)[::-1][:n]
        return [(self.biz_ids[r], float(sims[r])) for r in top_rows]

    def score_for_user(self,
                       user_id: str,
                       candidate_ids: list[str],
                       user_history: Optional[list[str]] = None,
                       exclude_seen: bool = True) -> dict[str, float]:
        """
        Construye un perfil del usuario como la media de los vectores
        de negocios que visitó con rating ≥ 4. Luego calcula la similitud
        coseno con cada candidato.

        user_history: lista de business_id visitados por el usuario.
                      Si None, usa el historial registrado en .fit().
        """
        if user_history is None:
            user_history = list(self._seen.get(user_id, []))

        # Filtrar a negocios que están en el modelo
        history_rows = [self.biz2row[b] for b in user_history if b in self.biz2row]
        if not history_rows:
            # Cold-start: devolver score 0 para todos los candidatos
            return {b: 0.0 for b in candidate_ids}

        # Perfil del usuario = centroide de su historial
        user_vec = self.item_matrix[history_rows].mean(axis=0)
        if hasattr(user_vec, "A"):         # matriz dispersa
            user_vec = user_vec.A
        user_vec = normalize(np.array(user_vec).reshape(1, -1), norm="l2")

        # Filtrar candidatos
        if exclude_seen:
            seen = self._seen.get(user_id, set())
            candidate_ids = [b for b in candidate_ids if b not in seen]

        # Calcular similitudes solo para candidatos
        candidate_rows = [self.biz2row[b] for b in candidate_ids if b in self.biz2row]
        if not candidate_rows:
            return {}

        cand_matrix = self.item_matrix[candidate_rows]
        sims        = cosine_similarity(user_vec, cand_matrix).flatten()

        scores = {}
        for i, b in enumerate(b for b in candidate_ids if b in self.biz2row):
            scores[b] = float(sims[i])
        return scores

    # ── Persistencia ──────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("ContentBasedModel guardado en %s", path)

    @classmethod
    def load(cls, path: Path) -> "ContentBasedModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("ContentBasedModel cargado desde %s", path)
        return model