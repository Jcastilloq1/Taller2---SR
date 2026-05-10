"""
models/collaborative_filtering.py
==================================
Filtrado colaborativo por factorización matricial.

Implementa dos variantes:
  1. SVDModel  – ratings EXPLÍCITOS via Surprise (SVD biased)
  2. ALSModel  – feedback IMPLÍCITO via implicit (ALS)

Interfaz común:
  .fit(train_df, user2idx, biz2idx)
  .predict(user_id, business_ids) → dict[business_id, score]
  .top_n(user_id, n, exclude_seen) → list[business_id]
  .save(path) / .load(path)
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Modelo 1: SVD explícito (Surprise)
# ─────────────────────────────────────────────────────────────────────────────

class SVDModel:
    """
    Filtrado colaborativo con SVD biased (Koren et al., 2009).
    Usa ratings explícitos de 1-5 estrellas de review.json.
    """

    def __init__(self):
        self.algo = None
        self.user2idx: dict = {}
        self.biz2idx:  dict = {}
        self.idx2biz:  dict = {}
        self._seen: dict[str, set] = {}   # user_id → set de business_id vistos

    def fit(self,
            train_df: pd.DataFrame,
            user2idx: dict,
            biz2idx: dict) -> "SVDModel":
        """
        Entrena SVD sobre el conjunto de train.
        train_df debe tener columnas: user_id, business_id, stars
        """
        try:
            from surprise import Dataset, Reader, SVD
            from surprise import accuracy as surp_acc
        except ImportError:
            raise ImportError("Instala scikit-surprise: pip install scikit-surprise")

        self.user2idx = user2idx
        self.biz2idx  = biz2idx
        self.idx2biz  = {v: k for k, v in biz2idx.items()}

        # Registrar ítems vistos por usuario (para excluirlos en inferencia)
        self._seen = (
            train_df.groupby("user_id")["business_id"]
            .apply(set).to_dict()
        )

        # Construir dataset de Surprise
        reader = Reader(rating_scale=(1, 5))
        data   = Dataset.load_from_df(
            train_df[["user_id", "business_id", "stars"]], reader
        )
        trainset = data.build_full_trainset()

        self.algo = SVD(
            n_factors   = cfg.SVD_N_FACTORS,
            n_epochs    = cfg.SVD_N_EPOCHS,
            lr_all      = cfg.SVD_LR_ALL,
            reg_all     = cfg.SVD_REG_ALL,
            biased      = cfg.SVD_BIASED,
            verbose     = False,
        )
        self.algo.fit(trainset)
        logger.info("SVDModel entrenado: %d factores, %d épocas",
                    cfg.SVD_N_FACTORS, cfg.SVD_N_EPOCHS)
        return self

    def predict(self,
                user_id: str,
                business_ids: list[str]) -> dict[str, float]:
        """
        Predice el rating esperado del usuario para cada negocio.
        Retorna dict {business_id: predicted_rating}.
        """
        if self.algo is None:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a .fit() primero.")
        scores = {}
        for bid in business_ids:
            pred = self.algo.predict(user_id, bid)
            scores[bid] = pred.est
        return scores

    def top_n(self,
              user_id: str,
              candidate_ids: list[str],
              exclude_seen: bool = True) -> dict[str, float]:
        """
        Devuelve scores para los candidatos, opcionalmente excluyendo
        los negocios que el usuario ya visitó.
        """
        if exclude_seen:
            seen = self._seen.get(user_id, set())
            candidate_ids = [b for b in candidate_ids if b not in seen]
        return self.predict(user_id, candidate_ids)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("SVDModel guardado en %s", path)

    @classmethod
    def load(cls, path: Path) -> "SVDModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("SVDModel cargado desde %s", path)
        return model


# ─────────────────────────────────────────────────────────────────────────────
# Modelo 2: ALS implícito (implicit library)
# ─────────────────────────────────────────────────────────────────────────────

class ALSModel:
    """
    Filtrado colaborativo con ALS para feedback implícito.
    Combina señales de reviews + tips ponderadas.
    Más robusto que SVD cuando los ratings son escasos.
    """

    def __init__(self):
        self.model = None
        self.user2idx: dict = {}
        self.biz2idx:  dict = {}
        self.idx2biz:  dict = {}
        self.implicit_matrix = None   # csr_matrix item×user (transpuesta)
        self._seen: dict[str, set] = {}

    def fit(self,
            implicit_matrix,      # scipy csr_matrix (users × items)
            user2idx: dict,
            biz2idx: dict,
            seen_df: pd.DataFrame) -> "ALSModel":
        """
        Entrena ALS sobre la matriz implícita usuario × negocio.
        seen_df: reviews para registrar ítems vistos.
        """
        try:
            import implicit
            from implicit.als import AlternatingLeastSquares
        except ImportError:
            raise ImportError("Instala implicit: pip install implicit")

        self.user2idx = user2idx
        self.biz2idx  = biz2idx
        self.idx2biz  = {v: k for k, v in biz2idx.items()}

        # ALS espera la matriz transpuesta: items × users
        self.implicit_matrix = implicit_matrix.T.tocsr()

        self.model = AlternatingLeastSquares(
            factors        = cfg.ALS_FACTORS,
            regularization = cfg.ALS_REGULARIZATION,
            iterations     = cfg.ALS_ITERATIONS,
            alpha          = cfg.ALS_ALPHA,
        )
        self.model.fit(self.implicit_matrix)

        self._seen = (
            seen_df.groupby("user_id")["business_id"]
            .apply(set).to_dict()
        )
        logger.info("ALSModel entrenado: %d factores, %d iters",
                    cfg.ALS_FACTORS, cfg.ALS_ITERATIONS)
        return self

    def top_n(self,
              user_id: str,
              n: int = cfg.TOP_N_CANDIDATES,
              exclude_seen: bool = True) -> dict[str, float]:
        """
        Devuelve los top-n negocios con mayor score ALS para el usuario.
        """
        if self.model is None:
            raise RuntimeError("El modelo no ha sido entrenado.")

        u_idx = self.user2idx.get(user_id)
        if u_idx is None:
            # Usuario nuevo (cold-start) → devuelve dict vacío;
            # el combinador fallback al modelo de contenido
            logger.warning("ALSModel: usuario desconocido '%s'", user_id)
            return {}

        # implicit devuelve (item_indices, scores)
        user_items = self.implicit_matrix.T.tocsr()
        ids_arr, scores_arr = self.model.recommend(
            u_idx,
            user_items[u_idx],
            N=n,
            filter_already_liked_items=exclude_seen,
        )
        results = {}
        for idx, score in zip(ids_arr, scores_arr):
            bid = self.idx2biz.get(int(idx))
            if bid:
                results[bid] = float(score)
        return results

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("ALSModel guardado en %s", path)

    @classmethod
    def load(cls, path: Path) -> "ALSModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("ALSModel cargado desde %s", path)
        return model