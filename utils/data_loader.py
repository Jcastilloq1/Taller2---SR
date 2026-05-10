"""
utils/data_loader.py
====================
Carga y pre-procesamiento del dataset de Yelp.

Responsabilidades:
  - Leer los archivos JSON línea a línea (formato Yelp)
  - Filtrar usuarios y negocios con suficientes interacciones
  - Construir la matriz usuario-ítem
  - Extraer señales de contexto (hora, día, distancia)
  - Realizar el split temporal (train / val / test)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lectura de archivos JSON de Yelp 
# ─────────────────────────────────────────────────────────────────────────────

def _read_json_lines(filepath: Path, max_rows: Optional[int] = None) -> list[dict]:
    """Lee un archivo JSON de Yelp (una entidad JSON por línea)."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Cargados %d registros de %s", len(records), filepath.name)
    return records


def load_businesses(filepath: Path = cfg.BUSINESS_FILE) -> pd.DataFrame:
    """
    Carga business.json.
    Columnas clave: business_id, name, city, state, latitude, longitude,
                    stars, review_count, categories, attributes, hours, is_open
    """
    records = _read_json_lines(filepath, cfg.MAX_REVIEWS_LOAD)
    df = pd.DataFrame(records)

    # Normalizar categorías: string → lista de strings
    df["categories"] = df["categories"].fillna("").apply(
        lambda x: [c.strip() for c in x.split(",")] if x else []
    )

    # Aplanar atributos anidados más relevantes
    def _extract_attr(attrs: Optional[dict], key: str, default=None):
        if not isinstance(attrs, dict):
            return default
        val = attrs.get(key, default)
        # Yelp codifica booleanos como strings: 'True', 'False'
        if val in ("True", "False"):
            return val == "True"
        return val

    df["wifi"]          = df["attributes"].apply(lambda a: _extract_attr(a, "WiFi"))
    df["outdoor"]       = df["attributes"].apply(lambda a: _extract_attr(a, "OutdoorSeating"))
    df["price_range"]   = df["attributes"].apply(lambda a: _extract_attr(a, "RestaurantsPriceRange2"))

    return df[["business_id", "name", "city", "state",
               "latitude", "longitude", "stars", "review_count",
               "categories", "wifi", "outdoor", "price_range", "is_open"]]


def load_reviews(filepath: Path = cfg.REVIEW_FILE) -> pd.DataFrame:
    """
    Carga review.json.
    Columnas clave: review_id, user_id, business_id, stars, date, text,
                    useful, funny, cool
    """
    records = _read_json_lines(filepath, cfg.MAX_REVIEWS_LOAD)
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["stars"] = df["stars"].astype(float)
    return df[["review_id", "user_id", "business_id",
               "stars", "date", "text", "useful", "funny", "cool"]]


def load_users(filepath: Path = cfg.USER_FILE) -> pd.DataFrame:
    """Carga user.json con metadatos del usuario."""
    records = _read_json_lines(filepath, cfg.MAX_REVIEWS_LOAD)
    df = pd.DataFrame(records)
    df["yelping_since"] = pd.to_datetime(df["yelping_since"])
    df["friends_count"] = df["friends"].fillna("").apply(
        lambda x: len(x.split(",")) if x and x != "None" else 0
    )
    return df[["user_id", "name", "review_count",
               "average_stars", "fans", "friends_count", "yelping_since"]]


def load_checkins(filepath: Path = cfg.CHECKIN_FILE) -> pd.DataFrame:
    """
    Carga checkin.json.
    Explota la lista de timestamps en filas individuales.
    Columnas: business_id, checkin_datetime, hour, day_of_week, day_type
    """
    records = _read_json_lines(filepath)
    rows = []
    for rec in records:
        bid = rec["business_id"]
        for ts in rec.get("date", "").split(","):
            ts = ts.strip()
            if ts:
                rows.append({"business_id": bid, "checkin_datetime": ts})
    df = pd.DataFrame(rows)
    df["checkin_datetime"] = pd.to_datetime(df["checkin_datetime"])
    df["hour"]        = df["checkin_datetime"].dt.hour
    df["day_of_week"] = df["checkin_datetime"].dt.dayofweek  # 0=lunes
    df["day_type"]    = df["day_of_week"].apply(
        lambda d: "fin_semana" if d >= 5 else "laborable"
    )
    return df


def load_tips(filepath: Path = cfg.TIP_FILE) -> pd.DataFrame:
    """Carga tip.json como señal implícita de interacción."""
    records = _read_json_lines(filepath, cfg.MAX_REVIEWS_LOAD)
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df[["user_id", "business_id", "text", "date", "compliment_count"]]


# ─────────────────────────────────────────────────────────────────────────────
# Filtrado y construcción de datasets limpios
# ─────────────────────────────────────────────────────────────────────────────

def filter_interactions(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica filtros de k-core: solo usuarios y negocios con al menos
    MIN_USER_REVIEWS / MIN_BUSINESS_REVIEWS interacciones.
    Itera hasta convergencia.
    """
    df = reviews.copy()
    prev_len = -1
    iteration = 0
    while len(df) != prev_len:
        prev_len = len(df)
        user_counts = df.groupby("user_id")["review_id"].count()
        biz_counts  = df.groupby("business_id")["review_id"].count()
        valid_users = user_counts[user_counts >= cfg.MIN_USER_REVIEWS].index
        valid_biz   = biz_counts[biz_counts >= cfg.MIN_BUSINESS_REVIEWS].index
        df = df[df["user_id"].isin(valid_users) & df["business_id"].isin(valid_biz)]
        iteration += 1
    logger.info("k-core filtrado en %d iteraciones → %d interacciones, %d usuarios, %d negocios",
                iteration, len(df),
                df["user_id"].nunique(),
                df["business_id"].nunique())
    return df.reset_index(drop=True)


def build_id_maps(reviews: pd.DataFrame) -> tuple[dict, dict, dict, dict]:
    """
    Crea mapeos enteros para usuarios y negocios.
    Retorna: user2idx, idx2user, biz2idx, idx2biz
    """
    users = sorted(reviews["user_id"].unique())
    bizs  = sorted(reviews["business_id"].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    biz2idx  = {b: i for i, b in enumerate(bizs)}
    return user2idx, {i: u for u, i in user2idx.items()}, \
           biz2idx,  {i: b for b, i in biz2idx.items()}


def build_rating_matrix(reviews: pd.DataFrame,
                         user2idx: dict,
                         biz2idx: dict) -> csr_matrix:
    """
    Construye la matriz dispersa usuario × negocio con ratings explícitos (1-5).
    """
    rows = reviews["user_id"].map(user2idx)
    cols = reviews["business_id"].map(biz2idx)
    data = reviews["stars"].values
    n_users = len(user2idx)
    n_biz   = len(biz2idx)
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_biz), dtype=np.float32)


def build_implicit_matrix(reviews: pd.DataFrame,
                           tips: pd.DataFrame,
                           user2idx: dict,
                           biz2idx: dict) -> csr_matrix:
    """
    Construye matriz de feedback implícito combinando:
      - reseñas (peso = 1 + useful + funny + cool normalizado)
      - tips (peso = 1 + compliment_count)
    Usado por el modelo ALS.
    """
    rows, cols, data = [], [], []

    # Reseñas
    for _, row in reviews.iterrows():
        u = user2idx.get(row["user_id"])
        b = biz2idx.get(row["business_id"])
        if u is None or b is None:
            continue
        weight = 1.0 + (row.get("useful", 0) + row.get("funny", 0) + row.get("cool", 0)) / 10.0
        rows.append(u); cols.append(b); data.append(weight)

    # Tips
    for _, row in tips.iterrows():
        u = user2idx.get(row["user_id"])
        b = biz2idx.get(row["business_id"])
        if u is None or b is None:
            continue
        weight = 0.5 + row.get("compliment_count", 0) / 10.0
        rows.append(u); cols.append(b); data.append(weight)

    n_users = len(user2idx)
    n_biz   = len(biz2idx)
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_biz), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Split temporal
# ─────────────────────────────────────────────────────────────────────────────

def temporal_split(reviews: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide por fecha para evitar data leakage.
    - Train: primeras 70% de fechas
    - Val:   siguiente 10%
    - Test:  últimas 20%
    """
    df = reviews.sort_values("date").reset_index(drop=True)
    n  = len(df)
    t1 = int(n * cfg.TRAIN_RATIO)
    t2 = int(n * (cfg.TRAIN_RATIO + cfg.VAL_RATIO))
    train = df.iloc[:t1]
    val   = df.iloc[t1:t2]
    test  = df.iloc[t2:]
    logger.info("Split temporal → train=%d | val=%d | test=%d", len(train), len(val), len(test))
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Extracción de contexto para una petición de recomendación en tiempo real
# ─────────────────────────────────────────────────────────────────────────────

def extract_request_context(dt: Optional[datetime] = None) -> dict:
    """
    Extrae las variables de contexto de una petición entrante.
    Si no se pasa datetime, usa el momento actual.
    """
    if dt is None:
        dt = datetime.now()
    hour = dt.hour
    dow  = dt.weekday()

    # Franja horaria
    franja = "noche"
    for nombre, (h_ini, h_fin) in cfg.CONTEXT_HOURS.items():
        if h_ini <= hour < h_fin:
            franja = nombre
            break

    day_type = "fin_semana" if dow >= 5 else "laborable"
    return {
        "hour":      hour,
        "franja":    franja,
        "day_of_week": dow,
        "day_type":  day_type,
    }