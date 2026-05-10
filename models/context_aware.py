"""
models/context_aware.py
=======================
Modelo de recomendación sensible al contexto (CARS).

Estrategia implementada: Post-filtering contextual.
  1. El CF/CB genera una lista de candidatos.
  2. Este módulo ajusta (boost/penaliza) los scores según:
     - Franja horaria (checkins históricos por hora)
     - Tipo de día (laborable vs. fin de semana)
     - Distancia geoespacial del usuario al negocio
     - Popularidad horaria del negocio

Ventaja: No requiere reentrenamiento cuando cambia el contexto,
lo que lo hace ideal para inferencia en tiempo real vía API.

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


def _haversine_km(lat1: float, lon1: float,
                   lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Distancia haversine en km entre un punto y un array de puntos."""
    R = 6371.0
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


class ContextAwareModel:
    """
    Modelo CARS por post-filtering.

    Aprende distribuciones históricas de checkins por negocio,
    hora del día y tipo de día. En inferencia, multiplica los
    scores de CF/CB por un factor de popularidad contextual y
    aplica penalización por distancia.
    """

    def __init__(self):
        # popularity[business_id][franja][day_type] = conteo normalizado
        self.popularity: dict = {}
        # hora_peak[business_id] = hora de máxima afluencia
        self.hora_peak: dict = {}
        self._businesses_lat_lon: dict = {}   # business_id → (lat, lon)

    def fit(self,
            checkins: pd.DataFrame,
            businesses: pd.DataFrame) -> "ContextAwareModel":
        """
        Calcula perfiles de popularidad horaria y de día a partir
        de checkin.json.
        businesses necesita: business_id, latitude, longitude
        """
        logger.info("ContextAwareModel: calculando perfiles de popularidad...")

        # Mapa de coordenadas
        self._businesses_lat_lon = {
            row["business_id"]: (row["latitude"], row["longitude"])
            for _, row in businesses[["business_id", "latitude", "longitude"]].iterrows()
            if pd.notna(row["latitude"]) and pd.notna(row["longitude"])
        }

        # Calcular franja y day_type para cada checkin
        checkins = checkins.copy()
        checkins["franja"] = checkins["hour"].apply(self._hour_to_franja)

        # Conteos por business × franja × day_type
        grouped = (
            checkins.groupby(["business_id", "franja", "day_type"])
            .size()
            .reset_index(name="count")
        )

        for bid, grp in grouped.groupby("business_id"):
            total = grp["count"].sum()
            self.popularity[bid] = {}
            for _, row in grp.iterrows():
                franja   = row["franja"]
                day_type = row["day_type"]
                if franja not in self.popularity[bid]:
                    self.popularity[bid][franja] = {}
                self.popularity[bid][franja][day_type] = row["count"] / total

        # Hora pico por negocio
        peak = checkins.groupby(["business_id", "hour"])["checkin_datetime"].count()
        for bid in peak.index.get_level_values("business_id").unique():
            sub = peak.loc[bid]
            self.hora_peak[bid] = int(sub.idxmax())

        logger.info("ContextAwareModel listo: %d negocios con perfil contextual",
                    len(self.popularity))
        return self

    @staticmethod
    def _hour_to_franja(hour: int) -> str:
        for nombre, (h_ini, h_fin) in cfg.CONTEXT_HOURS.items():
            if h_ini <= hour < h_fin:
                return nombre
        return "noche"

    def get_context_boost(self,
                          business_id: str,
                          franja: str,
                          day_type: str) -> float:
        """
        Factor de popularidad contextual normalizado [0, 1].
        Si no hay datos históricos para el negocio, retorna 0.5 (neutral).
        """
        biz_data = self.popularity.get(business_id, {})
        franja_data = biz_data.get(franja, {})
        boost = franja_data.get(day_type, None)
        if boost is None:
            # Fallback: media de todas las franjas disponibles
            vals = [v for fd in biz_data.values() for v in fd.values()]
            return float(np.mean(vals)) if vals else 0.5
        return float(boost)

    def adjust_scores(self,
                      scores: dict[str, float],
                      context: dict,
                      user_lat: Optional[float] = None,
                      user_lon: Optional[float] = None) -> dict[str, float]:
        """
        Ajusta los scores de los candidatos según el contexto de la petición.

        scores:   {business_id: score_base}
        context:  salida de utils.data_loader.extract_request_context()
        user_lat/lon: coordenadas del usuario (para penalización por distancia)

        Fórmula:
            score_ctx = score_base × (1 + λ_ctx × boost) × dist_penalty
        """
        franja   = context.get("franja", "tarde")
        day_type = context.get("day_type", "laborable")

        adjusted = {}
        for bid, base_score in scores.items():
            boost = self.get_context_boost(bid, franja, day_type)

            # Factor contextual: amplifica en ±50%
            ctx_factor = 0.5 + boost   # rango [0.5, 1.5]

            # Penalización por distancia
            dist_factor = 1.0
            if user_lat is not None and user_lon is not None:
                coords = self._businesses_lat_lon.get(bid)
                if coords:
                    dist_km = float(_haversine_km(
                        user_lat, user_lon,
                        np.array([coords[0]]),
                        np.array([coords[1]])
                    )[0])
                    # Sigmoid inversa: negocios muy lejanos se penalizan
                    # dist_factor → 1 cuando dist→0, → 0.1 cuando dist→MAX_KM
                    normalized = min(dist_km / cfg.MAX_DISTANCE_KM, 1.0)
                    dist_factor = max(1.0 - 0.9 * normalized, 0.1)

            adjusted[bid] = base_score * ctx_factor * dist_factor

        return adjusted

    def filter_by_distance(self,
                            candidate_ids: list[str],
                            user_lat: float,
                            user_lon: float,
                            max_km: float = cfg.MAX_DISTANCE_KM) -> list[str]:
        """
        Filtra candidatos que estén dentro del radio de distancia especificado.
        Útil como pre-filtro antes de calcular scores.
        """
        result = []
        for bid in candidate_ids:
            coords = self._businesses_lat_lon.get(bid)
            if coords is None:
                result.append(bid)   # sin coordenadas → incluir por defecto
                continue
            dist = float(_haversine_km(
                user_lat, user_lon,
                np.array([coords[0]]),
                np.array([coords[1]])
            )[0])
            if dist <= max_km:
                result.append(bid)
        return result

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("ContextAwareModel guardado en %s", path)

    @classmethod
    def load(cls, path: Path) -> "ContextAwareModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("ContextAwareModel cargado desde %s", path)
        return model