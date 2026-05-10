"""
api/main.py
===========
API REST con FastAPI para el sistema de recomendación híbrido de Yelp.

Endpoints:
  GET  /api/v1/health                   → estado del servicio
  POST /api/v1/recommend                → recomendaciones para un usuario
  GET  /api/v1/similar/{business_id}    → negocios similares (CB)
  GET  /api/v1/business/{business_id}   → detalle de un negocio
  GET  /api/v1/users/{user_id}/history  → historial del usuario
  POST /api/v1/evaluate                 → métricas offline (uso interno)

Cómo conectar a una interfaz:
  - Cualquier frontend (React, Vue, Flutter, etc.) puede consumir
    estos endpoints directamente vía fetch/axios/http.
  - CORS está habilitado para * en desarrollo; restringirlo en producción.
  - Documentación interactiva: http://localhost:8000/docs (Swagger UI)
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from models.hybrid_recommender import HybridRecommender

logger = logging.getLogger("api")

# ─────────────────────────────────────────────────────────────────────────────
# Estado global de la aplicación
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    recommender: Optional[HybridRecommender] = None
    models_loaded: bool = False
    load_error: Optional[str] = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga los modelos al iniciar la aplicación."""
    logger.info("Cargando modelos desde %s ...", cfg.MODELS_DIR)
    try:
        state.recommender  = HybridRecommender.load(cfg.MODELS_DIR)
        state.models_loaded = True
        logger.info("Modelos cargados correctamente.")
    except FileNotFoundError:
        state.load_error = (
            "Modelos no encontrados. Ejecuta train.py primero."
        )
        logger.warning(state.load_error)
    except Exception as e:
        state.load_error = str(e)
        logger.error("Error cargando modelos: %s", e)
    yield
    # Limpieza al apagar (opcional)
    state.recommender = None


# ─────────────────────────────────────────────────────────────────────────────
# Aplicación FastAPI
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Yelp Hybrid Recommender API",
    description=(
        "Sistema de recomendación híbrido para negocios de Yelp. "
        "Combina filtrado colaborativo (SVD/ALS), recomendación sensible "
        "al contexto y basada en contenido."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # ← restringir en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Schemas Pydantic (contratos de entrada/salida para el frontend)
# ─────────────────────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    user_id: str = Field(..., description="ID del usuario de Yelp")
    top_n: int = Field(
        default=10, ge=1, le=50,
        description="Número de recomendaciones a devolver"
    )
    # Contexto espacio-temporal (opcionales)
    latitude: Optional[float]  = Field(None, description="Latitud del usuario (GPS)")
    longitude: Optional[float] = Field(None, description="Longitud del usuario (GPS)")
    request_datetime: Optional[datetime] = Field(
        None, description="Momento de la petición (ISO 8601). Default: ahora"
    )
    # Filtros opcionales
    city: Optional[str]            = Field(None, description="Filtrar por ciudad")
    categories: Optional[list[str]] = Field(None, description="Filtrar por categorías")
    exclude_seen: bool             = Field(True, description="Excluir negocios ya visitados")


class BusinessSummary(BaseModel):
    rank: int
    business_id: str
    name: str
    city: str
    state: str
    stars: Optional[float]
    review_count: Optional[int]
    categories: list[str]
    hybrid_score: float
    cf_score: float
    ctx_score: float
    cb_score: float
    distance_km: Optional[float] = None


class RecommendResponse(BaseModel):
    user_id: str
    context: dict
    recommendations: list[BusinessSummary]
    generated_at: datetime


class SimilarBusinessResponse(BaseModel):
    business_id: str
    similar: list[dict]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    error: Optional[str] = None


class EvaluateRequest(BaseModel):
    k: int = Field(default=10, ge=1, le=50)
    sample_users: int = Field(default=200, ge=10, le=2000)
    rating_threshold: float = Field(default=4.0, ge=1.0, le=5.0)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    f"{cfg.API_PREFIX}/health",
    response_model=HealthResponse,
    summary="Estado del servicio",
    tags=["Sistema"],
)
async def health():
    """Verifica si los modelos están cargados y el servicio está operativo."""
    return HealthResponse(
        status="ok" if state.models_loaded else "degraded",
        models_loaded=state.models_loaded,
        error=state.load_error,
    )


@app.post(
    f"{cfg.API_PREFIX}/recommend",
    response_model=RecommendResponse,
    summary="Obtener recomendaciones para un usuario",
    tags=["Recomendación"],
)
async def recommend(body: RecommendRequest):
    """
    Genera las top-N recomendaciones de negocios para un usuario.

    - Incluye contexto temporal automático (hora, día de la semana).
    - Si se proporciona latitud/longitud, aplica filtro de distancia y
      penalización geoespacial.
    - Los scores devueltos son: híbrido (combinado), CF, contextual y contenido.
    """
    if not state.models_loaded:
        raise HTTPException(
            status_code=503,
            detail=state.load_error or "Modelos no disponibles."
        )

    from utils.data_loader import extract_request_context
    context = extract_request_context(body.request_datetime)

    try:
        recs = state.recommender.recommend(
            user_id          = body.user_id,
            top_n            = body.top_n,
            user_lat         = body.latitude,
            user_lon         = body.longitude,
            request_datetime = body.request_datetime,
            city_filter      = body.city,
            category_filter  = body.categories,
            exclude_seen     = body.exclude_seen,
        )
    except Exception as e:
        logger.error("Error generando recomendaciones para '%s': %s", body.user_id, e)
        raise HTTPException(status_code=500, detail=str(e))

    return RecommendResponse(
        user_id        = body.user_id,
        context        = context,
        recommendations= [BusinessSummary(**r) for r in recs],
        generated_at   = datetime.now(),
    )


@app.get(
    f"{cfg.API_PREFIX}/similar/{{business_id}}",
    response_model=SimilarBusinessResponse,
    summary="Negocios similares (basado en contenido)",
    tags=["Recomendación"],
)
async def similar_businesses(
    business_id: str,
    n: int = Query(default=10, ge=1, le=50, description="Número de similares"),
):
    """
    Devuelve los N negocios más similares al negocio dado,
    basado en similitud de contenido (categorías + texto de reseñas).
    Útil para el widget "Te puede interesar también..." en el frontend.
    """
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no disponibles.")

    cb = state.recommender.cb_model
    similar = cb.get_similar(business_id, n=n)
    if not similar:
        raise HTTPException(status_code=404, detail=f"Negocio '{business_id}' no encontrado.")

    # Enriquecer con metadata
    biz_meta = {}
    if state.recommender.businesses_df is not None:
        biz_meta = state.recommender.businesses_df.set_index("business_id").to_dict("index")

    result = []
    for bid, score in similar:
        meta = biz_meta.get(bid, {})
        result.append({
            "business_id": bid,
            "name":        meta.get("name", ""),
            "city":        meta.get("city", ""),
            "stars":       meta.get("stars"),
            "categories":  meta.get("categories", []),
            "similarity":  round(score, 4),
        })

    return SimilarBusinessResponse(business_id=business_id, similar=result)


@app.get(
    f"{cfg.API_PREFIX}/business/{{business_id}}",
    summary="Detalle de un negocio",
    tags=["Negocios"],
)
async def get_business(business_id: str):
    """Devuelve los metadatos de un negocio por su ID."""
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no disponibles.")

    if state.recommender.businesses_df is None:
        raise HTTPException(status_code=503, detail="Datos de negocios no disponibles.")

    df = state.recommender.businesses_df
    row = df[df["business_id"] == business_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Negocio '{business_id}' no encontrado.")

    return row.iloc[0].to_dict()


@app.get(
    f"{cfg.API_PREFIX}/users/{{user_id}}/history",
    summary="Historial de un usuario",
    tags=["Usuarios"],
)
async def get_user_history(
    user_id: str,
    limit: int = Query(default=20, ge=1, le=100),
):
    """Devuelve los negocios visitados por el usuario (registrados en entrenamiento)."""
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no disponibles.")

    history_ids = state.recommender._user_history.get(user_id, [])
    if not history_ids:
        raise HTTPException(status_code=404,
                            detail=f"Usuario '{user_id}' no encontrado o sin historial.")

    history_ids = history_ids[:limit]

    biz_meta = {}
    if state.recommender.businesses_df is not None:
        biz_meta = state.recommender.businesses_df.set_index("business_id").to_dict("index")

    result = []
    for bid in history_ids:
        meta = biz_meta.get(bid, {})
        result.append({
            "business_id": bid,
            "name":        meta.get("name", ""),
            "city":        meta.get("city", ""),
            "stars":       meta.get("stars"),
            "categories":  meta.get("categories", []),
        })

    return {"user_id": user_id, "count": len(result), "history": result}


@app.post(
    f"{cfg.API_PREFIX}/evaluate",
    summary="Evaluar el modelo offline",
    tags=["Sistema"],
)
async def evaluate(body: EvaluateRequest):
    """
    Ejecuta la evaluación offline del modelo sobre los datos de test.
    Solo para uso interno / monitoreo del sistema.
    """
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no disponibles.")

    # Requiere acceso a test_df; en producción se cargaría desde disco
    raise HTTPException(
        status_code=501,
        detail=(
            "En producción, los datos de test se cargan desde disco. "
            "Usa train.py para la evaluación completa."
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=cfg.API_HOST,
        port=cfg.API_PORT,
        reload=True,   # recarga automática en desarrollo
        log_level="info",
    )