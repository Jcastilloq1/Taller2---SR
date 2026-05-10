"""
config.py
=========
Configuración central del sistema de recomendación híbrido Yelp.
Todos los hiperparámetros y rutas se definen aquí para facilitar
la sintonización sin tocar el código de los modelos.
"""

from pathlib import Path

# ── Rutas ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)

# Archivos del dataset Yelp (colocar los JSON aquí)
BUSINESS_FILE = DATA_DIR / "yelp_academic_dataset_business.json"
REVIEW_FILE   = DATA_DIR / "yelp_academic_dataset_review.json"
USER_FILE     = DATA_DIR / "yelp_academic_dataset_user.json"
CHECKIN_FILE  = DATA_DIR / "yelp_academic_dataset_checkin.json"
TIP_FILE      = DATA_DIR / "yelp_academic_dataset_tip.json"

# ── Filtros de datos ───────────────────────────────────────────────────────
MIN_USER_REVIEWS     = 5    # usuarios con al menos N reseñas
MIN_BUSINESS_REVIEWS = 5    # negocios con al menos N reseñas
MAX_REVIEWS_LOAD     = None # None = cargar todo; int = subconjunto para dev

# ── Split temporal ─────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20          # últimas fechas → sin data leakage

# ── Filtrado colaborativo por factorización (SVD via Surprise) ─────────────
SVD_N_FACTORS   = 100
SVD_N_EPOCHS    = 20
SVD_LR_ALL      = 0.005
SVD_REG_ALL     = 0.02
SVD_BIASED      = True

# ── ALS implícito (implicit library) ──────────────────────────────────────
ALS_FACTORS     = 64
ALS_REGULARIZATION = 0.01
ALS_ITERATIONS  = 15
ALS_ALPHA       = 40        # escala de confianza para feedback implícito

# ── Modelo de contenido (TF-IDF + Sentence-BERT) ─────────────────────────
TFIDF_MAX_FEATURES  = 10_000
TFIDF_NGRAM_RANGE   = (1, 2)
SBERT_MODEL_NAME    = "all-MiniLM-L6-v2"   # 384-dim, rápido y ligero
USE_SBERT           = False  # True activa Sentence-BERT (requiere más RAM)

# ── Modelo sensible al contexto ────────────────────────────────────────────
MAX_DISTANCE_KM     = 10.0  # radio geoespacial máximo para filtrar candidatos
CONTEXT_HOURS = {           # franjas horarias para discretización
    "madrugada": (0, 6),
    "mañana":    (6, 12),
    "tarde":     (12, 18),
    "noche":     (18, 24),
}
CONTEXT_DAYS = {            # tipo de día
    "laborable": range(0, 5),   # lunes-viernes
    "fin_semana": range(5, 7),  # sábado-domingo
}

# ── Combinador híbrido (pesos α + β + γ = 1) ─────────────────────────────
HYBRID_WEIGHT_CF  = 0.50    # α  – filtrado colaborativo
HYBRID_WEIGHT_CTX = 0.30    # β  – sensible al contexto
HYBRID_WEIGHT_CB  = 0.20    # γ  – contenido

# ── Re-ranking por diversidad (MMR) ──────────────────────────────────────
MMR_LAMBDA          = 0.7   # 0=máx diversidad, 1=máx relevancia
TOP_N_CANDIDATES    = 100   # candidatos antes del re-ranking
TOP_N_FINAL         = 10    # recomendaciones finales a devolver

# ── API ────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
API_PREFIX = "/api/v1"