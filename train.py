"""
train.py
========
Script de entrenamiento del sistema híbrido de recomendación.

Uso:
    python train.py
    python train.py --max-reviews 100000    # subconjunto para desarrollo

Pasos:
  1. Carga de datos (Yelp JSON)
  2. Filtrado k-core
  3. Split temporal
  4. Construcción de matrices
  5. Entrenamiento del HybridRecommender
  6. Evaluación offline (val set)
  7. Guardado de modelos
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train")

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from utils.data_loader import (
    load_businesses, load_reviews, load_users,
    load_checkins, load_tips,
    filter_interactions, build_id_maps,
    build_rating_matrix, build_implicit_matrix,
    temporal_split,
)
from models.hybrid_recommender import HybridRecommender
from utils.evaluation import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenar sistema de recomendación Yelp")
    parser.add_argument("--max-reviews", type=int, default=None,
                        help="Máx. reseñas a cargar (útil para desarrollo rápido)")
    parser.add_argument("--save-dir", type=str, default=str(cfg.MODELS_DIR),
                        help="Directorio donde guardar los modelos entrenados")
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)

    # ── 1. Carga de datos ────────────────────────────────────────────────────
    logger.info("=== Cargando datos de Yelp ===")

    if args.max_reviews:
        cfg.MAX_REVIEWS_LOAD = args.max_reviews
        logger.info("Modo desarrollo: cargando máximo %d reseñas", args.max_reviews)

    businesses = load_businesses()
    reviews    = load_reviews()
    users      = load_users()
    checkins   = load_checkins()
    tips       = load_tips()

    logger.info("Negocios: %d | Reseñas: %d | Usuarios: %d | Tips: %d",
                len(businesses), len(reviews), len(users), len(tips))

    # ── 2. Filtrado k-core ───────────────────────────────────────────────────
    logger.info("=== Filtrado k-core ===")
    reviews_filtered = filter_interactions(reviews)

    # ── 3. Split temporal ────────────────────────────────────────────────────
    logger.info("=== Split temporal ===")
    train_df, val_df, test_df = temporal_split(reviews_filtered)

    # ── 4. Mapeos y matrices ─────────────────────────────────────────────────
    logger.info("=== Construyendo matrices ===")
    user2idx, idx2user, biz2idx, idx2biz = build_id_maps(train_df)

    implicit_matrix = build_implicit_matrix(train_df, tips, user2idx, biz2idx)
    logger.info("Matriz implícita: %s, densidad=%.4f%%",
                implicit_matrix.shape,
                100 * implicit_matrix.nnz / (implicit_matrix.shape[0] * implicit_matrix.shape[1]))

    # Filtrar businesses a solo los que están en el train
    biz_ids_train  = set(biz2idx.keys())
    businesses_use = businesses[businesses["business_id"].isin(biz_ids_train)].reset_index(drop=True)
    tips_use       = tips[tips["business_id"].isin(biz_ids_train)]

    # ── 5. Entrenamiento ─────────────────────────────────────────────────────
    logger.info("=== Entrenando HybridRecommender ===")
    recommender = HybridRecommender()
    recommender.fit(
        train_df       = train_df,
        implicit_matrix= implicit_matrix,
        businesses_df  = businesses_use,
        checkins_df    = checkins,
        reviews_df     = reviews_filtered,
        tips_df        = tips_use,
        user2idx       = user2idx,
        biz2idx        = biz2idx,
    )
    recommender.businesses_df = businesses_use

    # ── 6. Evaluación offline ────────────────────────────────────────────────
    logger.info("=== Evaluación en validación ===")
    val_metrics = evaluate_model(recommender, val_df, k=10, sample_users=200)
    logger.info("Métricas@10 en val:")
    for metric, value in val_metrics.items():
        logger.info("  %-25s %s", metric, value)

    logger.info("=== Evaluación en test ===")
    test_metrics = evaluate_model(recommender, test_df, k=10, sample_users=500)
    logger.info("Métricas@10 en test:")
    for metric, value in test_metrics.items():
        logger.info("  %-25s %s", metric, value)

    # ── 7. Guardar modelos ───────────────────────────────────────────────────
    logger.info("=== Guardando modelos en %s ===", save_dir)
    recommender.save(save_dir)
    logger.info("Entrenamiento completo.")


if __name__ == "__main__":
    main()