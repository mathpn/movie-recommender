import asyncpg

from app.logger import logger
from app.ml.kmf import (create_kmf_inference, online_user_pipeline,
                        run_train_pipeline, train_kmf_model)


async def background_online_training(
    state, pool: asyncpg.Pool, user_id: int, verbose: bool = False, min_count: int = 10
) -> None:
    state.pending_training.add(user_id)
    trained = await online_user_pipeline(pool, user_id, verbose, min_count)
    if trained:
        state.kmf_inference = await create_kmf_inference(state.pool)
        state.global_change_count[0] += state.new_ratings.pop(user_id, 0)
    state.pending_training.discard(user_id)


async def background_full_training(state, verbose: bool = False) -> None:
    logger.info("training a full KMF model, this is a blocking task")
    await run_train_pipeline(state.pool, verbose=verbose)
    state.global_change_count = [0]
