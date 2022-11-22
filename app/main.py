import os
from time import perf_counter

import asyncpg
from fastapi import FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from app.db.postgres import (get_all_movies_genres,
                             get_keyword_searcher_fields, get_user_id,
                             insert_user)
from app.logger import logger
from app.lookup import (GenreSearcher, KeywordSearcher, collaborative_search,
                        create_genre_searcher, create_keyword_searcher)
from app.ml.kmf import KMFInferece, create_kmf_inference
from app.utils import timed

# from fastapi_cprofile.profiler import CProfileMiddleware


app = FastAPI()
# app.add_middleware(CProfileMiddleware, enable=True, print_each_request=True, strip_dirs=False, sort_by='tottime')


@app.on_event("startup")
@timed
async def startup_event():
    app.state.pool = await asyncpg.create_pool(os.environ["POSTGRES_URI"])
    movies_genres = await get_all_movies_genres(app.state.pool)
    app.state.genre_searcher = create_genre_searcher(
        movies_genres["movie_ids"], movies_genres["genres"]
    )
    keyword_fields = await get_keyword_searcher_fields(app.state.pool)
    app.state.keyword_searcher = create_keyword_searcher(keyword_fields)
    app.state.kmf_inference = await create_kmf_inference(app.state.pool)


@app.post("/create_user")
@timed
async def create_user(request: Request, username: str) -> JSONResponse:
    try:
        await insert_user(request.app.state.pool, username)
        return JSONResponse(f"created user {username}")
    except asyncpg.UniqueViolationError:
        msg = f"user {username} already exists"
        logger.error(msg)
        return JSONResponse(msg, status_code=400)
    except Exception as exc:
        msg = "create user crashed"
        logger.error(msg + " " + str(exc))
        return JSONResponse(msg, status_code=500)


@app.get("/recommend")
@timed
async def recommend(
    request: Request,
    username: str,
    movie_id: int = Query(ge=0),
    k: int = Query(ge=1, le=50),
) -> JSONResponse:
    user_id = await get_user_id(app.state.pool, username)
    if user_id is None:
        return JSONResponse({"error": "username not found"}, 400)

    genre_searcher: GenreSearcher = request.app.state.genre_searcher
    keyword_searcher: KeywordSearcher = request.app.state.keyword_searcher
    recos_by_genre = genre_searcher.search_by_movie(movie_id=movie_id, k=2000)

    # keyword-based recommendations
    recos_by_kw, _ = keyword_searcher.search_by_movie(
        movie_id=movie_id, k=k, allowed_movie_ids=recos_by_genre
    )

    # user-based recommendations
    kmf_inference: KMFInferece = request.app.state.kmf_inference
    recos_by_user, _ = collaborative_search(
        kmf_inference, user_id, k=k, allowed_movies=recos_by_genre
    )
    recos_by_user = [reco for reco in recos_by_user if reco not in set(recos_by_kw)]

    n_user_recos = min(k // 2, len(recos_by_user)) + max(0, k // 2 - len(recos_by_kw))
    merged_recos = recos_by_kw[:(k - n_user_recos)] + recos_by_user[:n_user_recos]
    end = perf_counter()
    logger.info(f"found {len(merged_recos)} recommendations for {username}")
    return JSONResponse(merged_recos)


@app.post("/rate")
@timed
async def rate_movie(
    request: Request,
    username: str,
    movie_id: int = Query(ge=0),
    rating: float = Query(ge=0.0, le=5.0),
) -> JSONResponse:
    # TODO
    raise NotImplementedError()
