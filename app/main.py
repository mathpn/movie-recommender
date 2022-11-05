import os
from time import perf_counter

import asyncpg
from fastapi import FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from app.db.postgres import (get_all_movies_genres,
                             get_keyword_searcher_fields, insert_user)
from app.logger import logger
from app.lookup import (GenreSearcher, KeywordSearcher, create_genre_searcher,
                        create_keyword_searcher)

# from fastapi_cprofile.profiler import CProfileMiddleware


app = FastAPI()
# app.add_middleware(CProfileMiddleware, enable=True, print_each_request=True, strip_dirs=False, sort_by='tottime')


@app.on_event("startup")
async def startup_event():
    app.state.pool = await asyncpg.create_pool(os.environ["POSTGRES_URI"])
    movies_genres = await get_all_movies_genres(app.state.pool)
    app.state.genre_searcher = create_genre_searcher(
        movies_genres["movie_ids"], movies_genres["genres"]
    )
    keyword_fields = await get_keyword_searcher_fields(app.state.pool)
    app.state.keyword_searcher = create_keyword_searcher(keyword_fields)


@app.post("/create_user")
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
async def recommend(request: Request, username: str, movie_id: int) -> JSONResponse:
    init = perf_counter()
    genre_searcher: GenreSearcher = request.app.state.genre_searcher
    keyword_searcher: KeywordSearcher = request.app.state.keyword_searcher
    recos_by_genre = genre_searcher.search_by_movie(movie_id=movie_id, k=2000)
    logger.info(len(recos_by_genre))
    recos_by_kw, _ = keyword_searcher.search_by_movie(
        movie_id=movie_id, k=5, allowed_movie_ids=recos_by_genre
    )
    end = perf_counter()
    logger.info(f"{(end - init) * 1000:.2f} ms")
    return JSONResponse(recos_by_kw)


@app.post("/rate")
async def rate_movie(
    request: Request, username: str, movie_id: int, rating: float = Query(ge=0.0, le=5.0)
) -> JSONResponse:
    # TODO
    raise NotImplementedError()
