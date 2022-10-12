import os

import asyncpg
from fastapi import FastAPI, Query, Request
from starlette.responses import JSONResponse

from app.db.postgres import insert_user, get_all_movies_genres
from app.lookup import create_genre_searcher

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    app.state.pool = await asyncpg.create_pool(os.environ["POSTGRES_URI"])
    movie_ids, genres = await get_all_movies_genres(app.state.pool)
    app.state.genre_searcher = create_genre_searcher(movie_ids, genres)



@app.post("/create_user")
async def create_user(request: Request, username: str) -> JSONResponse:
    try:
        await insert_user(request.app.state.pool, username)
        return JSONResponse(f"created user {username}")
    except asyncpg.UniqueViolationError:
        # TODO insert logging
        return JSONResponse(f"user {username} already exists", status_code=400)
    except Exception as exc:
        # TODO logging
        return JSONResponse("create user crashed", status_code=500)


@app.get("/recommend")
async def recommend(request: Request, username: str, movie_id: int) -> JSONResponse:
    # TODO
    raise NotImplementedError()


@app.post("/rate")
async def rate_movie(
    request: Request, username: str, movie_id: int, rating: float = Query(ge=0.0, le=5.0)
) -> JSONResponse:
    # TODO
    raise NotImplementedError()
