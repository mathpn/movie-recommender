import os
from collections import defaultdict
from datetime import datetime

import asyncpg
from fastapi import BackgroundTasks, FastAPI, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from app.db.postgres import (get_all_movies_genres,
                             get_keyword_searcher_fields, get_user_id,
                             insert_movie_rating, insert_user,
                             is_movie_present)
from app.logger import logger
from app.lookup import (GenreSearcher, KeywordSearcher, collaborative_search,
                        create_genre_searcher, create_keyword_searcher)
from app.ml.kmf import KMFInferece, create_kmf_inference, online_user_pipeline
from app.models import Rating
from app.utils import timed

# from fastapi_cprofile.profiler import CProfileMiddleware


app = FastAPI()
app.mount("/static", StaticFiles(directory="./templates"), name="static")

templates = Jinja2Templates(directory="templates")
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
    app.state.new_ratings = defaultdict(int)
    app.state.verbose_bg = bool(os.environ.get("VERBOSE")) or False
    app.state.online_threshold = os.environ.get("ONLINE_THRESHOLD") or 5

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


# TODO somehow store username in session
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
    logger.info(f"found {len(merged_recos)} recommendations for {username}")
    return JSONResponse(merged_recos)


# TODO somehow store username in session
@app.post("/rate")
@timed
async def rate_movie(
    request: Request,
    bg_tasks: BackgroundTasks,
    username: str,
    movie_id: int = Query(ge=0),
    rating: float = Query(ge=0.0, le=5.0),
) -> JSONResponse:
    pool = request.app.state.pool
    if not await is_movie_present(pool, movie_id):
        return JSONResponse({"error": f"movie ID {movie_id} not found"}, status_code=400)

    user_id = await get_user_id(app.state.pool, username)
    if user_id is None:
        return JSONResponse({"error": "username not found"}, 400)

    rating_obj = Rating(
        user_id,
        movie_id=movie_id,
        rating=int(10 * rating),
        timestamp=datetime.now()
    )
    change_count = request.app.state.new_ratings
    try:
        await insert_movie_rating(pool, rating_obj)
        change_count[user_id] += 1
        logger.info(f"inserted new rating of user {user_id} - change count = {change_count[user_id]}")
    except Exception as exc:
        logger.error(f"failed to insert rating {rating_obj}: {exc}")
        return JSONResponse({"error": "unkown internal exception"}, status_code=500)

    if change_count[user_id] >= request.app.state.online_threshold:
        logger.info(f"starting online training for user ID {user_id}")
        verbose_bg = request.app.state.verbose_bg
        change_count.pop(user_id, None)
        bg_tasks.add_task(online_user_pipeline, pool, user_id, verbose=verbose_bg)

    return JSONResponse({"status": "ok"}, status_code=200)


@app.get("/")
async def landing(request: Request):
    return RedirectResponse("/home")


@app.get("/home")
async def home(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})


@app.get("/search")
async def search(request: Request, movie_name: str):
    # TODO finish endpoint
    logger.info(movie_name)
    movie_names = ["foo", "bar", "ping", "pong"]
    movies = list(enumerate(movie_names))
    return templates.TemplateResponse("search_results.html", {"request": request, "movies": movies})


# XXX remove
@app.post("/fake_rate")
async def fake_rate(request: Request, rating):
    logger.info(f"rating {rating}")


# XXX remove
@app.get("/fake_reco")
async def fake_reco(request: Request, movie_id: int):
    logger.info(f"recommending for {movie_id}")
