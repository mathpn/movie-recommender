import json
import os
from collections import defaultdict
from datetime import datetime

import asyncpg
from fastapi import BackgroundTasks, FastAPI, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette import status
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import JSONResponse, RedirectResponse

from app.db.postgres import (get_all_movies_genres,
                             get_keyword_searcher_fields, get_movie_titles,
                             get_user_id, get_user_rate_count,
                             insert_movie_rating, insert_user,
                             is_movie_present, vector_bias_exist)
from app.logger import logger
from app.lookup import (GenreSearcher, KeywordSearcher, create_genre_searcher,
                        create_keyword_searcher, recommend)
from app.ml.background import (background_full_training,
                               background_online_training)
from app.ml.kmf import KMFInferece, create_kmf_inference, online_user_pipeline
from app.models import Rating
from app.search.fuzzy_search import get_searcher
from app.utils import timed

app = FastAPI()
app.mount("/static", StaticFiles(directory="./templates"), name="static")
app.add_middleware(SessionMiddleware, secret_key="foobar")

templates = Jinja2Templates(directory="templates")


class RateParams(BaseModel):
    movie_id: int = Query(ge=0)
    rating: float = Query(ge=0.0, le=5.0)


class UserParams(BaseModel):
    username: str


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
    if not await vector_bias_exist(app.state.pool):
        await background_full_training(app.state, verbose=True)
    app.state.kmf_inference = await create_kmf_inference(app.state.pool)
    app.state.pending_training = set()
    app.state.new_ratings = defaultdict(set)
    app.state.verbose_bg = bool(os.environ.get("VERBOSE")) or False
    app.state.online_threshold = os.environ.get("ONLINE_THRESHOLD") or 5
    app.state.global_change_count = [0]


@app.post("/get_user_id")
@timed
async def get_user_id(request: Request, username: str) -> JSONResponse:
    try:
        user_id = await insert_user(request.app.state.pool, username)
        logger.info(f"username {username} already exists, returning user_id")
        return JSONResponse({"username": username, "user_id": user_id})
    except Exception as exc:
        msg = "create user crashed"
        logger.error(msg + " " + str(exc))
        return JSONResponse(msg, status_code=500)


@app.get("/recommend_json")
@timed
async def recommend_json(
    request: Request,
    movie_id: int = Query(ge=0),
    k: int = Query(ge=1, le=50, default=6),
) -> JSONResponse:
    pool = request.app.state.pool
    user_id = request.session.get("user_id")
    if user_id is None:
        return RedirectResponse("/", status_code=status.HTTP_302_FOUND)

    genre_searcher: GenreSearcher = request.app.state.genre_searcher
    keyword_searcher: KeywordSearcher = request.app.state.keyword_searcher
    kmf_inference: KMFInferece = request.app.state.kmf_inference

    recos = await recommend(
        movie_id,
        user_id,
        genre_searcher,
        keyword_searcher,
        kmf_inference,
        k=k,
        pool=pool,
    )
    recos_titles = await get_movie_titles(pool, recos)
    return JSONResponse(recos_titles)


@app.get("/recommend_html")
@timed
async def recommend_html(
    request: Request,
    movie_id: int = Query(ge=0),
    k: int = Query(ge=1, le=50, default=6),
) -> JSONResponse:
    response = await recommend_json(request, movie_id, k)
    recos = json.loads(response.body)
    return templates.TemplateResponse(
        "recommendations.html", {"request": request, "movies": recos}
    )


@app.post("/rate_movie")
@timed
async def rate_movie(
    request: Request,
    bg_tasks: BackgroundTasks,
    body: RateParams,
) -> JSONResponse:
    user_id = request.session.get("user_id")
    if user_id is None:
        return RedirectResponse("/", status_code=status.HTTP_302_FOUND)

    pool = request.app.state.pool
    if not await is_movie_present(pool, body.movie_id):
        return JSONResponse(
            {"error": f"movie ID {body.movie_id} not found"}, status_code=400
        )

    rating_obj = Rating(
        user_id,
        movie_id=body.movie_id,
        rating=int(10 * body.rating),
        timestamp=datetime.now(),
    )
    new_ratings = request.app.state.new_ratings
    try:
        await insert_movie_rating(pool, rating_obj)
        new_ratings[user_id].add(body.movie_id)
        logger.info(
            f"inserted new rating of user {user_id} - movie {body.movie_id} - new ratings = {len(new_ratings[user_id])}"
        )
    except Exception as exc:
        logger.error(f"failed to insert rating {rating_obj}: {exc}")
        return JSONResponse({"error": "unkown internal exception"}, status_code=500)

    state = request.app.state
    if (
        len(new_ratings[user_id]) >= state.online_threshold
        and user_id not in state.pending_training
    ):
        logger.info(f"starting online training for user ID {user_id}")
        verbose_bg = request.app.state.verbose_bg
        bg_tasks.add_task(
            background_online_training,
            state,
            pool,
            user_id,
            verbose=verbose_bg,
            min_count=state.online_threshold,
        )

    return JSONResponse({"status": "ok"}, status_code=200)


@app.get("/")
async def landing(request: Request):
    if request.session.get("user_id") is not None:
        return RedirectResponse("/home")
    return templates.TemplateResponse("username.html", {"request": request})


@app.post("/login")
async def login(request: Request, body: UserParams):
    user = await get_user_id(request, body.username)
    try:
        user = json.loads(user.body)
        request.session["username"] = user["username"]
        request.session["user_id"] = user["user_id"]
        logger.info(f"storing username {body.username} in session")
    except Exception as exc:
        logger.error(f"setting user data failed: {exc}")
        return RedirectResponse(
            "/", status_code=status.HTTP_302_FOUND
        )  # changing from POST to GET

    return RedirectResponse("/home", status_code=status.HTTP_302_FOUND)


@app.get("/home")
async def home(request: Request):
    if request.session.get("user_id") is None:
        return RedirectResponse("/")
    return templates.TemplateResponse("search.html", {"request": request})


@app.get("/search")
async def search(request: Request, query: str, limit: int = 5):
    if request.session.get("user_id") is None:
        return RedirectResponse("/")
    searcher = await get_searcher(request.app.state.pool)
    movies = searcher(query, limit=5)
    return templates.TemplateResponse(
        "search_results.html", {"request": request, "movies": movies}
    )


@app.get("/user_rate_count")
async def user_rate_count(request: Request):
    user_id = request.session.get("user_id")
    if user_id is None:
        return JSONResponse("not logged in")

    pool = request.app.state.pool
    rate_count = await get_user_rate_count(pool, user_id)
    return JSONResponse(rate_count)
