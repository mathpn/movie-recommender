"""
Insert all relevant data into Database.

Will be included in Docker compose. For now, initialize postgres Docker with:
    docker run --name postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=movies -p 5401:5432 -d postgres
"""
import asyncio
import math
from ast import literal_eval
from datetime import datetime
from functools import partial

import asyncpg
import pandas as pd

from app.db.postgres import (create_metadata_table, create_ratings_table,
                             create_users_table, insert_movie_metadata,
                             insert_movie_rating, insert_user)
from app.models import MovieMetadata, Rating

POSTGRES_URI = "postgresql://postgres:postgres@localhost:5401/movies"


def get_genres(genres):
    return [x["name"] for x in genres]


def get_director(x):
    for i in x:
        if i["job"] == "Director":
            return i["name"]


def get_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names if names else None


def is_integer(field):
    try:
        int(field)
        return True
    except Exception:
        return False


def get_tmdb_id(rating_id, links) -> int:
    values = links[links["movieId"] == rating_id]["tmdbId"].values
    if len(values) == 0:
        return -1
    elif math.isnan(values[0]):
        return -1
    return int(values[0])


async def main():
    movies_metadata = pd.read_csv("./data/movies_metadata.csv")
    movie_credits = pd.read_csv("./data/credits.csv")
    keywords = pd.read_csv("./data/keywords.csv")

    movies_metadata = movies_metadata[movies_metadata["id"].apply(is_integer)]
    movies_metadata = movies_metadata.astype({"id": "int64"})
    movies_metadata = movies_metadata.merge(movie_credits, on="id")
    movies_metadata = movies_metadata.merge(keywords, on="id")

    features = ["cast", "crew", "keywords", "genres"]
    for feature in features:
        movies_metadata[feature] = movies_metadata[feature].apply(literal_eval)

    movies_metadata["director"] = movies_metadata["crew"].apply(get_director)

    features = ["cast", "keywords", "genres"]
    for feature in features:
        movies_metadata[feature] = movies_metadata[feature].apply(get_list)

    pool = await asyncpg.create_pool(POSTGRES_URI)
    await create_metadata_table(pool)
    row_dicts = movies_metadata.to_dict(orient="records")
    tasks = []
    for row in row_dicts:
        if not isinstance(row["title"], str) and math.isnan(row["title"]):
            continue
        metadata = MovieMetadata(
            row["id"],
            row["title"],
            row["cast"],
            row["director"],
            row["keywords"],
            row["genres"],
            float(row["popularity"]),
            row["vote_average"],
            row["vote_count"],
        )
        tasks.append(asyncio.create_task(insert_movie_metadata(pool, metadata)))
    await asyncio.gather(*tasks)

    ratings = pd.read_csv("./data/ratings_small.csv")
    links = pd.read_csv("./data/links.csv")
    get_ids = partial(get_tmdb_id, links=links)
    ratings["tmdb_id"] = ratings["movieId"].apply(get_ids)
    await create_ratings_table(pool)
    tasks = []
    for row_dict in ratings.to_dict(orient="records"):
        if row_dict["tmdb_id"] < 0:
            continue
        rating = Rating(
            user_id=row_dict["userId"],
            movie_id=row_dict["tmdb_id"],
            rating=int(row_dict["rating"] * 10),
            timestamp=datetime.fromtimestamp(row_dict["timestamp"]),
        )
        tasks.append(asyncio.create_task(insert_movie_rating(pool, rating)))
    await asyncio.gather(*tasks)

    await create_users_table(pool)
    tasks = []
    max_user_id = ratings['userId'].max()
    for i in range(max_user_id):
        tasks.append(asyncio.create_task(insert_user(pool, f"data_{i + 1}")))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
