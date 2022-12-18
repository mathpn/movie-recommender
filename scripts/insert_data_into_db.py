"""
Insert all relevant data into Database.

If you're running outise of Docker, start a local Postgres instance:
    docker run --name postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=movies -p 5401:5432 -d postgres
"""
import asyncio
import math
import os
from ast import literal_eval
from datetime import datetime

import asyncpg
import pandas as pd

from app.db.postgres import (count_movies, create_global_table,
                             create_movies_table, create_ratings_primary_key,
                             create_ratings_table, create_users_table,
                             drop_ratings_primary_key, insert_movie_metadatas,
                             insert_movie_ratings, insert_users)
from app.logger import logger
from app.models import MovieMetadata, Rating

POSTGRES_URI = "postgresql://postgres:postgres@localhost:5401/movies"


def get_director(x):
    for i in x:
        if i["job"] == "Director":
            return i["name"]


def get_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names if names else []


def is_integer(field):
    try:
        int(field)
        return True
    except Exception:
        return False


def get_tmdb_id(raw_id) -> int:
    if math.isnan(raw_id):
        return -1
    return int(raw_id)


def process_metadata(row):
    if not isinstance(row["title"], str) and math.isnan(row["title"]):
        return None
    return MovieMetadata(
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


def process_rating(row):
    if row["tmdb_id"] < 0:
        return None
    return Rating(
        user_id=row["userId"],
        movie_id=row["tmdb_id"],
        rating=int(row["rating"] * 10),
        timestamp=datetime.fromtimestamp(row["timestamp"]),
    )


async def insert_data(
    pool: asyncpg.Pool, data_table: pd.DataFrame, processing_fn, callback
):
    batch = []
    tasks = set()
    for _, row in data_table.iterrows():
        row = processing_fn(row)
        if row is None:
            continue
        batch.append(row)
        if len(batch) >= 1024:
            task = asyncio.create_task(callback(pool, batch))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
            batch = []
        if len(tasks) > 10:
            await asyncio.gather(*tasks)
    if batch:
        tasks.add(asyncio.create_task(callback(pool, batch)))
    await asyncio.gather(*tasks)


async def main():
    logger.info("importing data to the database")
    pool = await asyncpg.create_pool(os.environ.get("POSTGRES_URI", POSTGRES_URI))

    count = await count_movies(pool)
    if count > 500 and not os.environ.get("FORCE_IMPORT"):
        logger.info("WARNING: database already has data, skipping data insertion")
        return

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

    await create_movies_table(pool)
    await create_global_table(pool)
    logger.info("inserting movie metadata")
    await insert_data(pool, movies_metadata, process_metadata, insert_movie_metadatas)

    ratings = pd.read_csv("./data/ratings_small.csv")
    links = pd.read_csv("./data/links.csv")
    logger.info("loaded ratings table")
    movie_ids = links["movieId"].tolist()
    tmdb_ids = [get_tmdb_id(x) for x in links["tmdbId"].tolist()]
    movie_2_tmdb = dict(zip(movie_ids, tmdb_ids))
    ratings["tmdb_id"] = ratings["movieId"].apply(
        lambda movie_id: movie_2_tmdb.get(movie_id, -1)
    )
    logger.info("got movie ids")
    ratings = ratings.drop(columns="movieId")
    ratings["timestamp"] = ratings["timestamp"].apply(
        lambda x: datetime.fromtimestamp(x)
    )
    ratings["rating"] = ratings["rating"].apply(lambda x: int(x * 10))
    ratings = ratings[ratings["tmdb_id"] != -1]
    await create_ratings_table(pool)
    ratings_tuples = (tuple(x) for x in ratings.values)
    logger.info("inserting ratings")
    await drop_ratings_primary_key(pool)
    async with pool.acquire() as conn:
        await conn.copy_records_to_table(
            "ratings",
            records=ratings_tuples,
            columns=["user_id", "rating", "rating_timestamp", "movie_id"],
        )
        await conn.execute(
            """
            DELETE FROM ratings a USING ratings b
            WHERE a.ctid < b.ctid AND (a.user_id, a.movie_id) = (b.user_id, b.movie_id)
        """
        )
    await create_ratings_primary_key(pool)
    logger.info("inserting users")
    await create_users_table(pool)
    batch = []
    max_user_id = ratings["userId"].max()
    for i in range(max_user_id):
        batch.append((f"data_{i + 1}",))
        if len(batch) >= 1024:
            await insert_users(pool, batch)
            batch = []
    if batch:
        await insert_users(pool, batch)
    logger.info("finished inserting the data")


if __name__ == "__main__":
    asyncio.run(main())
