"""
Insert all relevant data into Database.

Will be included in Docker compose. For now, initialize postgres Docker with:
    docker run --name postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=movies -p 5401:5432 -d postgres
"""
import asyncio
import json
from ast import literal_eval
import math

import pandas as pd
import numpy as np
import asyncpg

from app.db.postgres import insert_movie_metadata
from app.models import MovieMetadata

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
    row_dicts = movies_metadata.to_dict(orient="records")
    tasks = []
    for row in row_dicts:
        if not isinstance(row['title'], str) and math.isnan(row['title']):
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


if __name__ == "__main__":
    asyncio.run(main())
