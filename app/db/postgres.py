"""
Functions to interact with PostgreSQL database.
"""

from typing import Any, Optional

import asyncpg

from app.models import KeywordFields, MovieMetadata, Rating, VectorBias


async def create_movies_table(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as connection:
        await connection.execute(open("./sql/create_movies.sql", "r").read())


def _movie_metadata_to_tuple(metadata: MovieMetadata):
    return (
        metadata.movie_id,
        metadata.movie_title,
        metadata.movie_cast,
        metadata.director,
        metadata.keywords,
        metadata.genres,
        metadata.popularity,
        metadata.vote_average,
        metadata.vote_count,
    )


async def insert_movie_metadata(pool: asyncpg.Pool, metadata: MovieMetadata) -> None:
    async with pool.acquire() as connection:
        row = _movie_metadata_to_tuple(metadata)
        await connection.execute(
            """
            INSERT INTO movies (
                movie_id, movie_title, movie_cast,
                director, keywords, genres, popularity,
                vote_average, vote_count)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT DO NOTHING
        """,
            *row,
        )


async def insert_movie_metadatas(pool: asyncpg.Pool, metadatas: list[MovieMetadata]) -> None:
    async with pool.acquire() as connection:
        row_gen = (_movie_metadata_to_tuple(metadata) for metadata in metadatas)
        await connection.executemany(
            """
            INSERT INTO movies (
                movie_id, movie_title, movie_cast,
                director, keywords, genres, popularity,
                vote_average, vote_count)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT DO NOTHING
        """,
            row_gen,
        )


async def get_movie_metadata(pool: asyncpg.Pool, movie_id: int) -> MovieMetadata:
    async with pool.acquire() as connection:
        result = await connection.fetchrow(
            """
                SELECT 
                    movie_id, movie_title, movie_cast,
                    director, keywords, genres, popularity,
                    vote_average, vote_count
                FROM movies WHERE movie_id = $1
            """,
            movie_id,
        )
    return MovieMetadata(**result)


async def get_all_movies_genres(pool: asyncpg.Pool) -> dict[str, Any]:
    async with pool.acquire() as connection:
        rows = await connection.fetch("SELECT movie_id, genres FROM movies ORDER BY movie_id")
    movie_ids = [row["movie_id"] for row in rows]
    genres = [row["genres"] for row in rows]
    return {"movie_ids": movie_ids, "genres": genres}


async def get_keyword_searcher_fields(pool: asyncpg.Pool) -> list[KeywordFields]:
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            "SELECT movie_id, genres, keywords, movie_cast, director FROM movies ORDER BY movie_id"
        )
    return [
        KeywordFields(
            movie_id=row["movie_id"],
            genres=row["genres"],
            keywords=row["keywords"],
            cast=row["movie_cast"],
            director=row["director"],
        )
        for row in rows
    ]


async def write_movie_vector_bias(pool: asyncpg.Pool, vector_biases: VectorBias, movie_id: int):
    async with pool.acquire() as connection:
        await connection.execute(
            "UPDATE movies SET vector = $1, bias = $2 WHERE movie_id = $3",
            vector_biases.vector,
            vector_biases.bias,
            movie_id,
        )


async def write_bulk_movie_vector_bias(pool: asyncpg.Pool, chunk: list[tuple[VectorBias, int]]):
    async with pool.acquire() as connection:
        row_generator = ((x[0].vector, x[0].bias, x[1]) for x in chunk)
        await connection.executemany(
            "UPDATE movies SET vector = $1, bias = $2 WHERE movie_id = $3",
            row_generator
        )


async def get_movie_vector_bias(pool: asyncpg.Pool, movie_id: int) -> Optional[VectorBias]:
    async with pool.acquire() as connection:
        row = await connection.fetchrow(
            "SELECT vector, bias FROM movies WHERE user_id = $1", movie_id
        )
    vector = row.get("vector")
    bias = row.get("bias")
    if vector is None or bias is None:
        return None
    return VectorBias(vector=vector, bias=bias)


async def delete_all_movie_vector_bias(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as connection:
        await connection.execute("UPDATE movies SET vector = NULL, bias = NULL")


async def create_ratings_table(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as connection:
        await connection.execute(open("./sql/create_ratings.sql", "r").read())


def _rating_to_tuple(rating: Rating) -> tuple:
    return (rating.user_id, rating.movie_id, rating.rating, rating.timestamp)


async def insert_movie_rating(pool: asyncpg.Pool, rating: Rating) -> None:
    async with pool.acquire() as connection:
        row = _rating_to_tuple(rating)
        await connection.execute(
            """
            INSERT INTO ratings
            (user_id, movie_id, rating, rating_timestamp)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (user_id, movie_id) DO UPDATE
            SET rating = $3, rating_timestamp = $4
        """,
            *row,
        )


async def drop_ratings_primary_key(pool: asyncpg.Pool):
    """Drop primary key. Used for large inserts on startup."""
    async with pool.acquire() as connection:
        await connection.execute("ALTER TABLE ratings DROP CONSTRAINT ratings_pkey")


async def create_ratings_primary_key(pool: asyncpg.Pool):
    """Recreate primary key. Used for large inserts on startup."""
    async with pool.acquire() as connection:
        await connection.execute(
            "ALTER TABLE ratings ADD CONSTRAINT ratings_pkey PRIMARY KEY (user_id, movie_id)"
        )


async def insert_movie_ratings(pool: asyncpg.Pool, ratings: list[Rating]) -> None:
    """
    Bulk insert of movie ratings.
    """
    async with pool.acquire() as connection:
        ratings_gen = (_rating_to_tuple(rating) for rating in ratings)
        await connection.executemany(
            """
            INSERT INTO ratings
            (user_id, movie_id, rating, rating_timestamp)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT DO NOTHING
        """,
            ratings_gen,
        )


async def get_user_movie_rating(pool: asyncpg.Pool, user_id: int, movie_id: int) -> Optional[int]:
    async with pool.acquire() as connection:
        return await connection.fetchval(
            "SELECT rating FROM ratings WHERE user_id = $1 AND movie_id = $2", user_id, movie_id
        )


async def get_all_ratings(pool: asyncpg.Pool):
    async with pool.acquire() as connection:
        async with connection.transaction():
            async for row in connection.cursor(
                "SELECT user_id, movie_id, rating FROM ratings ORDER BY user_id"
            ):
                yield dict(row)


async def sample_ratings(pool: asyncpg.Pool, prob: float = 0.1, limit: Optional[int] = None):
    async with pool.acquire() as connection:
        async with connection.transaction():
            query = f"SELECT user_id, movie_id, rating FROM ratings WHERE random() < $1"
            if limit:
                query += f" LIMIT {limit}"
            async for row in connection.cursor(query, prob):
                yield dict(row)


async def create_users_table(pool: asyncpg.Pool):
    async with pool.acquire() as connection:
        await connection.execute(open("./sql/create_users.sql", "r").read())


async def insert_user(pool: asyncpg.Pool, username: str) -> None:
    async with pool.acquire() as connection:
        await connection.execute("INSERT INTO users (username) VALUES ($1)", username)


async def insert_users(pool: asyncpg.Pool, usernames: list[str]) -> None:
    async with pool.acquire() as connection:
        await connection.executemany("INSERT INTO users (username) VALUES ($1)", usernames)


async def get_user_id(pool: asyncpg.Pool, username: str) -> Optional[int]:
    async with pool.acquire() as connection:
        return await connection.fetchval("SELECT user_id FROM users WHERE username = $1", username)


async def write_user_vector_bias(pool: asyncpg.Pool, vector_biases: VectorBias, user_id: int):
    async with pool.acquire() as connection:
        await connection.execute(
            "UPDATE users SET vector = $1, bias = $2 WHERE user_id = $3",
            vector_biases.vector,
            vector_biases.bias,
            user_id,
        )


async def write_bulk_user_vector_bias(pool: asyncpg.Pool, chunk: list[tuple[VectorBias, int]]):
    async with pool.acquire() as connection:
        row_generator = ((x[0].vector, x[0].bias, x[1]) for x in chunk)
        await connection.executemany(
            "UPDATE users SET vector = $1, bias = $2 WHERE user_id = $3", row_generator
        )


async def get_user_vector_bias(pool: asyncpg.Pool, user_id: int) -> Optional[VectorBias]:
    async with pool.acquire() as connection:
        row = await connection.fetchrow(
            "SELECT vector, bias FROM users WHERE user_id = $1", user_id
        )
    vector = row.get("vector")
    bias = row.get("bias")
    if vector is None or bias is None:
        return None
    return VectorBias(vector=vector, bias=bias)


async def delete_all_user_vector_bias(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as connection:
        await connection.execute("UPDATE users SET vector = NULL, bias = NULL")
