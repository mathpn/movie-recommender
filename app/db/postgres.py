"""
Functions to interact with PostgreSQL database.
"""

from typing import Optional, Union

import asyncpg

from app.models import MovieMetadata, Rating


async def create_metadata_table(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as connection:
        await connection.execute(open("./sql/create_metadata.sql", "r").read())


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
            INSERT INTO metadata (
                movie_id, movie_title, movie_cast,
                director, keywords, genres, popularity,
                vote_average, vote_count)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT DO NOTHING
        """,
            *row,
        )


async def get_movie_metadata(pool: asyncpg.Pool, movie_id: int) -> MovieMetadata:
    async with pool.acquire() as connection:
        result = await connection.fetchrow(
            """
                SELECT 
                    movie_id, movie_title, movie_cast,
                    director, keywords, genres, popularity,
                    vote_average, vote_count
                FROM metadata WHERE movie_id = $1
            """,
            movie_id,
        )
    return MovieMetadata(**result)


async def get_all_movies_genres(pool: asyncpg.Pool) -> tuple[tuple[int], tuple[list[str]]]:
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            "SELECT movie_id, genres FROM metadata ORDER BY movie_id"
        )
    return tuple(zip(*[tuple(row) for row in rows]))


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


async def get_user_movie_rating(pool: asyncpg.Pool, user_id: int, movie_id: int) -> Optional[int]:
    async with pool.acquire() as connection:
        return await connection.fetchval(
            "SELECT rating FROM ratings WHERE user_id = $1 AND movie_id = $2", user_id, movie_id
        )


async def create_users_table(pool: asyncpg.Pool):
    async with pool.acquire() as connection:
        await connection.execute(open("./sql/create_users.sql", "r").read())


async def insert_user(pool: asyncpg.Pool, username: str) -> None:
    async with pool.acquire() as connection:
        await connection.execute("INSERT INTO users (username) VALUES ($1)", username)


async def get_user_id(pool: asyncpg.Pool, username: str) -> Optional[int]:
    async with pool.acquire() as connection:
        return await connection.fetchval(
            "SELECT user_id FROM users WHERE username = $1",
            username
        )
