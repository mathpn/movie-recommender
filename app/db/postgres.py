"""
Functions to interact with PostgreSQL database.
"""

import asyncpg

from app.models import MovieMetadata


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


async def insert_movie_metadata(pool: asyncpg.Pool, metadata: MovieMetadata):
    async with pool.acquire() as connection:
        row = _movie_metadata_to_tuple(metadata)
        try:
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
        except Exception as exc:
            print(exc)
            print(row)
            print()


async def get_movie_metadata(pool: asyncpg.Pool, movie_id: int) -> asyncpg.Record:
    async with pool.acquire() as connection:
        return await connection.fetchval("SELECT * FROM metadata WHERE movie_id = $1", movie_id)
