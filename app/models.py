"""
Data models and types.
"""

from dataclasses import dataclass


@dataclass
class MovieMetadata:
    movie_id: int
    movie_title: str
    movie_cast: list[str]
    director: str
    keywords: list[str]
    genres: list[str]
    popularity: float
    vote_average: float
    vote_count: int
