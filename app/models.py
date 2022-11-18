"""
Data models and types.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import NamedTuple


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


class Rating(NamedTuple):
    user_id: int
    movie_id: int
    rating: int
    timestamp: datetime


class VectorBias(NamedTuple):
    vector: list[float]
    bias: float
    entry_id: int


class KeywordFields(NamedTuple):
    movie_id: int
    genres: list[str]
    keywords: list[str]
    cast: list[str]
    director: str
