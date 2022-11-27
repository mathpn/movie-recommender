"""
Classes, functions and data structures used to search for similar movies.
"""

from typing import Optional

import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.db.postgres import get_movie_titles
from app.logger import logger
from app.ml.kmf import KMFInferece
from app.models import KeywordFields


def _hamming_distance(a: int, b: np.ndarray, max_bits: int) -> np.ndarray:
    r = (1 << np.arange(max_bits)).reshape(1, -1)
    b = b.reshape(-1, 1)
    return np.count_nonzero((a & r) != (b & r), axis=1)


class GenreSearcher:
    def __init__(self, movie_ids: list[int], encoded_genres: list[int], n_genres: int):
        self.encoded_genres = np.array(encoded_genres)
        self.movie_ids = list(movie_ids)
        self.movie2genres = dict(zip(movie_ids, list(encoded_genres)))
        self.n_genres = n_genres

    def search_by_movie(self, movie_id: int, k: Optional[int] = None) -> Optional[list[int]]:
        movie_genres = self.movie2genres.get(movie_id)
        if movie_genres is None:
            return None

        dists = _hamming_distance(movie_genres, self.encoded_genres, self.n_genres)
        top_idx = np.argsort(dists)
        # keep only movies with at least one common genre
        top_idx = top_idx[dists[top_idx] < self.n_genres]
        if k is not None:
            top_idx = top_idx[:k]
        return [self.movie_ids[idx] for idx in top_idx]

    def __len__(self):
        return len(self.movie_ids)


def create_genre_searcher(movie_ids: list[int], genres: list[list[str]]) -> GenreSearcher:
    all_genres = set()
    for genres_ in genres:
        if not genres_:
            continue
        all_genres.update(genres_)
    genre_to_int = {genre: 2**i for i, genre in enumerate(all_genres)}
    encoded_genres = list(map(lambda genre_list: sum(genre_to_int[x] for x in genre_list), genres))
    return GenreSearcher(
        movie_ids=movie_ids, encoded_genres=encoded_genres, n_genres=len(all_genres)
    )


class KeywordSearcher:
    def __init__(self, movie_ids: list[int], sparse_keyword_count: scipy.sparse.csr.csr_matrix):
        self.movie_ids = list(movie_ids)
        self.sparse_keywords = sparse_keyword_count.copy()
        self.movie_2_internal_id = {movie_id: i for i, movie_id in enumerate(movie_ids)}

    def search_by_movie(
        self, movie_id: int, k: int, allowed_movie_ids: Optional[list[int]] = None
    ) -> tuple[list[int], list[float]]:
        internal_id = self.movie_2_internal_id.get(movie_id)
        if internal_id is None:
            return [], []

        movie_sparse_row = self.sparse_keywords[internal_id, :]
        if allowed_movie_ids is not None:
            allowed_movies = np.array([self.movie_2_internal_id[idx] for idx in allowed_movie_ids])
            allowed_movie_rows = self.sparse_keywords[allowed_movies, :]
        else:
            allowed_movie_ids = self.movie_ids
            allowed_movie_rows = self.sparse_keywords
        sim = cosine_similarity(movie_sparse_row, allowed_movie_rows).ravel()
        # exclude the input movie
        top_idx = np.argsort(-sim)[1 : k + 1]
        top_k_sim = sim[top_idx]
        top_k_sim /= top_k_sim.max()
        return [allowed_movie_ids[idx] for idx in top_idx], [sim[idx] for idx in top_idx]


def _remove_spaces(string) -> str:
    if string is None:
        return ""
    return "_".join(string.lower().split(" "))


def create_keyword_searcher(keyword_fields: list[KeywordFields]) -> KeywordSearcher:
    count = CountVectorizer(stop_words="english")
    soups = []
    movie_ids = []
    for keywords in keyword_fields:
        movie_ids.append(keywords.movie_id)
        soup = [
            *map(_remove_spaces, keywords.keywords),
            *map(_remove_spaces, keywords.cast),
            *map(_remove_spaces, keywords.genres),
            _remove_spaces(keywords.director),
        ]
        soups.append(" ".join(soup))
    sparse_keywords = count.fit_transform(soups)
    return KeywordSearcher(movie_ids, sparse_keywords)


def collaborative_search(
    kmf_inference: KMFInferece,
    user_id: int,
    k: int,
    allowed_movies: Optional[list[int]] = None
) -> tuple[list[int], list[float]]:
    """Search recommended movies for a given user."""
    allowed_movies = allowed_movies if allowed_movies is not None else kmf_inference.movie_ids
    scores = kmf_inference(user_id, allowed_movies)

    if scores is None:
        return [], []

    top_idx = np.argsort(-scores)[:k]
    return [allowed_movies[idx] for idx in top_idx], [scores[idx] for idx in top_idx]


async def recommend(
    movie_id: int,
    user_id: int,
    genre_searcher: GenreSearcher,
    keyword_searcher: KeywordSearcher,
    kmf_inference: KMFInferece,
    k: int = 6,
    pool=None,  # TODO remove
) -> list[int]:
    recos_by_genre = genre_searcher.search_by_movie(movie_id=movie_id, k=2000)

    # keyword-based recommendations
    recos_by_kw, _ = keyword_searcher.search_by_movie(
        movie_id=movie_id, k=k, allowed_movie_ids=recos_by_genre
    )

    # user-based recommendations    
    if kmf_inference is not None:
        recos_by_user, _ = collaborative_search(
            kmf_inference, user_id, k=k, allowed_movies=recos_by_genre
        )
        recos_by_user = [reco for reco in recos_by_user if reco not in set(recos_by_kw)]
    else:
        recos_by_user = []

    logger.debug(f"{await get_movie_titles(pool, recos_by_kw) = }\n{await get_movie_titles(pool, recos_by_user) = }") # TODO remove
    n_user_recos = min(k // 2, len(recos_by_user)) + max(0, k // 2 - len(recos_by_kw))
    merged_recos = recos_by_kw[:(k - n_user_recos)] + recos_by_user[:n_user_recos]
    logger.info(f"found {len(merged_recos)} recommendations for user {user_id}")
    return merged_recos
