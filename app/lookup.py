"""
Classes and data structures used to search for similar movies.
"""

import numpy as np


def _hamming_distance(a: int, b: np.ndarray, max_bits: int) -> np.ndarray:
    r = (1 << np.arange(max_bits)).reshape(1, -1)
    b = b.reshape(-1, 1)
    return np.count_nonzero((a & r) != (b & r), axis=1)


class GenreSearcher:
    def __init__(self, movie_ids: list[int], encoded_genres: list[int], n_genres: int):
        self.encoded_genres = np.array(encoded_genres)
        self.movie_ids = np.array(movie_ids)
        self.movie2genres = dict(zip(movie_ids, encoded_genres))
        self.n_genres = n_genres

    def search_by_movie(self, movie_id: int, k: int) -> list[int]:
        movie_genres = self.movie2genres.get(movie_id)
        if movie_genres is None:
            raise ValueError("movie ID not fonud")
        dists = _hamming_distance(movie_genres, self.encoded_genres, self.n_genres)
        top_idx = np.argsort(dists)
        # keep only movies with at least one common genre
        top_idx = top_idx[dists[top_idx] < self.n_genres][:k]
        return self.movie_ids[top_idx].tolist()

    def __len__(self):
        return len(self.movie_ids)


def create_genre_searcher(movie_ids: tuple[int], genres: tuple[list[str]]) -> GenreSearcher:
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
