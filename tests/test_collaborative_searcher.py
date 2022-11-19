import random

import numpy as np

from app.ml.kmf import KMFInferece
from app.lookup import collaborative_search


def _create_fake_embedding(length: int = 1000) -> dict[int, np.ndarray]:
    return {i: np.random.randn(100) for i in range(length)}


def _create_fake_bias(length: int = 1000) -> dict[int, float]:
    return {i: random.random() for i in range(length)}


def test_kmf_inference():
    emb = _create_fake_embedding()
    bias = _create_fake_bias()
    kmf_inf = KMFInferece(emb, emb, bias, bias, global_bias=0, max_score=5)
    pred = kmf_inf(1, list(range(1000)))
    assert all(0 <= x <= 5 for x in pred)


def test_kmf_with_global_bias():
    emb = _create_fake_embedding()
    bias = _create_fake_bias()
    kmf_inf = KMFInferece(emb, emb, bias, bias, global_bias=1.0, max_score=5)
    pred = kmf_inf(1, list(range(1000)))
    assert all(0 <= x <= 5 for x in pred)


def test_kmf_perfect_correlation():
    emb = _create_fake_embedding()
    bias = _create_fake_bias()
    kmf_inf = KMFInferece(emb, emb, bias, bias, global_bias=0, max_score=5)
    for i in range(10):
        pred = kmf_inf(i, [i])
        assert pred[0] >= 0.9999


def test_kmf_predict_movie():
    emb = _create_fake_embedding()
    bias = _create_fake_bias()
    kmf_inf = KMFInferece(emb, emb, bias, bias, global_bias=1.0, max_score=5)
    pred = kmf_inf.predict_movie(10, 42)
    assert isinstance(pred, float)
    assert 0 <= pred <= 5


def test_collaborative_search():
    emb = _create_fake_embedding()
    bias = _create_fake_bias()
    kmf_inf = KMFInferece(emb, emb, bias, bias, global_bias=1.0, max_score=5)
    movie_ids, scores = collaborative_search(kmf_inf, 42, k=5)
    # embeddings are identical, thus 42 (user) <-> 42 (movie) is a perfect score
    assert 42 in movie_ids
    assert all(0 <= x <= 5 for x in scores)
    assert len(movie_ids) == 5
    assert len(scores) == 5


def test_collaborative_search_allowed_movies():
    emb = _create_fake_embedding()
    bias = _create_fake_bias()
    kmf_inf = KMFInferece(emb, emb, bias, bias, global_bias=1.0, max_score=5)
    allowed_movies = [11, 12, 99, 101]
    movie_ids, _ = collaborative_search(kmf_inf, 42, k=5, allowed_movies=allowed_movies)
    assert sorted(movie_ids) == allowed_movies


def test_collaborative_search_k():
    emb = _create_fake_embedding()
    bias = _create_fake_bias()
    kmf_inf = KMFInferece(emb, emb, bias, bias, global_bias=1.0, max_score=5)
    movie_ids, _ = collaborative_search(kmf_inf, 42, k=10)
    assert len(movie_ids) == 10
