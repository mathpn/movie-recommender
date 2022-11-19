from app.lookup import create_keyword_searcher
from app.models import KeywordFields

MOVIE_1 = KeywordFields(1, ["action"], ["foo", "bar"], ["ping", "pong"], "foobar")
MOVIE_2 = KeywordFields(2, ["drama"], ["f#o", "b@r"], ["ping", "pong"], "f#ob@r")
MOVIE_3 = KeywordFields(3, ["adventure"], ["something", "else"], ["pang", "pung"], "foobar")


def test_keyword_searcher_match_order():
    searcher = create_keyword_searcher([MOVIE_1, MOVIE_2, MOVIE_3])
    result = searcher.search_by_movie(2, k=3)
    movie_idx, scores = result
    assert movie_idx == [1, 3]
    assert sorted(scores, reverse=True) == list(scores)


def test_keyword_searcher_invalid_movie():
    searcher = create_keyword_searcher([MOVIE_1, MOVIE_2, MOVIE_3])
    result = searcher.search_by_movie(999, k=3)
    movie_idx, scores = result
    assert movie_idx == []
    assert scores == []


def test_keyword_searcher_k():
    searcher = create_keyword_searcher([MOVIE_1, MOVIE_2, MOVIE_3])
    result = searcher.search_by_movie(2, k=2)
    movie_idx, _ = result
    assert movie_idx == [1, 3]


def test_keyword_searcher_allowed_movies():
    searcher = create_keyword_searcher([MOVIE_1, MOVIE_2, MOVIE_3])
    result = searcher.search_by_movie(1, k=3, allowed_movie_ids=[2, 3])
    movie_idx, _ = result
    assert movie_idx == [3]


def test_keyword_searcher_large_k():
    searcher = create_keyword_searcher([MOVIE_1, MOVIE_2, MOVIE_3])
    result = searcher.search_by_movie(1, k=100)
    movie_idx, _ = result
    assert movie_idx == [2, 3]
