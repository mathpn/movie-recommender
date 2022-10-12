from app.lookup import create_keyword_searcher
from app.models import KeywordFields


def test_keyword_searcher_match_order():
    movie_1 = KeywordFields(1, ["foo", "bar"], ["ping", "pong"], "foobar")
    movie_2 = KeywordFields(2, ["f#o", "b@r"], ["ping", "pong"], "f#ob@r")
    movie_3 = KeywordFields(3, ["something", "else"], ["pang", "pung"], "foobar")
    searcher = create_keyword_searcher([movie_1, movie_2, movie_3])
    result = searcher.search_by_movie(2, k=3)
    movie_idx, scores = zip(*result)
    assert movie_idx == (2, 1, 3)
    assert scores[0] - 1.0 < 1e-4
    assert sorted(scores, reverse=True) == list(scores)


def test_keyword_searcher_k():
    movie_1 = KeywordFields(1, ["foo", "bar"], ["ping", "pong"], "foobar")
    movie_2 = KeywordFields(2, ["f#o", "b@r"], ["ping", "pong"], "f#ob@r")
    movie_3 = KeywordFields(3, ["something", "else"], ["pang", "pung"], "foobar")
    searcher = create_keyword_searcher([movie_1, movie_2, movie_3])
    result = searcher.search_by_movie(2, k=2)
    movie_idx, _ = zip(*result)
    assert movie_idx == (2, 1)


def test_keyword_searcher_allowed_movies():
    movie_1 = KeywordFields(1, ["foo", "bar"], ["ping", "pong"], "foobar")
    movie_2 = KeywordFields(2, ["f#o", "b@r"], ["ping", "pong"], "f#ob@r")
    movie_3 = KeywordFields(3, ["something", "else"], ["pang", "pung"], "foobar")
    searcher = create_keyword_searcher([movie_1, movie_2, movie_3])
    result = searcher.search_by_movie(1, k=3, allowed_movie_ids=[2, 3])
    movie_idx, _ = zip(*result)
    assert movie_idx == (2, 3)


def test_keyword_searcher_large_k():
    movie_1 = KeywordFields(1, ["foo", "bar"], ["ping", "pong"], "foobar")
    movie_2 = KeywordFields(2, ["f#o", "b@r"], ["ping", "pong"], "f#ob@r")
    movie_3 = KeywordFields(3, ["something", "else"], ["pang", "pung"], "foobar")
    searcher = create_keyword_searcher([movie_1, movie_2, movie_3])
    result = searcher.search_by_movie(1, k=100)
    movie_idx, _ = zip(*result)
    assert movie_idx == (1, 2, 3)
