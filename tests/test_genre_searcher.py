from app.lookup import create_genre_searcher


def test_genre_searcher_match_self():
    movie_ids = [1, 2, 3]
    genres = [["foo", "bar"], ["foo", "ping"], ["ping", "pong"]]
    searcher = create_genre_searcher(movie_ids, genres)
    result = searcher.search_by_movie(2, k=1)
    assert result[0] == 2


def test_genre_searcher_invalid_movie():
    movie_ids = [1, 2, 3]
    genres = [["foo", "bar"], ["foo", "ping"], ["ping", "pong"]]
    searcher = create_genre_searcher(movie_ids, genres)
    result = searcher.search_by_movie(999, k=1)
    assert result is None


def test_genre_searcher_match_two():
    movie_ids = [1, 2, 3]
    genres = [["foo", "bar"], ["foo", "ping"], ["ping", "pong"]]
    searcher = create_genre_searcher(movie_ids, genres)
    result = searcher.search_by_movie(2, k=2)
    assert result == [2, 1]


def test_genre_searcher_no_match_without_overlap():
    movie_ids = [1, 2, 3]
    genres = [["foo", "bar"], ["foo", "ping"], ["ping", "pong"]]
    searcher = create_genre_searcher(movie_ids, genres)
    result = searcher.search_by_movie(1, k=3)
    assert 3 not in result


def test_genre_searcher_k_larger_than_len():
    movie_ids = [1, 2, 3]
    genres = [["foo", "bar"], ["foo", "ping"], ["ping", "pong"]]
    searcher = create_genre_searcher(movie_ids, genres)
    result = searcher.search_by_movie(1, k=1000)
    assert result == [1, 2]
