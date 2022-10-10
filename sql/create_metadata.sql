CREATE TABLE IF NOT EXISTS metadata (
    movie_id integer PRIMARY KEY,
    movie_title text,
    movie_cast text[],
    director text,
    keywords text[],
    genres text[],
    popularity double precision,
    vote_average double precision,
    vote_count integer
);