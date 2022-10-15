CREATE TABLE IF NOT EXISTS movies (
    movie_id INTEGER PRIMARY KEY,
    movie_title TEXT,
    movie_cast TEXT[],
    director TEXT,
    keywords TEXT[],
    genres TEXT[],
    popularity REAL,
    vote_average REAL,
    vote_count INTEGER,
    vector REAL[],
    bias REAL
);