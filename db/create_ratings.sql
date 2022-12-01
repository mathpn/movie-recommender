CREATE TABLE IF NOT EXISTS ratings (
    user_id INTEGER,
    movie_id INTEGER,
    rating INTEGER,
    rating_timestamp TIMESTAMP,
    PRIMARY KEY(user_id, movie_id)
);