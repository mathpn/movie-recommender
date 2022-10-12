CREATE TABLE IF NOT EXISTS ratings (
    user_id integer,
    movie_id integer,
    rating integer,
    rating_timestamp timestamp,
    PRIMARY KEY(user_id, movie_id)
);