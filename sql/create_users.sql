CREATE TABLE IF NOT EXISTS users (
    user_id serial PRIMARY KEY,
    username text,
    CONSTRAINT unique_username UNIQUE (username)
);