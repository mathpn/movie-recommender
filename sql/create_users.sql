CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    username TEXT,
    vector REAL[],
    CONSTRAINT unique_username UNIQUE (username)
);