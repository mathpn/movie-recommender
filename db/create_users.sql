CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    username TEXT,
    vector REAL[],
    bias REAL,
    CONSTRAINT unique_username UNIQUE (username)
);