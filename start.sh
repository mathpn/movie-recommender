export POSTGRES_URI='postgresql://postgres:postgres@localhost:5401/movies'
uvicorn main:app --reload
